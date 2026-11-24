from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .checkpoint import load_model_state_only
from .config import Config
from .data import build_dataloader, move_batch
from .metrics import PredictionAccumulator
from .model import LogMelProjectedMultitaskECAPA
from .utils import atomic_json, utc_now


PREDICTION_COLUMNS = [
    "utt_id",
    "filepath",
    "speaker_id",
    "oracle_language",
    "true_language_index",
    "predicted_language_index",
    "true_dialect_index",
    "predicted_dialect_index",
    "true_gender_index",
    "predicted_gender_index",
    "crop_start_frame",
    "source_num_frames",
]


def evaluate_loader(
    *,
    config: Config,
    model: LogMelProjectedMultitaskECAPA,
    loader,
    device: torch.device,
    description: str,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    from speech_mtl.common.training.losses import compute_multitask_loss
    from speech_mtl.exp61.metrics import EpochMetricAccumulator

    model.eval()
    accumulator = PredictionAccumulator(dict(config.dialect_class_counts))
    dialect_helper = EpochMetricAccumulator()
    loss_sum = 0.0
    examples = 0
    real_frames = 0
    padded_frames = 0
    started = time.monotonic()

    handle = None
    writer = None
    temporary_predictions = None
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_predictions = predictions_path.with_name(
            f".{predictions_path.name}.tmp"
        )
        handle = temporary_predictions.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()

    try:
        iterator = tqdm(loader, total=len(loader), desc=description, dynamic_ncols=True, leave=True)
        with torch.inference_mode():
            for raw_batch in iterator:
                batch = move_batch(raw_batch, device)
                batch_size = int(batch["features"].shape[0])
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(config.use_amp and device.type == "cuda"),
                ):
                    output = model(
                        features=batch["features"],
                        relative_lengths=batch["relative_lengths"],
                        oracle_languages=batch["language"],
                        valid_mask=batch["valid_mask"],
                    )
                    loss_output = compute_multitask_loss(
                        model=model.base,
                        output=output,
                        language_targets=batch["lid"],
                        dialect_targets=batch["dialect"],
                        gender_targets=batch["gender"],
                    )
                dialect_predictions = dialect_helper.update(output=output, batch=batch)
                accumulator.update(
                    output=output,
                    batch=batch,
                    dialect_predictions=dialect_predictions,
                )
                loss_value = float(loss_output.total.detach().cpu().item())
                if not torch.isfinite(loss_output.total.detach()):
                    raise RuntimeError("Non-finite evaluation loss")
                loss_sum += loss_value * batch_size

                if writer is not None:
                    language_predictions = torch.argmax(output.language_logits, dim=1).detach().cpu()
                    gender_predictions = torch.argmax(output.gender_logits, dim=1).detach().cpu()
                    dialect_cpu = dialect_predictions.detach().cpu()
                    for index in range(batch_size):
                        writer.writerow(
                            {
                                "utt_id": raw_batch["utt_id"][index],
                                "filepath": raw_batch["filepath"][index],
                                "speaker_id": raw_batch["speaker_id"][index],
                                "oracle_language": raw_batch["language"][index],
                                "true_language_index": int(raw_batch["lid"][index].item()),
                                "predicted_language_index": int(language_predictions[index].item()),
                                "true_dialect_index": int(raw_batch["dialect"][index].item()),
                                "predicted_dialect_index": int(dialect_cpu[index].item()),
                                "true_gender_index": int(raw_batch["gender"][index].item()),
                                "predicted_gender_index": int(gender_predictions[index].item()),
                                "crop_start_frame": int(raw_batch["crop_start_frame"][index].item()),
                                "source_num_frames": int(raw_batch["source_num_frames"][index].item()),
                            }
                        )

                examples += batch_size
                real_frames += int(raw_batch["real_frames"])
                padded_frames += int(raw_batch["padded_frames"])
                iterator.set_postfix(examples=examples)
    finally:
        if handle is not None:
            handle.flush()
            handle.close()

    if predictions_path is not None:
        temporary_predictions.replace(predictions_path)

    elapsed = time.monotonic() - started
    metrics = accumulator.finalize()
    metrics.update(
        {
            "mean_total_loss": loss_sum / examples,
            "real_frames": real_frames,
            "padded_frames": padded_frames,
            "padding_ratio": 1.0 - real_frames / padded_frames if padded_frames else 0.0,
            "elapsed_seconds": elapsed,
            "examples_per_second": examples / elapsed,
        }
    )
    return metrics


def result_directory(
    config: Config,
    *,
    train_domain: str,
    train_duration: str,
    input_domain: str,
    split: str,
    test_duration: str,
) -> Path:
    return (
        config.path("results_root")
        / train_domain
        / train_duration
        / f"input_{input_domain}"
        / f"{split}_{test_duration}"
    )


def evaluate_condition(
    config: Config,
    *,
    train_domain: str,
    train_duration: str,
    input_domain: str,
    split: str,
    test_duration: str,
    device: torch.device,
) -> dict[str, Any]:
    output_root = result_directory(
        config,
        train_domain=train_domain,
        train_duration=train_duration,
        input_domain=input_domain,
        split=split,
        test_duration=test_duration,
    )
    marker_path = output_root / "EVALUATION_COMPLETE.json"
    metrics_path = output_root / "metrics.json"
    predictions_path = output_root / "predictions.csv" if split == "test" else None

    if marker_path.is_file() and metrics_path.is_file():
        print(
            "EVALUATION_SKIP_COMPLETE "
            f"train={train_domain}/{train_duration} input={input_domain} "
            f"split={split} duration={test_duration}",
            flush=True,
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_root = config.path("run_root") / train_domain / train_duration
    checkpoint_path = run_root / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    model = LogMelProjectedMultitaskECAPA(
        int(config.feature_dimension), int(config.projected_dimension)
    ).to(device)
    checkpoint_metadata = load_model_state_only(checkpoint_path, model)
    model.eval()
    model.requires_grad_(False)

    loader, sampler, dataset = build_dataloader(
        config,
        split=split,
        domain=input_domain,
        duration_name=test_duration,
        training=False,
    )
    sampler.set_epoch(0)
    dataset.set_epoch(0)
    metrics = evaluate_loader(
        config=config,
        model=model,
        loader=loader,
        device=device,
        description=(
            f"eval {train_domain}/{train_duration} -> "
            f"{input_domain}/{split}/{test_duration}"
        ),
        predictions_path=predictions_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "EVALUATION_COMPLETE",
        "created_utc": utc_now(),
        "train_domain": train_domain,
        "train_duration": train_duration,
        "input_domain": input_domain,
        "split": split,
        "test_duration": test_duration,
        "checkpoint": str(checkpoint_path),
        "checkpoint_training_state": checkpoint_metadata["training_state"],
        **metrics,
    }
    atomic_json(metrics_path, payload)
    atomic_json(
        marker_path,
        {
            "status": "EVALUATION_COMPLETE",
            "created_utc": utc_now(),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path) if predictions_path else None,
            "examples": int(metrics["examples"]),
        },
    )
    print(
        "EXP69_EVALUATION_PASS "
        f"train={train_domain}/{train_duration} input={input_domain} "
        f"split={split} duration={test_duration}",
        flush=True,
    )
    return payload

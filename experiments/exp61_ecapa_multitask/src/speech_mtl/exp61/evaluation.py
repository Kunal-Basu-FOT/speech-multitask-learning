from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from speech_mtl.common.evaluation.predictions import (
    PredictionWriter,
)
from speech_mtl.common.training.losses import (
    compute_multitask_loss,
)

from .metrics import EpochMetricAccumulator


class _NullPredictionWriter:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return None


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved = dict(batch)

    for key in (
        "features",
        "relative_lengths",
        "valid_mask",
        "lid",
        "dialect",
        "gender",
    ):
        moved[key] = batch[key].to(
            device=device,
            non_blocking=True,
        )

    return moved


def evaluate_split_with_loss(
    *,
    model,
    dataloader,
    device: torch.device,
    use_amp: bool,
    predictions_path: Path | str | None = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    model.eval()

    accumulator = EpochMetricAccumulator()
    total_loss_sum = 0.0
    raw_loss_sums = {
        "language": 0.0,
        "dialect": 0.0,
        "gender": 0.0,
    }
    examples = 0
    real_frames = 0
    padded_frames = 0
    started = time.monotonic()

    context = (
        PredictionWriter(predictions_path)
        if predictions_path is not None
        else _NullPredictionWriter()
    )

    iterator = tqdm(
        dataloader,
        total=len(dataloader),
        desc="evaluation",
        dynamic_ncols=True,
        disable=not show_progress_bar,
        leave=True,
    )

    with context as writer:
        with torch.inference_mode():
            for raw_batch in iterator:
                batch = _move_batch(raw_batch, device)
                batch_size = int(batch["features"].shape[0])

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=bool(
                        use_amp and device.type == "cuda"
                    ),
                ):
                    output = model(
                        features=batch["features"],
                        relative_lengths=batch[
                            "relative_lengths"
                        ],
                        oracle_languages=batch["language"],
                        valid_mask=batch["valid_mask"],
                    )
                    loss_output = compute_multitask_loss(
                        model=model,
                        output=output,
                        language_targets=batch["lid"],
                        dialect_targets=batch["dialect"],
                        gender_targets=batch["gender"],
                    )

                dialect_predictions = accumulator.update(
                    output=output,
                    batch=batch,
                )

                loss_value = float(
                    loss_output.total.detach().cpu().item()
                )
                total_loss_sum += loss_value * batch_size

                for task, task_loss in (
                    loss_output.raw_losses.items()
                ):
                    raw_loss_sums[task] += (
                        float(
                            task_loss.detach().cpu().item()
                        )
                        * batch_size
                    )

                if writer is not None:
                    language_predictions = torch.argmax(
                        output.language_logits,
                        dim=1,
                    )
                    gender_predictions = torch.argmax(
                        output.gender_logits,
                        dim=1,
                    )

                    rows = []

                    for index in range(batch_size):
                        rows.append(
                            {
                                "utt_id": raw_batch["utt_id"][
                                    index
                                ],
                                "filepath": raw_batch[
                                    "filepath"
                                ][index],
                                "oracle_language": raw_batch[
                                    "language"
                                ][index],
                                "true_language_index": int(
                                    batch["lid"][index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                                "predicted_language_index": int(
                                    language_predictions[index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                                "true_dialect_index": int(
                                    batch["dialect"][index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                                "predicted_dialect_index": int(
                                    dialect_predictions[index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                                "true_gender_index": int(
                                    batch["gender"][index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                                "predicted_gender_index": int(
                                    gender_predictions[index]
                                    .detach()
                                    .cpu()
                                    .item()
                                ),
                            }
                        )

                    writer.write_batch(rows)

                examples += batch_size
                real_frames += int(raw_batch["real_frames"])
                padded_frames += int(
                    raw_batch["padded_frames"]
                )
                iterator.set_postfix(examples=examples)

    summary, confusion_matrices = accumulator.finalize()
    elapsed = time.monotonic() - started

    return {
        "examples": examples,
        "mean_total_loss": total_loss_sum / examples,
        "mean_language_loss": (
            raw_loss_sums["language"] / examples
        ),
        "mean_dialect_loss": (
            raw_loss_sums["dialect"] / examples
        ),
        "mean_gender_loss": (
            raw_loss_sums["gender"] / examples
        ),
        "summary": summary,
        "confusion_matrices": confusion_matrices,
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_ratio": (
            1.0 - real_frames / padded_frames
            if padded_frames
            else 0.0
        ),
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "predictions_path": (
            str(Path(predictions_path))
            if predictions_path is not None
            else None
        ),
    }

from __future__ import annotations

from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from .checkpoint import load_checkpoint
from .data import build_dataset, build_loader
from .gradient_methods import GradNormController
from .labels import LabelSpace
from .metrics import PredictionAccumulator
from .model import CONDITIONS, TASKS, HardSharingMTLModel
from .objectives import compute_raw_losses, uncertainty_total, weighted_total
from .utils import atomic_json_dump


def _to_device(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in (
        "features", "frame_lengths", "relative_lengths", "valid_mask",
        "lid", "dialect", "gender", "dataset_index",
    ):
        result[key] = result[key].to(device, non_blocking=True)
    return result


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        sums,
        out=np.zeros_like(matrix, dtype=float),
        where=sums > 0,
    )


def _write_matrix_csv(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def _plot_confusion(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    values = _row_normalize(matrix) * 100.0
    fig, axis = plt.subplots(
        figsize=(max(5.0, len(labels) * 0.8), max(4.0, len(labels) * 0.7))
    )
    image = axis.imshow(values, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def evaluate_condition(
    cfg,
    labels: LabelSpace,
    train_domain: str,
    test_domain: str,
    condition: str,
    device: torch.device,
) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    checkpoint_path = (
        cfg.paths.run_root / train_domain / condition / "checkpoints" / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_root = (
        cfg.paths.results_root / train_domain / condition / f"test_{test_domain}"
    )
    completion_path = output_root / "EVALUATION_COMPLETE.json"
    if completion_path.is_file() and (output_root / "metrics.json").is_file():
        print(
            f"EVALUATION_ALREADY_COMPLETE condition={condition} "
            f"train={train_domain} test={test_domain}",
            flush=True,
        )
        return output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model = HardSharingMTLModel(cfg.model, labels).to(device)
    model.configure_condition(condition)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    gradnorm_controller = None
    if condition == "gradnorm":
        gradnorm_controller = GradNormController(cfg.training.gradnorm_alpha).to(device)
        state = checkpoint.get("gradnorm_controller")
        if state is None:
            raise RuntimeError("GradNorm checkpoint has no controller state")
        gradnorm_controller.load_state_dict(state)
        gradnorm_controller.eval()

    dataset = build_dataset(cfg, test_domain, "test", labels)
    loader, _ = build_loader(cfg, dataset, train=False)
    accumulator = PredictionAccumulator(labels)
    weighted_loss = 0.0
    examples = 0
    start_time = time.time()
    amp_enabled = bool(cfg.training.amp and device.type == "cuda")

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval {condition} {train_domain}->{test_domain}"):
            batch = _to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(batch)
                raw_losses = compute_raw_losses(model, batch, output)
                if condition == "gradnorm":
                    assert gradnorm_controller is not None
                    total = weighted_total(raw_losses, gradnorm_controller.weights().detach())
                else:
                    total = uncertainty_total(model, raw_losses)
            batch_size = len(batch["utt_id"])
            weighted_loss += float(total.item()) * batch_size
            examples += batch_size
            accumulator.update(batch, output)

    metrics, matrices = accumulator.finalize()
    metrics.update(
        {
            "loss": weighted_loss / max(1, examples),
            "examples": examples,
            "condition": condition,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "matched": train_domain == test_domain,
            "best_epoch": int(checkpoint["best_epoch"]),
            "selection_score": float(checkpoint["best_score"]),
            "elapsed_seconds": time.time() - start_time,
            "parameter_counts": checkpoint.get("parameter_counts", model.parameter_counts),
            "dialect_routing": "ground_truth_language",
            "gradient_method": condition,
            "gradnorm_final_weights": (
                None
                if gradnorm_controller is None
                else {
                    task: value
                    for task, value in zip(
                        TASKS,
                        gradnorm_controller.normalized_weights_list(),
                        strict=True,
                    )
                }
            ),
        }
    )
    atomic_json_dump(metrics, output_root / "metrics.json")

    if not accumulator.rows:
        raise RuntimeError("Evaluation produced no predictions")
    with (output_root / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accumulator.rows[0].keys()))
        writer.writeheader()
        writer.writerows(accumulator.rows)

    confusion_root = output_root / "confusion"
    label_sets = {
        "lid": list(labels.languages),
        "gender": list(labels.gender_values),
        **{
            f"dialect_{language}": list(labels.dialect_values_by_language[language])
            for language in labels.languages
        },
    }
    for name, matrix in matrices.items():
        matrix_labels = label_sets[name]
        _write_matrix_csv(matrix, matrix_labels, confusion_root / f"{name}_counts.csv")
        _write_matrix_csv(
            _row_normalize(matrix),
            matrix_labels,
            confusion_root / f"{name}_row_normalized.csv",
        )
        if name in {"lid", "gender", "dialect_bn"}:
            _plot_confusion(
                matrix,
                matrix_labels,
                f"EXP 6.6 {condition} {train_domain}->{test_domain}: {name}",
                confusion_root / f"{name}.png",
            )

    atomic_json_dump(
        {
            "status": "complete",
            "condition": condition,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "matched": train_domain == test_domain,
            "timestamp": time.time(),
        },
        completion_path,
    )
    print(
        f"EXP66_EVALUATION_PASS condition={condition} "
        f"train={train_domain} test={test_domain}",
        flush=True,
    )
    return output_root

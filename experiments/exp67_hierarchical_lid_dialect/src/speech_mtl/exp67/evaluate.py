from __future__ import annotations

from pathlib import Path
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint
from .data import build_dataset, build_loader
from .losses import compute_losses
from .metrics import PredictionAccumulator
from .model import CONDITION_NAME, HierarchicalLIDDialectModel
from .utils import atomic_json_dump


def _to_device(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in (
        "features",
        "frame_lengths",
        "relative_lengths",
        "valid_mask",
        "lid",
        "dialect",
        "gender",
        "dataset_index",
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


def _plot_confusion(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
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
            axis.text(
                column,
                row,
                f"{values[row, column]:.0f}",
                ha="center",
                va="center",
                fontsize=6,
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_confidence_analysis(analysis: dict, root: Path) -> None:
    atomic_json_dump(analysis, root / "lid_confidence_dialect_analysis.json")
    bins = analysis["bins"]
    with (root / "lid_confidence_dialect_analysis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "label",
            "lower",
            "upper",
            "count",
            "mean_lid_confidence",
            "lid_accuracy",
            "dialect_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bins)

    populated = [row for row in bins if int(row["count"]) > 0]
    if not populated:
        return
    labels = [row["label"] for row in populated]
    lid_values = [row["lid_accuracy"] for row in populated]
    dialect_values = [row["dialect_accuracy"] for row in populated]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    axis.plot(x, lid_values, marker="o", label="LID accuracy")
    axis.plot(x, dialect_values, marker="o", label="Dialect accuracy")
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_xlabel("Maximum LID posterior bin")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("LID confidence versus dialect correctness")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "lid_confidence_vs_dialect_accuracy.png", dpi=300)
    plt.close(fig)


def evaluate_condition(
    cfg,
    labels,
    train_domain: str,
    test_domain: str,
    device: torch.device,
) -> Path:
    checkpoint_path = (
        cfg.paths.run_root
        / train_domain
        / CONDITION_NAME
        / "checkpoints"
        / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_root = (
        cfg.paths.results_root
        / train_domain
        / CONDITION_NAME
        / f"test_{test_domain}"
    )
    completion_path = output_root / "EVALUATION_COMPLETE.json"
    if completion_path.is_file() and (output_root / "metrics.json").is_file():
        print(
            f"EVALUATION_ALREADY_COMPLETE train={train_domain} test={test_domain}",
            flush=True,
        )
        return output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model = HierarchicalLIDDialectModel(cfg.model, labels).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(cfg, test_domain, "test", labels)
    loader, _ = build_loader(cfg, dataset, train=False)
    accumulator = PredictionAccumulator(labels)
    weighted_loss = 0.0
    examples = 0
    start_time = time.time()
    amp_enabled = bool(cfg.training.amp and device.type == "cuda")

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"eval {CONDITION_NAME} {train_domain}->{test_domain}",
        ):
            batch = _to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(batch)
                losses = compute_losses(model, batch, output)
            batch_size = len(batch["utt_id"])
            weighted_loss += float(losses["total"].item()) * batch_size
            examples += batch_size
            accumulator.update(batch, output)

    metrics, matrices = accumulator.finalize()
    metrics.update(
        {
            "loss": weighted_loss / max(1, examples),
            "examples": examples,
            "condition": CONDITION_NAME,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "matched": train_domain == test_domain,
            "best_epoch": int(checkpoint["best_epoch"]),
            "selection_score": float(checkpoint["best_score"]),
            "elapsed_seconds": time.time() - start_time,
            "parameter_counts": checkpoint.get(
                "parameter_counts", model.parameter_counts
            ),
            "dialect_routing": "ground_truth_language",
            "dialect_conditioning": "stop_gradient_predicted_lid_probabilities",
        }
    )
    atomic_json_dump(metrics, output_root / "metrics.json")

    if not accumulator.rows:
        raise RuntimeError("Evaluation produced no predictions")
    prediction_fields = list(accumulator.rows[0].keys())
    with (output_root / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(accumulator.rows)

    confusion_root = output_root / "confusion"
    label_sets = {
        "lid": list(labels.languages),
        "gender": list(labels.gender_values),
        **{
            f"dialect_{language}": list(
                labels.dialect_values_by_language[language]
            )
            for language in labels.languages
        },
    }
    for name, matrix in matrices.items():
        matrix_labels = label_sets[name]
        _write_matrix_csv(
            matrix,
            matrix_labels,
            confusion_root / f"{name}_counts.csv",
        )
        _write_matrix_csv(
            _row_normalize(matrix),
            matrix_labels,
            confusion_root / f"{name}_row_normalized.csv",
        )
        if name in {"lid", "gender", "dialect_bn"}:
            _plot_confusion(
                matrix,
                matrix_labels,
                f"EXP 6.7 {train_domain}->{test_domain}: {name}",
                confusion_root / f"{name}.png",
            )

    _write_confidence_analysis(
        metrics["lid_confidence_dialect_analysis"],
        output_root,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "train_domain": train_domain,
            "test_domain": test_domain,
            "matched": train_domain == test_domain,
            "timestamp": time.time(),
        },
        completion_path,
    )
    print(
        f"EXP67_EVALUATION_PASS train={train_domain} test={test_domain}",
        flush=True,
    )
    return output_root

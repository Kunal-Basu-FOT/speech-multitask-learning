"""Matched and cross-precision evaluation, predictions, and confusion figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
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
from .conditions import Condition
from .config import ExperimentConfig
from .data import build_dataset, build_loader
from .labels import LabelSpace
from .losses import compute_losses
from .metrics import PredictionAccumulator
from .model import ReducedMTLModel
from .utils import atomic_json_dump


def _to_device(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in ("features", "frame_lengths", "relative_lengths", "valid_mask", "lid", "dialect", "gender", "dataset_index"):
        result[key] = result[key].to(device, non_blocking=True)
    return result


def _row_normalized(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)


def _save_matrix_csv(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def _plot_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    dpi: int,
    *,
    normalized: bool = True,
) -> None:
    values = _row_normalized(matrix) * 100.0 if normalized else matrix
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.9), max(4, len(labels) * 0.75)))
    image = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.1f}" if normalized else str(int(values[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row %" if normalized else "Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_three_panel_overview(
    matrices: dict[str, np.ndarray],
    labels: LabelSpace,
    condition: Condition,
    representative_language: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [
        ("lid", list(labels.languages), "Language ID"),
        ("gender", list(labels.gender_values), "Gender"),
        (f"dialect_{representative_language}", list(labels.dialect_values_by_language[representative_language]), f"Dialect ({representative_language})"),
    ]
    for ax, (key, tick_labels, panel_title) in zip(axes, panels, strict=True):
        if key not in matrices:
            ax.axis("off")
            ax.text(0.5, 0.55, panel_title, ha="center", va="center", fontsize=13)
            ax.text(0.5, 0.42, "Task not trained — N/A", ha="center", va="center", fontsize=11)
            continue
        values = _row_normalized(matrices[key]) * 100.0
        image = ax.imshow(values, aspect="auto")
        ax.set_title(panel_title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(tick_labels)), tick_labels)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def evaluate_condition(
    cfg: ExperimentConfig,
    labels: LabelSpace,
    condition: Condition,
    train_domain: str,
    test_domain: str,
    device: torch.device,
) -> Path:
    checkpoint_path = cfg.paths.run_root / train_domain / condition.name / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    output_root = cfg.paths.results_root / train_domain / condition.name / f"test_{test_domain}"
    output_root.mkdir(parents=True, exist_ok=True)
    model = ReducedMTLModel(cfg.model, labels, condition).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_set = build_dataset(cfg, test_domain, "test", labels)
    test_loader, _ = build_loader(cfg, test_set, train=False)
    accumulator = PredictionAccumulator(condition, labels)
    loss_sum = 0.0
    examples = 0
    amp_enabled = bool(cfg.training.amp and device.type == "cuda")
    started = time.time()
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"eval {train_domain}->{test_domain} {condition.name}"):
            batch = _to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(batch)
                losses = compute_losses(model, batch, output)
            batch_size = len(batch["utt_id"])
            loss_sum += float(losses["total"].item()) * batch_size
            examples += batch_size
            accumulator.update(batch, output)
    metrics, matrices = accumulator.finalize()
    metrics.update(
        {
            "loss": loss_sum / max(1, examples),
            "examples": examples,
            "train_domain": train_domain,
            "test_domain": test_domain,
            "condition": condition.name,
            "matched": train_domain == test_domain,
            "checkpoint": str(checkpoint_path),
            "best_epoch": int(checkpoint["best_epoch"]),
            "selection_score": float(checkpoint["best_score"]),
            "elapsed_seconds": time.time() - started,
            "parameter_counts": checkpoint.get("parameter_counts", model.parameter_counts),
        }
    )
    atomic_json_dump(metrics, output_root / "metrics.json")

    prediction_fields = [
        "utt_id", "filepath", "oracle_language", "true_language_index",
        "predicted_language_index", "true_dialect_index", "predicted_dialect_index",
    ]
    with (output_root / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(accumulator.rows)

    confusion_root = output_root / "confusion"
    if "lid" in matrices:
        lid_labels = list(labels.languages)
        _save_matrix_csv(matrices["lid"], lid_labels, confusion_root / "lid_counts.csv")
        _save_matrix_csv(_row_normalized(matrices["lid"]), lid_labels, confusion_root / "lid_row_normalized.csv")
        _plot_matrix(matrices["lid"], lid_labels, "Language ID confusion (row %)", confusion_root / "lid.png", cfg.evaluation.figure_dpi)
    for lang in labels.languages:
        key = f"dialect_{lang}"
        if key not in matrices:
            continue
        local_labels = list(labels.dialect_values_by_language[lang])
        _save_matrix_csv(matrices[key], local_labels, confusion_root / f"dialect_{lang}_counts.csv")
        _save_matrix_csv(_row_normalized(matrices[key]), local_labels, confusion_root / f"dialect_{lang}_row_normalized.csv")
        if cfg.evaluation.save_all_dialect_confusions or lang == cfg.evaluation.representative_dialect_language:
            _plot_matrix(
                matrices[key],
                local_labels,
                f"Dialect confusion — {lang} (row %)",
                confusion_root / f"dialect_{lang}.png",
                cfg.evaluation.figure_dpi,
            )
    _plot_three_panel_overview(
        matrices,
        labels,
        condition,
        cfg.evaluation.representative_dialect_language,
        f"{condition.name}: trained {train_domain}, tested {test_domain}",
        output_root / "confusion_overview_lid_gender_bn.png",
        cfg.evaluation.figure_dpi,
    )
    atomic_json_dump(
        {
            "status": "complete",
            "train_domain": train_domain,
            "test_domain": test_domain,
            "condition": condition.name,
            "metrics": str(output_root / "metrics.json"),
            "timestamp": time.time(),
        },
        output_root / "EVALUATION_COMPLETE.json",
    )
    return output_root

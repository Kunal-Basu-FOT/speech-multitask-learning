"""Metrics, prediction accumulation, and confusion matrices without sklearn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
import torch

from .conditions import Condition
from .labels import LabelSpace


def confusion_matrix(true: list[int], pred: list[int], classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(true, pred, strict=True):
        if not (0 <= t < classes and 0 <= p < classes):
            raise ValueError(f"Label outside [0,{classes}): true={t}, pred={p}")
        matrix[t, p] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    total = matrix.sum()
    accuracy = float(tp.sum() / total) if total else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()) if len(precision) else 0.0,
        "macro_recall": float(recall.mean()) if len(recall) else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
    }


@dataclass
class PredictionAccumulator:
    condition: Condition
    labels: LabelSpace
    lid_true: list[int] = field(default_factory=list)
    lid_pred: list[int] = field(default_factory=list)
    dialect_true: dict[str, list[int]] = field(default_factory=dict)
    dialect_pred: dict[str, list[int]] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        for lang in self.labels.languages:
            self.dialect_true[lang] = []
            self.dialect_pred[lang] = []

    def update(self, batch: dict, output: dict) -> None:
        batch_size = len(batch["utt_id"])
        lid_predictions: list[int | None] = [None] * batch_size
        dialect_predictions: list[int | None] = [None] * batch_size
        if "lid" in self.condition.active_tasks:
            preds = output["lid_logits"].detach().argmax(dim=-1).cpu().tolist()
            truth = batch["lid"].detach().cpu().tolist()
            self.lid_pred.extend(preds)
            self.lid_true.extend(truth)
            lid_predictions = preds
        if "dialect" in self.condition.active_tasks:
            for lang, (indices, logits) in output["dialect_logits"].items():
                idx = indices.detach().cpu().tolist()
                preds = logits.detach().argmax(dim=-1).cpu().tolist()
                truth = batch["dialect"][indices].detach().cpu().tolist()
                self.dialect_pred[lang].extend(preds)
                self.dialect_true[lang].extend(truth)
                for item_index, pred in zip(idx, preds, strict=True):
                    dialect_predictions[item_index] = pred
        for i in range(batch_size):
            self.rows.append(
                {
                    "utt_id": batch["utt_id"][i],
                    "filepath": batch["filepath"][i],
                    "oracle_language": batch["language"][i],
                    "true_language_index": int(batch["lid"][i].item()),
                    "predicted_language_index": lid_predictions[i],
                    "true_dialect_index": int(batch["dialect"][i].item()),
                    "predicted_dialect_index": dialect_predictions[i],
                }
            )

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metrics: dict[str, Any] = {}
        matrices: dict[str, np.ndarray] = {}
        if "lid" in self.condition.active_tasks:
            matrix = confusion_matrix(self.lid_true, self.lid_pred, self.labels.num_lid_classes)
            matrices["lid"] = matrix
            lid = classification_metrics(matrix)
            metrics.update({f"lid_{key}": value for key, value in lid.items()})
        if "dialect" in self.condition.active_tasks:
            per_language: dict[str, Any] = {}
            accuracies: list[float] = []
            macro_f1s: list[float] = []
            total_correct = 0
            total_examples = 0
            for lang in self.labels.languages:
                matrix = confusion_matrix(
                    self.dialect_true[lang],
                    self.dialect_pred[lang],
                    self.labels.num_dialect_classes(lang),
                )
                matrices[f"dialect_{lang}"] = matrix
                local = classification_metrics(matrix)
                per_language[lang] = local
                accuracies.append(local["accuracy"])
                macro_f1s.append(local["macro_f1"])
                total_correct += int(np.diag(matrix).sum())
                total_examples += int(matrix.sum())
            metrics["dialect_per_language"] = per_language
            metrics["dialect_sample_weighted_accuracy"] = (
                total_correct / total_examples if total_examples else 0.0
            )
            metrics["dialect_language_macro_accuracy"] = float(np.mean(accuracies))
            metrics["dialect_language_macro_f1"] = float(np.mean(macro_f1s))
        return metrics, matrices


def selection_score(condition: Condition, metrics: dict[str, Any]) -> float:
    values = [float(metrics[name]) for name in condition.selection_metrics]
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError(f"Invalid selection values for {condition.name}: {values}")
    return float(sum(values) / len(values))

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationResult:
    accuracy: float
    macro_f1: float
    examples: int
    classes: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "examples": self.examples,
            "classes": self.classes,
        }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray | None = None,
) -> ClassificationResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("classification arrays must be aligned one-dimensional vectors")
    if len(y_true) == 0:
        raise ValueError("classification arrays are empty")
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    accuracy = float(np.mean(y_true == y_pred))
    f1_values: list[float] = []
    for label in classes:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
    return ClassificationResult(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1_values)),
        examples=int(len(y_true)),
        classes=int(len(classes)),
    )


def macro_average(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")

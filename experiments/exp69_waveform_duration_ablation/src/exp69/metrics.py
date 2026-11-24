from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def classification_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    classes: int,
) -> dict[str, Any]:
    true = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if true.shape != predicted.shape:
        raise ValueError("true/predicted shapes differ")
    if true.ndim != 1:
        raise ValueError("classification arrays must be one-dimensional")
    if len(true) == 0:
        raise ValueError("classification arrays are empty")
    if np.any((true < 0) | (true >= classes)):
        raise ValueError("true label outside configured class range")
    if np.any((predicted < 0) | (predicted >= classes)):
        raise ValueError("predicted label outside configured class range")

    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (true, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    accuracy = float(tp.sum() / len(true))
    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "confusion_counts": confusion.astype(int).tolist(),
        "confusion_row_percent": np.divide(
            confusion.astype(np.float64) * 100.0,
            support[:, None],
            out=np.zeros_like(confusion, dtype=np.float64),
            where=support[:, None] != 0,
        ).tolist(),
    }


def multitask_metrics(
    *,
    true_lid: np.ndarray,
    pred_lid: np.ndarray,
    true_gender: np.ndarray,
    pred_gender: np.ndarray,
    true_dialect: np.ndarray,
    pred_dialect: np.ndarray,
    languages: np.ndarray,
    dialect_class_counts: dict[str, int],
) -> dict[str, Any]:
    language_result = classification_metrics(true_lid, pred_lid, 8)
    gender_result = classification_metrics(true_gender, pred_gender, 2)
    per_language: dict[str, Any] = {}
    dialect_accuracies: list[float] = []
    dialect_f1s: list[float] = []
    dialect_correct = 0
    dialect_total = 0
    languages = np.asarray(languages, dtype=str)

    for language, classes in dialect_class_counts.items():
        mask = languages == language
        if not np.any(mask):
            raise ValueError(f"No examples for language {language}")
        result = classification_metrics(
            np.asarray(true_dialect)[mask],
            np.asarray(pred_dialect)[mask],
            int(classes),
        )
        per_language[language] = result
        dialect_accuracies.append(float(result["accuracy"]))
        dialect_f1s.append(float(result["macro_f1"]))
        confusion = np.asarray(result["confusion_counts"], dtype=np.int64)
        dialect_correct += int(np.trace(confusion))
        dialect_total += int(confusion.sum())

    dialect_language_macro_accuracy = float(np.mean(dialect_accuracies))
    dialect_language_macro_f1 = float(np.mean(dialect_f1s))
    selection = float(
        np.mean(
            [
                language_result["macro_f1"],
                dialect_language_macro_f1,
                gender_result["macro_f1"],
            ]
        )
    )
    return {
        "examples": int(len(true_lid)),
        "language": language_result,
        "gender": gender_result,
        "dialect": {
            "sample_weighted_accuracy": float(dialect_correct / dialect_total),
            "language_macro_accuracy": dialect_language_macro_accuracy,
            "language_macro_f1": dialect_language_macro_f1,
            "per_language": per_language,
        },
        "three_task_mean_macro_f1": selection,
    }


@dataclass
class PredictionAccumulator:
    dialect_class_counts: dict[str, int]
    true_lid: list[int] = field(default_factory=list)
    pred_lid: list[int] = field(default_factory=list)
    true_gender: list[int] = field(default_factory=list)
    pred_gender: list[int] = field(default_factory=list)
    true_dialect: list[int] = field(default_factory=list)
    pred_dialect: list[int] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)

    def update(
        self,
        *,
        output,
        batch: dict[str, Any],
        dialect_predictions: torch.Tensor,
    ) -> None:
        self.true_lid.extend(batch["lid"].detach().cpu().tolist())
        self.pred_lid.extend(torch.argmax(output.language_logits, dim=1).detach().cpu().tolist())
        self.true_gender.extend(batch["gender"].detach().cpu().tolist())
        self.pred_gender.extend(torch.argmax(output.gender_logits, dim=1).detach().cpu().tolist())
        self.true_dialect.extend(batch["dialect"].detach().cpu().tolist())
        self.pred_dialect.extend(dialect_predictions.detach().cpu().tolist())
        self.languages.extend(str(value) for value in batch["language"])

    def finalize(self) -> dict[str, Any]:
        return multitask_metrics(
            true_lid=np.asarray(self.true_lid, dtype=np.int64),
            pred_lid=np.asarray(self.pred_lid, dtype=np.int64),
            true_gender=np.asarray(self.true_gender, dtype=np.int64),
            pred_gender=np.asarray(self.pred_gender, dtype=np.int64),
            true_dialect=np.asarray(self.true_dialect, dtype=np.int64),
            pred_dialect=np.asarray(self.pred_dialect, dtype=np.int64),
            languages=np.asarray(self.languages, dtype=str),
            dialect_class_counts=self.dialect_class_counts,
        )

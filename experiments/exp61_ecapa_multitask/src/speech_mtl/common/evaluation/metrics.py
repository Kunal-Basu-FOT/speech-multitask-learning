from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .confusion import confusion_matrix


@dataclass(frozen=True)
class ClassificationMetrics:
    examples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    support: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["per_class_precision"] = list(self.per_class_precision)
        value["per_class_recall"] = list(self.per_class_recall)
        value["per_class_f1"] = list(self.per_class_f1)
        value["support"] = list(self.support)
        return value


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def compute_classification_metrics(
    true_labels: np.ndarray | list[int],
    predicted_labels: np.ndarray | list[int],
    *,
    number_of_classes: int,
) -> ClassificationMetrics:
    true_array = np.asarray(true_labels, dtype=np.int64)
    predicted_array = np.asarray(predicted_labels, dtype=np.int64)

    matrix = confusion_matrix(
        true_array,
        predicted_array,
        number_of_classes=number_of_classes,
    )

    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)

    precision = _safe_divide(true_positive, predicted_positive)
    recall = _safe_divide(true_positive, actual_positive)
    f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    examples = int(matrix.sum())
    accuracy = (
        float(true_positive.sum() / examples)
        if examples
        else 0.0
    )

    return ClassificationMetrics(
        examples=examples,
        accuracy=accuracy,
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        per_class_precision=tuple(float(value) for value in precision),
        per_class_recall=tuple(float(value) for value in recall),
        per_class_f1=tuple(float(value) for value in f1),
        support=tuple(int(value) for value in actual_positive),
    )


def compute_multitask_summary(
    *,
    language_metrics: ClassificationMetrics,
    gender_metrics: ClassificationMetrics,
    dialect_metrics_by_language: dict[str, ClassificationMetrics],
) -> dict[str, Any]:
    if not dialect_metrics_by_language:
        raise ValueError("dialect_metrics_by_language cannot be empty")

    expected_languages = {"bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"}

    if set(dialect_metrics_by_language) != expected_languages:
        raise ValueError(
            "Dialect metric languages differ from the eight-language contract: "
            f"{sorted(dialect_metrics_by_language)}"
        )

    dialect_accuracies = np.asarray(
        [
            dialect_metrics_by_language[language].accuracy
            for language in sorted(dialect_metrics_by_language)
        ],
        dtype=np.float64,
    )

    dialect_macro_f1_values = np.asarray(
        [
            dialect_metrics_by_language[language].macro_f1
            for language in sorted(dialect_metrics_by_language)
        ],
        dtype=np.float64,
    )

    total_dialect_examples = sum(
        metric.examples
        for metric in dialect_metrics_by_language.values()
    )

    weighted_dialect_accuracy = (
        sum(
            metric.accuracy * metric.examples
            for metric in dialect_metrics_by_language.values()
        )
        / total_dialect_examples
        if total_dialect_examples
        else 0.0
    )

    dialect_language_macro_accuracy = float(
        dialect_accuracies.mean()
    )
    dialect_language_macro_f1 = float(
        dialect_macro_f1_values.mean()
    )

    selection_metric = float(
        np.mean(
            [
                language_metrics.macro_f1,
                dialect_language_macro_f1,
                gender_metrics.macro_f1,
            ]
        )
    )

    return {
        "language": language_metrics.to_dict(),
        "gender": gender_metrics.to_dict(),
        "dialect": {
            "examples": total_dialect_examples,
            "sample_weighted_accuracy": float(
                weighted_dialect_accuracy
            ),
            "language_macro_accuracy": (
                dialect_language_macro_accuracy
            ),
            "language_macro_f1": dialect_language_macro_f1,
            "per_language": {
                language: metric.to_dict()
                for language, metric
                in sorted(dialect_metrics_by_language.items())
            },
        },
        "selection_metric": {
            "name": "three_task_mean_macro_f1",
            "value": selection_metric,
            "components": {
                "language_macro_f1": language_metrics.macro_f1,
                "dialect_language_macro_f1": (
                    dialect_language_macro_f1
                ),
                "gender_macro_f1": gender_metrics.macro_f1,
            },
        },
    }

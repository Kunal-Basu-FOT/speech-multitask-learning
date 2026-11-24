from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def confusion_matrix(
    true_labels: np.ndarray | list[int],
    predicted_labels: np.ndarray | list[int],
    *,
    number_of_classes: int,
) -> np.ndarray:
    """Return an integer confusion matrix with rows=true, columns=predicted."""

    true_array = np.asarray(true_labels, dtype=np.int64)
    predicted_array = np.asarray(predicted_labels, dtype=np.int64)

    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("true_labels and predicted_labels must be one-dimensional")

    if true_array.shape != predicted_array.shape:
        raise ValueError(
            "true_labels and predicted_labels must have equal shape"
        )

    if number_of_classes <= 1:
        raise ValueError("number_of_classes must be at least two")

    if true_array.size == 0:
        return np.zeros(
            (number_of_classes, number_of_classes),
            dtype=np.int64,
        )

    if np.any(true_array < 0) or np.any(true_array >= number_of_classes):
        raise ValueError("A true label is outside the configured class range")

    if np.any(predicted_array < 0) or np.any(
        predicted_array >= number_of_classes
    ):
        raise ValueError("A predicted label is outside the configured class range")

    matrix = np.zeros(
        (number_of_classes, number_of_classes),
        dtype=np.int64,
    )

    np.add.at(matrix, (true_array, predicted_array), 1)
    return matrix


def row_normalized_percent(matrix: np.ndarray) -> np.ndarray:
    """Normalize each true-label row independently to percentages."""

    matrix = np.asarray(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(np.float64),
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )

    return normalized * 100.0


@dataclass(frozen=True)
class ConfusionMatrixBundle:
    counts: np.ndarray
    row_percent: np.ndarray
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "counts": self.counts.tolist(),
            "row_percent": self.row_percent.tolist(),
        }

    @classmethod
    def from_predictions(
        cls,
        true_labels: np.ndarray | list[int],
        predicted_labels: np.ndarray | list[int],
        labels: list[str] | tuple[str, ...],
    ) -> "ConfusionMatrixBundle":
        label_tuple = tuple(str(label) for label in labels)

        counts = confusion_matrix(
            true_labels,
            predicted_labels,
            number_of_classes=len(label_tuple),
        )

        return cls(
            counts=counts,
            row_percent=row_normalized_percent(counts),
            labels=label_tuple,
        )

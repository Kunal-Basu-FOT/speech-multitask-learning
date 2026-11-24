from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .io import ensure_parent


_DIALECT_ORDER = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")


def _matrix_from_item(item: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    matrix = np.asarray(item["row_percent"], dtype=np.float64)
    labels = [str(label) for label in item["labels"]]

    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            "Confusion-matrix dimensions do not match labels"
        )

    return matrix, labels


def _draw_matrix(axis, item: dict[str, Any], title: str) -> None:
    matrix, labels = _matrix_from_item(item)
    image = axis.imshow(
        matrix,
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )

    axis.set_title(title, fontsize=9)
    axis.set_xlabel("Predicted", fontsize=8)
    axis.set_ylabel("True", fontsize=8)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)

    threshold = 50.0

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value >= threshold else "black",
            )

    return image


def save_composite_confusion_figure(
    confusion_matrices: dict[str, Any],
    *,
    output_path: Path | str,
    title: str,
    dpi: int = 300,
) -> str:
    """Save a 2x5 composite: LID, gender, and eight dialect heads."""

    if set(confusion_matrices) != {
        "language",
        "gender",
        "dialect",
    }:
        raise ValueError(
            "Expected language, gender, and dialect confusion matrices"
        )

    dialect = confusion_matrices["dialect"]

    if set(dialect) != set(_DIALECT_ORDER):
        raise ValueError(
            "Dialect confusion matrices do not cover all eight languages"
        )

    destination = ensure_parent(output_path)
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8),
        constrained_layout=True,
    )
    flattened = list(axes.ravel())

    image = _draw_matrix(
        flattened[0],
        confusion_matrices["language"],
        "Language",
    )
    _draw_matrix(
        flattened[1],
        confusion_matrices["gender"],
        "Gender",
    )

    for offset, language in enumerate(_DIALECT_ORDER, start=2):
        _draw_matrix(
            flattened[offset],
            dialect[language],
            f"Dialect: {language}",
        )

    figure.suptitle(title, fontsize=14)

    colorbar = figure.colorbar(
        image,
        ax=flattened,
        shrink=0.75,
        pad=0.02,
    )
    colorbar.set_label("Row-normalized percentage")

    figure.savefig(
        destination,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return str(destination)

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import ensure_parent


_CONDITIONS = (
    "1-bit → 1-bit",
    "1-bit → 16-bit",
    "16-bit → 16-bit",
    "16-bit → 1-bit",
)


def _extract_row(
    label: str,
    evaluation: dict[str, Any],
) -> dict[str, float | str]:
    summary = evaluation["summary"]

    return {
        "condition": label,
        "language_accuracy": float(
            summary["language"]["accuracy"]
        ),
        "dialect_language_macro_accuracy": float(
            summary["dialect"]["language_macro_accuracy"]
        ),
        "gender_accuracy": float(
            summary["gender"]["accuracy"]
        ),
        "selection_metric": float(
            summary["selection_metric"]["value"]
        ),
    }


def cross_precision_frame(
    evaluations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if set(evaluations) != set(_CONDITIONS):
        raise ValueError(
            "Cross-precision evaluations must contain exactly: "
            f"{list(_CONDITIONS)}"
        )

    return pd.DataFrame(
        [
            _extract_row(condition, evaluations[condition])
            for condition in _CONDITIONS
        ]
    )


def save_cross_precision_summary(
    evaluations: dict[str, dict[str, Any]],
    *,
    output_path: Path | str,
    dpi: int = 300,
) -> str:
    frame = cross_precision_frame(evaluations)
    destination = ensure_parent(output_path)

    x = np.arange(len(frame))
    width = 0.2

    figure, axis = plt.subplots(figsize=(12, 6))

    series = (
        ("language_accuracy", "Language accuracy"),
        (
            "dialect_language_macro_accuracy",
            "Dialect language-macro accuracy",
        ),
        ("gender_accuracy", "Gender accuracy"),
        ("selection_metric", "Three-task mean macro-F1"),
    )

    for index, (column, label) in enumerate(series):
        axis.bar(
            x + (index - 1.5) * width,
            frame[column],
            width=width,
            label=label,
        )

    axis.set_title("Matched and Cross-Precision Performance")
    axis.set_xlabel("Training precision → evaluation precision")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(frame["condition"])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return str(destination)

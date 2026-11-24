from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .io import ensure_parent


_REQUIRED_COLUMNS = {"epoch", "train_loss", "dev_loss"}


def _validate_history(history: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(history.columns))

    if missing:
        raise ValueError(
            f"History is missing required columns: {missing}"
        )

    if history["epoch"].duplicated().any():
        raise ValueError("History contains duplicate epoch values")

    if history["epoch"].isna().any():
        raise ValueError("History contains missing epoch values")

    if not history["epoch"].is_monotonic_increasing:
        raise ValueError("History epochs must be increasing")


def _phase_lines(axis) -> None:
    # Boundaries after epoch 5 and epoch 35.
    axis.axvline(5.5, linestyle="--", linewidth=1.0)
    axis.axvline(35.5, linestyle="--", linewidth=1.0)


def save_training_curves(
    history: pd.DataFrame,
    *,
    loss_output: Path | str,
    accuracy_output: Path | str,
    title_prefix: str,
    dpi: int = 300,
) -> dict[str, str]:
    """Save one loss figure and one accuracy figure.

    Accuracy columns are expected when present:
      train_language_accuracy, dev_language_accuracy
      train_gender_accuracy, dev_gender_accuracy
      train_dialect_language_macro_accuracy,
      dev_dialect_language_macro_accuracy
      train_three_task_mean_accuracy,
      dev_three_task_mean_accuracy

    The final mean-accuracy pair is required for the accuracy figure.
    """

    _validate_history(history)

    required_accuracy = {
        "train_three_task_mean_accuracy",
        "dev_three_task_mean_accuracy",
    }

    missing_accuracy = sorted(
        required_accuracy - set(history.columns)
    )

    if missing_accuracy:
        raise ValueError(
            "History is missing required accuracy columns: "
            f"{missing_accuracy}"
        )

    loss_destination = ensure_parent(loss_output)
    accuracy_destination = ensure_parent(accuracy_output)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
        linewidth=2,
    )
    axis.plot(
        history["epoch"],
        history["dev_loss"],
        label="Development",
        linewidth=2,
    )
    _phase_lines(axis)
    axis.set_title(f"{title_prefix}: Train and Development Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        history["epoch"],
        history["train_three_task_mean_accuracy"],
        label="Train three-task mean",
        linewidth=2,
    )
    axis.plot(
        history["epoch"],
        history["dev_three_task_mean_accuracy"],
        label="Development three-task mean",
        linewidth=2,
    )

    optional_series: Iterable[tuple[str, str]] = (
        (
            "dev_language_accuracy",
            "Development language",
        ),
        (
            "dev_dialect_language_macro_accuracy",
            "Development dialect language-macro",
        ),
        (
            "dev_gender_accuracy",
            "Development gender",
        ),
    )

    for column, label in optional_series:
        if column in history.columns:
            axis.plot(
                history["epoch"],
                history[column],
                label=label,
                linewidth=1.3,
                alpha=0.8,
            )

    _phase_lines(axis)
    axis.set_title(f"{title_prefix}: Train and Development Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        accuracy_destination,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return {
        "loss_figure": str(loss_destination),
        "accuracy_figure": str(accuracy_destination),
    }

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


class EpochHistoryWriter:
    FIELDNAMES = (
        "epoch",
        "phase",
        "train_loss",
        "dev_loss",
        "train_language_accuracy",
        "dev_language_accuracy",
        "train_language_macro_f1",
        "dev_language_macro_f1",
        "train_dialect_language_macro_accuracy",
        "dev_dialect_language_macro_accuracy",
        "train_dialect_language_macro_f1",
        "dev_dialect_language_macro_f1",
        "train_gender_accuracy",
        "dev_gender_accuracy",
        "train_gender_macro_f1",
        "dev_gender_macro_f1",
        "train_three_task_mean_accuracy",
        "dev_three_task_mean_accuracy",
        "train_three_task_mean_macro_f1",
        "dev_three_task_mean_macro_f1",
        "optimizer_steps",
        "encoder_learning_rate",
        "task_learning_rate",
        "mean_gradient_norm",
        "padding_ratio",
        "elapsed_seconds",
        "examples_per_second",
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def completed_epochs(self) -> set[int]:
        if not self.path.is_file():
            return set()

        with self.path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            return {
                int(row["epoch"])
                for row in csv.DictReader(handle)
            }

    def append(self, row: dict[str, Any]) -> None:
        unknown = sorted(set(row) - set(self.FIELDNAMES))

        if unknown:
            raise ValueError(
                f"Unknown history fields: {unknown}"
            )

        if int(row["epoch"]) in self.completed_epochs():
            raise ValueError(
                f"History already contains epoch {row['epoch']}"
            )

        exists = self.path.is_file()

        with self.path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.FIELDNAMES,
            )

            if not exists:
                writer.writeheader()

            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in self.FIELDNAMES
                }
            )
            handle.flush()
            os.fsync(handle.fileno())


def history_row(
    train_report: dict[str, Any],
    dev_report: dict[str, Any],
) -> dict[str, Any]:
    train_summary = train_report["summary"]
    dev_summary = dev_report["summary"]

    train_accuracy_mean = (
        train_summary["language"]["accuracy"]
        + train_summary["dialect"][
            "language_macro_accuracy"
        ]
        + train_summary["gender"]["accuracy"]
    ) / 3.0

    dev_accuracy_mean = (
        dev_summary["language"]["accuracy"]
        + dev_summary["dialect"][
            "language_macro_accuracy"
        ]
        + dev_summary["gender"]["accuracy"]
    ) / 3.0

    return {
        "epoch": train_report["epoch"],
        "phase": train_report["phase"],
        "train_loss": train_report["mean_total_loss"],
        "dev_loss": dev_report["mean_total_loss"],
        "train_language_accuracy": (
            train_summary["language"]["accuracy"]
        ),
        "dev_language_accuracy": (
            dev_summary["language"]["accuracy"]
        ),
        "train_language_macro_f1": (
            train_summary["language"]["macro_f1"]
        ),
        "dev_language_macro_f1": (
            dev_summary["language"]["macro_f1"]
        ),
        "train_dialect_language_macro_accuracy": (
            train_summary["dialect"][
                "language_macro_accuracy"
            ]
        ),
        "dev_dialect_language_macro_accuracy": (
            dev_summary["dialect"][
                "language_macro_accuracy"
            ]
        ),
        "train_dialect_language_macro_f1": (
            train_summary["dialect"][
                "language_macro_f1"
            ]
        ),
        "dev_dialect_language_macro_f1": (
            dev_summary["dialect"][
                "language_macro_f1"
            ]
        ),
        "train_gender_accuracy": (
            train_summary["gender"]["accuracy"]
        ),
        "dev_gender_accuracy": (
            dev_summary["gender"]["accuracy"]
        ),
        "train_gender_macro_f1": (
            train_summary["gender"]["macro_f1"]
        ),
        "dev_gender_macro_f1": (
            dev_summary["gender"]["macro_f1"]
        ),
        "train_three_task_mean_accuracy": (
            train_accuracy_mean
        ),
        "dev_three_task_mean_accuracy": dev_accuracy_mean,
        "train_three_task_mean_macro_f1": (
            train_summary["selection_metric"]["value"]
        ),
        "dev_three_task_mean_macro_f1": (
            dev_summary["selection_metric"]["value"]
        ),
        "optimizer_steps": train_report["optimizer_steps"],
        "encoder_learning_rate": (
            train_report["encoder_learning_rate"]
        ),
        "task_learning_rate": (
            train_report["task_learning_rate"]
        ),
        "mean_gradient_norm": (
            train_report["mean_gradient_norm"]
        ),
        "padding_ratio": train_report["padding_ratio"],
        "elapsed_seconds": train_report["elapsed_seconds"],
        "examples_per_second": (
            train_report["examples_per_second"]
        ),
    }

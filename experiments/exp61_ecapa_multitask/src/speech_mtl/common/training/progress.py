from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ProgressWriter:
    """Atomic machine-readable progress JSON writer."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=self.path.parent,
        )

        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary_name, self.path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)


class HistoryWriter:
    """Append-only epoch history CSV with a stable schema."""

    FIELDNAMES = (
        "epoch",
        "phase",
        "optimizer_steps",
        "examples",
        "mean_total_loss",
        "mean_language_loss",
        "mean_dialect_loss",
        "mean_gender_loss",
        "encoder_learning_rate",
        "task_learning_rate",
        "mean_gradient_norm",
        "elapsed_seconds",
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        unknown = sorted(set(row) - set(self.FIELDNAMES))

        if unknown:
            raise ValueError(
                f"Unknown history fields: {unknown}"
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

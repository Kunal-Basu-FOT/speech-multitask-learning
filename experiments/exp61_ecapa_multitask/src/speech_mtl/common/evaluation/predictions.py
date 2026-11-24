from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


class PredictionWriter:
    """Streaming prediction CSV writer with fsync on close."""

    FIELDNAMES = (
        "utt_id",
        "filepath",
        "oracle_language",
        "true_language_index",
        "predicted_language_index",
        "true_dialect_index",
        "predicted_dialect_index",
        "true_gender_index",
        "predicted_gender_index",
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._writer = None

    def __enter__(self) -> "PredictionWriter":
        self._handle = self.path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=self.FIELDNAMES,
        )
        self._writer.writeheader()
        return self

    def write_batch(self, rows: list[dict[str, Any]]) -> None:
        if self._writer is None:
            raise RuntimeError(
                "PredictionWriter must be used as a context manager"
            )

        for row in rows:
            unknown = sorted(set(row) - set(self.FIELDNAMES))
            missing = sorted(set(self.FIELDNAMES) - set(row))

            if unknown or missing:
                raise ValueError(
                    f"Prediction row schema mismatch; "
                    f"missing={missing}, unknown={unknown}"
                )

            self._writer.writerow(row)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

        self._writer = None
        self._handle = None

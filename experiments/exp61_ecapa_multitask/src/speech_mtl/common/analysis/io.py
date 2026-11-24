from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_json(path: Path | str) -> dict[str, Any]:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {source}")

    value = json.loads(source.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")

    return value


def load_history(path: Path | str) -> pd.DataFrame:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"History CSV does not exist: {source}")

    frame = pd.read_csv(source)

    if frame.empty:
        raise ValueError(f"History CSV is empty: {source}")

    return frame


def ensure_parent(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.raw[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def path(self, key: str) -> Path:
        return Path(self.raw[key])


def load_config(path: str | Path) -> Config:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "project_root", "exp61_root", "source_cache_root",
        "logmel_cache_root", "run_root", "results_root",
        "report_root", "expected_rows", "domains", "durations",
        "duration_seconds", "sample_rate", "win_length",
        "hop_length", "n_fft", "n_mels", "seed", "epochs",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"Config missing keys: {missing}")
    return Config(value)

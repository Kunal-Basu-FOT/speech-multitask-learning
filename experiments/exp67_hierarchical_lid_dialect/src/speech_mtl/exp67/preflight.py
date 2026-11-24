"""Read-only scientific and binary cache validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import hashlib

DOMAINS = ("one_bit", "sixteen_bit")
from .config import ExperimentConfig
from .data import CachedFeatureDataset
from .labels import LabelSpace
from .utils import atomic_json_dump


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_preflight(cfg: ExperimentConfig, labels: LabelSpace, *, full_hash: bool = False) -> dict[str, Any]:
    expected_rows = {"train": 146359, "dev": 11507, "test": 17550}
    result: dict[str, Any] = {"domains": {}, "cross_domain_index_identity": {}}
    for domain in DOMAINS:
        result["domains"][domain] = {}
        for split in ("train", "dev", "test"):
            dataset = CachedFeatureDataset(
                cfg.paths.cache_root / domain / split,
                label_space=labels,
                feature_dim=cfg.model.input_dim,
            )
            observed = {
                "rows": len(dataset),
                "total_frames": dataset.total_frames,
                "min_frames": min(dataset.lengths),
                "max_frames": max(dataset.lengths),
                "features_bytes": dataset.features_path.stat().st_size,
                "index_sha256": _sha256(dataset.index_path) if full_hash else None,
            }
            if observed["rows"] != expected_rows[split]:
                raise ValueError(
                    f"{domain}/{split}: expected {expected_rows[split]} rows, got {observed['rows']}"
                )
            result["domains"][domain][split] = observed
    for split in ("train", "dev", "test"):
        one = cfg.paths.cache_root / "one_bit" / split / "index.csv"
        sixteen = cfg.paths.cache_root / "sixteen_bit" / split / "index.csv"
        identical = one.read_bytes() == sixteen.read_bytes()
        if not identical:
            raise ValueError(f"Cross-domain indexes differ for split {split}")
        result["cross_domain_index_identity"][split] = True
    output = cfg.paths.project_root / "validation" / "PREFLIGHT.json"
    atomic_json_dump(result | {"status": "pass"}, output)
    return result

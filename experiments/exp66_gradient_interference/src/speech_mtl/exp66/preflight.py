"""Read-only cache validation for EXP 6.6."""
from __future__ import annotations

from typing import Any
import hashlib

from .config import ExperimentConfig
from .data import CachedFeatureDataset
from .labels import LabelSpace
from .utils import atomic_json_dump

DOMAINS = ("one_bit", "sixteen_bit")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(
    cfg: ExperimentConfig,
    labels: LabelSpace,
    *,
    full_hash: bool = False,
) -> dict[str, Any]:
    expected_rows = {"train": 146359, "dev": 11507, "test": 17550}
    expected_bytes = {
        "train": 66300699648,
        "dev": 4858670592,
        "test": 7373816832,
    }
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
                "complete_marker": (dataset.split_root / "COMPLETE.json").is_file(),
                "index_sha256": _sha256(dataset.index_path) if full_hash else None,
            }
            if observed["rows"] != expected_rows[split]:
                raise ValueError(
                    f"{domain}/{split}: expected {expected_rows[split]} rows, "
                    f"got {observed['rows']}"
                )
            if observed["features_bytes"] != expected_bytes[split]:
                raise ValueError(
                    f"{domain}/{split}: expected {expected_bytes[split]} bytes, "
                    f"got {observed['features_bytes']}"
                )
            if not observed["complete_marker"]:
                raise ValueError(f"Missing COMPLETE.json for {domain}/{split}")
            result["domains"][domain][split] = observed
    for split in ("train", "dev", "test"):
        one = cfg.paths.cache_root / "one_bit" / split / "index.csv"
        sixteen = cfg.paths.cache_root / "sixteen_bit" / split / "index.csv"
        identical = one.read_bytes() == sixteen.read_bytes()
        if not identical:
            raise ValueError(f"Cross-domain indexes differ for {split}")
        result["cross_domain_index_identity"][split] = True
    atomic_json_dump(
        result | {"status": "pass"},
        cfg.paths.project_root / "validation" / "PREFLIGHT.json",
    )
    return result

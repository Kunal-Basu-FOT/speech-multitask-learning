from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import cell_path, source_specs
from .config import AnalysisConfig, COMBINED_ROOT
from .utils import atomic_json, sha256_file, utc_now


EXPECTED_EXP65_TABLES = (
    "table_01_all_probe_results.csv",
    "table_02_matched_probe_macro_f1.csv",
    "table_03_joint_vs_single_task_deltas.csv",
    "table_04_precision_deltas.csv",
    "table_05_geometry_metrics.csv",
    "table_06_native_head_metrics.csv",
    "table_07_native_head_vs_probe.csv",
)
EXPECTED_EXP68_TABLES = (
    "table_01_fewshot_gender_all_results.csv",
    "table_02_transfer_summary_100_percent.csv",
    "table_03_transfer_gain_and_recovery.csv",
    "table_04_matched_vs_cross_precision.csv",
)
EXPECTED_EXP65_FIGURES = (
    "figure_01_probe_macro_f1_heatmap.png",
    "figure_02_joint_vs_single_task_deltas.png",
    "figure_03_umap_one_bit_task_groupings.png",
    "figure_03_umap_sixteen_bit_task_groupings.png",
    "figure_04_umap_lid_dialect_precision.png",
    "figure_05_geometry_separation.png",
    "appendix_umap_dialect_one_bit.png",
    "appendix_umap_dialect_sixteen_bit.png",
    "appendix_tsne_lid_dialect_precision.png",
)
EXPECTED_EXP68_FIGURES = (
    "figure_01_gender_accuracy_learning_curves.png",
    "figure_02_gender_macro_f1_learning_curves.png",
    "figure_03_transfer_gain_over_random.png",
    "figure_04_matched_vs_cross_precision.png",
)


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"required output missing or empty: {path}")


def _validate_csv(path: Path) -> dict[str, Any]:
    _require_nonempty(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"CSV has no data rows: {path}")
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))[~np.isnan(numeric.to_numpy(dtype=float, na_value=np.nan))].all():
        raise RuntimeError(f"CSV contains non-finite numeric values: {path}")
    return {"path": str(path), "rows": len(frame), "columns": len(frame.columns), "sha256": sha256_file(path)}


def _embedding_inventory(config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for spec in source_specs(config):
        for input_domain in config.domains:
            for split in config.splits:
                path = cell_path(spec, input_domain, split)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                embeddings_path = path / "embeddings.npy"
                metadata_path = path / "metadata.csv"
                marker_path = path / "COMPLETE.json"
                _require_nonempty(embeddings_path)
                _require_nonempty(metadata_path)
                _require_nonempty(marker_path)
                embeddings = np.load(embeddings_path, mmap_mode="r")
                metadata = pd.read_csv(metadata_path, usecols=["utt_id"], dtype={"utt_id": str})
                expected = config.expected_rows[split]
                if embeddings.shape != (expected, config.embedding_dimension):
                    raise RuntimeError(f"wrong embedding shape: {embeddings_path} {embeddings.shape}")
                if embeddings.dtype != np.float32:
                    raise RuntimeError(f"wrong embedding dtype: {embeddings_path} {embeddings.dtype}")
                if len(metadata) != expected or metadata.utt_id.duplicated().any():
                    raise RuntimeError(f"metadata validation failed: {metadata_path}")
                rows.append(
                    {
                        "family": spec.family,
                        "source": spec.display_name,
                        "model_domain": spec.model_domain,
                        "condition": spec.condition,
                        "input_domain": input_domain,
                        "split": split,
                        "rows": expected,
                        "dimension": config.embedding_dimension,
                        "embeddings_bytes": embeddings_path.stat().st_size,
                        "metadata_bytes": metadata_path.stat().st_size,
                        "cell": str(path),
                    }
                )
    return pd.DataFrame(rows)


def run(config: AnalysisConfig) -> dict[str, Any]:
    report_marker = COMBINED_ROOT / "reports/combined/REPORT_COMPLETE.json"
    _require_nonempty(report_marker)
    _require_nonempty(COMBINED_ROOT / "reports/combined/README_RESULTS.md")
    csv_records: list[dict[str, Any]] = []
    for name in EXPECTED_EXP65_TABLES:
        csv_records.append(_validate_csv(COMBINED_ROOT / "reports/exp65/tables" / name))
    for name in EXPECTED_EXP68_TABLES:
        csv_records.append(_validate_csv(COMBINED_ROOT / "reports/exp68/tables" / name))
    figure_records: list[dict[str, Any]] = []
    for name in EXPECTED_EXP65_FIGURES:
        path = COMBINED_ROOT / "reports/exp65/figures" / name
        _require_nonempty(path)
        figure_records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    for name in EXPECTED_EXP68_FIGURES:
        path = COMBINED_ROOT / "reports/exp68/figures" / name
        _require_nonempty(path)
        figure_records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    inventory = _embedding_inventory(config)
    inventory_path = COMBINED_ROOT / "reports/combined/table_embedding_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    result = {
        "status": "EXPERIMENT_COMPLETE",
        "created_utc": utc_now(),
        "experiments": ["EXP6.5", "EXP6.8"],
        "encoder_training_performed": False,
        "embedding_cells": len(inventory),
        "tables": csv_records,
        "figures": figure_records,
        "summary": str(COMBINED_ROOT / "reports/combined/README_RESULTS.md"),
        "report_marker": json.loads(report_marker.read_text(encoding="utf-8")),
    }
    atomic_json(COMBINED_ROOT / "runtime/EXPERIMENT_COMPLETE.json", result)
    print("EXP65_EXP68_COMPLETE")
    return result

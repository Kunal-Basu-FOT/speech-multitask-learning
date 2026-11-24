from __future__ import annotations

import importlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import AnalysisConfig, CACHE_ROOT, COMBINED_ROOT, EXP61_ROOT, EXP62_ROOT
from .encoder import load_frozen_encoder
from .extract import exp62_checkpoint
from .utils import atomic_json, sha256_file, utc_now


def _history_rows(path: Path) -> int:
    frame = pd.read_csv(path)
    return len(frame)


def _check_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "torch", "matplotlib", "sklearn", "umap"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _exp61_reference_gate(config: AnalysisConfig) -> list[dict[str, Any]]:
    root = COMBINED_ROOT / "runtime/reference_embeddings/exp61"
    overall = root / "REFERENCE_EXTRACTION_COMPLETE.json"
    if not overall.is_file():
        raise RuntimeError(f"EXP 6.1 reference marker missing: {overall}")
    outputs: list[dict[str, Any]] = []
    for model_domain in config.domains:
        for input_domain in config.domains:
            for split in config.splits:
                cell = root / f"model_{model_domain}" / f"input_{input_domain}" / split
                marker = cell / "COMPLETE.json"
                embeddings = cell / "embeddings.npy"
                metadata = cell / "metadata.csv"
                for path in (marker, embeddings, metadata):
                    if not path.is_file() or path.stat().st_size == 0:
                        raise RuntimeError(f"incomplete EXP 6.1 reference cell: {path}")
                record = json.loads(marker.read_text(encoding="utf-8"))
                if int(record.get("rows", -1)) != config.expected_rows[split]:
                    raise RuntimeError(f"wrong EXP 6.1 reference row count: {marker}")
                outputs.append(record)
    if len(outputs) != 12:
        raise RuntimeError(f"EXP 6.1 reference cells={len(outputs)} expected=12")
    return outputs


def _exp62_gate(config: AnalysisConfig) -> list[dict[str, Any]]:
    run = EXP62_ROOT / "runtime/runs/exp62_reduced"
    records: list[dict[str, Any]] = []
    for domain in config.domains:
        for condition in config.conditions:
            root = run / domain / condition
            paths = {
                "training_complete": root / "TRAINING_COMPLETE.json",
                "history": root / "history.csv",
                "best": root / "checkpoints/best.pt",
                "last": root / "checkpoints/last.pt",
                "best_dev": root / "best_dev_metrics.json",
            }
            for name, path in paths.items():
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"EXP 6.2 {domain}/{condition} missing {name}: {path}")
            if _history_rows(paths["history"]) != 40:
                raise RuntimeError(f"EXP 6.2 {domain}/{condition} does not have 40 history rows")
            marker = json.loads(paths["training_complete"].read_text(encoding="utf-8"))
            marker_status = str(marker.get("status", "")).strip().lower()
            if marker_status not in {"complete", "completed", "training_complete"}:
                raise RuntimeError(
                    f"invalid training marker status={marker.get('status')!r}: "
                    f"{paths['training_complete']}"
                )
            if str(marker.get("domain", domain)) != domain:
                raise RuntimeError(
                    f"training marker domain mismatch for {domain}/{condition}: "
                    f"{paths['training_complete']}"
                )
            if str(marker.get("condition", condition)) != condition:
                raise RuntimeError(
                    f"training marker condition mismatch for {domain}/{condition}: "
                    f"{paths['training_complete']}"
                )
            records.append(
                {
                    "domain": domain,
                    "condition": condition,
                    "checkpoint": str(paths["best"]),
                    "checkpoint_sha256": sha256_file(paths["best"]),
                    "history_rows": 40,
                }
            )
    complete_marker = run / "EXPERIMENT_COMPLETE.json"
    queue_marker = run / "QUEUE_PROGRESS.json"
    if not complete_marker.is_file():
        if queue_marker.is_file():
            queue = json.loads(queue_marker.read_text(encoding="utf-8"))
            if str(queue.get("status", "")).lower() not in {"complete", "completed"}:
                raise RuntimeError(
                    "all condition artifacts exist but EXP 6.2 global completion marker is missing and queue is not complete"
                )
        else:
            raise RuntimeError("EXP 6.2 global completion evidence is missing")
    return records


def _cache_gate(config: AnalysisConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in config.splits:
        paths = [CACHE_ROOT / domain / split / "index.csv" for domain in config.domains]
        hashes = [sha256_file(path) for path in paths]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"one-bit and sixteen-bit indexes differ for split={split}")
        frame = pd.read_csv(paths[0], usecols=["utt_id"])
        if len(frame) != config.expected_rows[split]:
            raise RuntimeError(f"cache split={split} rows={len(frame)} expected={config.expected_rows[split]}")
        if frame["utt_id"].astype(str).duplicated().any():
            raise RuntimeError(f"duplicate utt_id in cache split={split}")
        records.append({"split": split, "rows": len(frame), "index_sha256": hashes[0]})
    return records


def _gpu_gate() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"launcher must expose exactly one GPU; visible GPUs={torch.cuda.device_count()}"
        )
    return {
        "visible_device_count": torch.cuda.device_count(),
        "visible_device_name": torch.cuda.get_device_name(0),
    }


def run(config: AnalysisConfig, validate_checkpoints: bool = True) -> dict[str, Any]:
    host = socket.gethostname()
    user = os.environ.get("USER", "")
    if host != "wtc9-server" or user != "wtc9":
        raise RuntimeError(f"expected wtc9@wtc9-server, observed {user}@{host}")
    for root in (COMBINED_ROOT, EXP61_ROOT, EXP62_ROOT, CACHE_ROOT):
        if not root.is_dir():
            raise RuntimeError(f"required root missing: {root}")
    dependencies = _check_dependencies()
    cache = _cache_gate(config)
    exp62 = _exp62_gate(config)
    exp61_references = _exp61_reference_gate(config)
    gpu = _gpu_gate()
    if validate_checkpoints:
        device = torch.device("cuda:0")
        for domain in config.domains:
            for condition in config.conditions:
                encoder = load_frozen_encoder(exp62_checkpoint(domain, condition), device)
                parameters = sum(parameter.numel() for parameter in encoder.parameters())
                if parameters <= 0:
                    raise RuntimeError(f"empty encoder for {domain}/{condition}")
                del encoder
                torch.cuda.empty_cache()
    result = {
        "status": "EXP65_68_PREFLIGHT_PASS",
        "created_utc": utc_now(),
        "host": host,
        "user": user,
        "dependencies": dependencies,
        "gpu": gpu,
        "cache": cache,
        "exp62": exp62,
        "exp61_reference_cells": len(exp61_references),
    }
    atomic_json(COMBINED_ROOT / "runtime/PREFLIGHT.json", result)
    print("EXP65_68_PREFLIGHT_PASS")
    return result

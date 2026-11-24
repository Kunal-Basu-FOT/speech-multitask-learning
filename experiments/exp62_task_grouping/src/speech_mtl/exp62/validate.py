"""Synthetic integration validation for all three reduced conditions."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch

from .conditions import CONDITION_ORDER, get_condition
from .config import ExperimentConfig
from .labels import LabelSpace
from .losses import compute_losses
from .metrics import PredictionAccumulator, selection_score
from .model import ReducedMTLModel
from .utils import atomic_json_dump


def synthetic_label_space() -> LabelSpace:
    payload = {
        "languages": ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"],
        "gender_values": ["0", "1"],
        "dialect_values_by_language": {
            "bh": ["0", "1", "2"],
            "bn": ["0", "1", "2", "3", "4"],
            "ch": ["0", "1", "2", "3"],
            "kn": ["0", "1", "2", "3", "4"],
            "mg": ["0", "1", "2", "3"],
            "mr": ["0", "1", "2", "3"],
            "mt": ["0", "1", "2", "3"],
            "te": ["0", "1", "2", "3"],
        },
    }
    return LabelSpace.from_dict(payload)


def synthetic_batch(labels: LabelSpace, device: torch.device) -> dict:
    languages = list(labels.languages)
    lengths = torch.tensor([47 + i for i in range(len(languages))], dtype=torch.long)
    max_len = int(lengths.max())
    features = torch.randn(len(languages), max_len, 768)
    valid = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    dialect = torch.tensor([0] * len(languages), dtype=torch.long)
    return {
        "features": features.to(device),
        "frame_lengths": lengths.to(device),
        "relative_lengths": (lengths.float() / max_len).to(device),
        "valid_mask": valid.to(device),
        "lid": torch.arange(len(languages), dtype=torch.long).to(device),
        "dialect": dialect.to(device),
        "gender": torch.tensor([i % 2 for i in range(len(languages))], dtype=torch.long).to(device),
        "language": languages,
        "utt_id": [f"utt_{i}" for i in range(len(languages))],
        "filepath": [f"/tmp/{i}.wav" for i in range(len(languages))],
        "dataset_index": torch.arange(len(languages), dtype=torch.long).to(device),
    }


def run_synthetic_validation(cfg: ExperimentConfig, device: torch.device | None = None) -> dict:
    device = device or torch.device("cpu")
    labels = synthetic_label_space()
    results = {}
    for name in CONDITION_ORDER:
        condition = get_condition(name)
        model = ReducedMTLModel(cfg.model, labels, condition).to(device)
        if sum(p.numel() for p in model.ecapa.parameters()) != 7_212_660:
            raise AssertionError("ECAPA parameter contract mismatch")
        batch = synthetic_batch(labels, device)
        output = model(batch)
        losses = compute_losses(model, batch, output)
        if not torch.isfinite(losses["total"]):
            raise AssertionError(f"Non-finite synthetic loss for {name}")
        losses["total"].backward()
        accumulator = PredictionAccumulator(condition, labels)
        accumulator.update(batch, output)
        metrics, matrices = accumulator.finalize()
        score = selection_score(condition, metrics)
        if not np.isfinite(score):
            raise AssertionError(f"Invalid selection score for {name}")
        results[name] = {
            "loss": float(losses["total"].item()),
            "selection_score": score,
            "parameter_counts": model.parameter_counts,
            "matrix_keys": sorted(matrices),
        }
    output_path = cfg.paths.project_root / "validation" / "SYNTHETIC_VALIDATION.json"
    atomic_json_dump(results | {"status": "pass"}, output_path)
    return results

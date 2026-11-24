"""Epoch-boundary checkpoints with RNG state and run-contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch

from .utils import atomic_torch_save


def capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(
        payload["torch_cpu"]
        .detach()
        .to(device="cpu", dtype=torch.uint8)
    )
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all([
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["torch_cuda"]
        ])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["rng_state"] = capture_rng_state()
    atomic_torch_save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    return payload

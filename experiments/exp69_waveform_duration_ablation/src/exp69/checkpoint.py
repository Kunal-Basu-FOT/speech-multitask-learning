from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import atomic_torch_save


def _cpu_byte_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return an RNG state in the exact format required by torch.set_rng_state.

    This conversion is intentionally explicit. It prevents the
    `RNG state must be a torch.ByteTensor` failure that occurs when a
    checkpoint is loaded with map_location set to a CUDA device.
    """
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": _cpu_byte_tensor(torch.get_rng_state()),
        "torch_cuda": [
            _cpu_byte_tensor(state)
            for state in (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        ],
    }


def restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(payload["torch_cpu"]))

    cuda_states = payload.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        visible = torch.cuda.device_count()
        if len(cuda_states) != visible:
            raise RuntimeError(
                "Checkpoint CUDA RNG-state count does not match visible devices: "
                f"saved={len(cuda_states)} visible={visible}"
            )
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(state) for state in cuda_states])


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    training_state: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    atomic_torch_save(
        path,
        {
            "checkpoint_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "training_state": training_state,
            "rng_state": capture_rng_state(),
            "contract": contract,
        },
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    # Always load on CPU. Never map a full training checkpoint to CUDA.
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload: {path}")
    if payload.get("contract") != expected_contract:
        raise RuntimeError("Checkpoint run contract differs from the active run contract")

    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to_device(optimizer, device)
    scaler.load_state_dict(payload.get("scaler", {}))
    restore_rng_state(payload["rng_state"])
    return dict(payload["training_state"])


def load_model_state_only(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load only model weights for evaluation.

    Evaluation deliberately ignores optimizer, scaler and RNG state. This is
    another fail-safe against the historical CUDA RNG restoration error.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"Checkpoint has no model state: {path}")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "checkpoint_version": payload.get("checkpoint_version"),
        "training_state": payload.get("training_state", {}),
        "contract": payload.get("contract", {}),
    }

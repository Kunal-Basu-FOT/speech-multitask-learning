from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .state import TrainingState


CHECKPOINT_VERSION = 1


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()

    return value


def _restore_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])

    if (
        torch.cuda.is_available()
        and "torch_cuda" in value
    ):
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def save_training_checkpoint(
    path: Path | str,
    *,
    model,
    optimizer,
    scheduler,
    amp_controller,
    state: TrainingState,
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_controller.state_dict(),
        "training_state": state.to_dict(),
        "rng_state": _rng_state(),
        "extra": extra or {},
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        torch.save(payload, temporary_path)

        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())

        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_training_checkpoint(
    path: Path | str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    amp_controller=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[TrainingState, dict[str, Any]]:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {source}"
        )

    # PyTorch 2.6 defaults weights_only=True. This checkpoint contains
    # optimizer, scheduler, RNG, and training-state objects, so full loading
    # is explicitly requested.
    payload = torch.load(
        source,
        map_location=map_location,
        weights_only=False,
    )

    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{payload.get('checkpoint_version')}"
        )

    model.load_state_dict(payload["model"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])

    if amp_controller is not None:
        amp_controller.load_state_dict(payload["amp"])

    if restore_rng:
        _restore_rng_state(payload["rng_state"])

    state = TrainingState.from_dict(
        payload["training_state"]
    )

    return state, dict(payload.get("extra", {}))


class CheckpointManager:
    """Atomic last/best checkpoint management."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def save_last(self, **kwargs) -> Path:
        save_training_checkpoint(self.last_path, **kwargs)
        return self.last_path

    def save_best(self, **kwargs) -> Path:
        save_training_checkpoint(self.best_path, **kwargs)
        return self.best_path

    def has_last(self) -> bool:
        return self.last_path.is_file()

    def has_best(self) -> bool:
        return self.best_path.is_file()

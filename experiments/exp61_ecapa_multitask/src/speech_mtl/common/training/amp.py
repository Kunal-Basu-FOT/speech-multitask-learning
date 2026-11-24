from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


class AmpController:
    """Device-aware automatic mixed precision and gradient scaling."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = torch.device(device)
        self.device_type = self.device.type
        self.dtype = dtype
        self.enabled = bool(
            enabled and self.device_type == "cuda"
        )

        try:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.enabled,
            )
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.enabled,
            )

    def autocast(self) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer) -> None:
        self.scaler.step(optimizer)

    def update(self) -> None:
        self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "device_type": self.device_type,
            "dtype": str(self.dtype),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_enabled = bool(state.get("enabled", False))

        if stored_enabled != self.enabled:
            raise ValueError(
                "AMP enabled state differs from checkpoint"
            )

        self.scaler.load_state_dict(
            state.get("scaler", {})
        )

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainingState:
    """Serializable state for exact training resumption."""

    epoch: int = 0
    batch_in_epoch: int = 0
    global_batch: int = 0
    optimizer_step: int = 0
    phase_name: str = "not_started"
    best_dev_metric: float = float("-inf")
    best_epoch: int = 0
    completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingState":
        return cls(
            epoch=int(value.get("epoch", 0)),
            batch_in_epoch=int(value.get("batch_in_epoch", 0)),
            global_batch=int(value.get("global_batch", 0)),
            optimizer_step=int(value.get("optimizer_step", 0)),
            phase_name=str(value.get("phase_name", "not_started")),
            best_dev_metric=float(
                value.get("best_dev_metric", float("-inf"))
            ),
            best_epoch=int(value.get("best_epoch", 0)),
            completed=bool(value.get("completed", False)),
        )

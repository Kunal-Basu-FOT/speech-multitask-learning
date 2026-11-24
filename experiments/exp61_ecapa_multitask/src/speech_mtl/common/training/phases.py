from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    start_epoch: int
    end_epoch: int
    encoder_trainable: bool
    description: str

    def contains(self, epoch: int) -> bool:
        return self.start_epoch <= epoch <= self.end_epoch


class PhaseController:
    """Frozen 5 + 30 + 5 EXP 6.1 training schedule."""

    TOTAL_EPOCHS = 40

    def __init__(self) -> None:
        self.phases = (
            PhaseSpec(
                name="phase1_warmup_joint",
                start_epoch=1,
                end_epoch=5,
                encoder_trainable=True,
                description=(
                    "Joint ECAPA, task-head, and uncertainty training "
                    "during linear learning-rate warmup."
                ),
            ),
            PhaseSpec(
                name="phase2_full_joint",
                start_epoch=6,
                end_epoch=35,
                encoder_trainable=True,
                description=(
                    "Full joint multitask optimization with cosine "
                    "learning-rate decay."
                ),
            ),
            PhaseSpec(
                name="phase3_heads_only",
                start_epoch=36,
                end_epoch=40,
                encoder_trainable=False,
                description=(
                    "Frozen shared encoder; task heads and uncertainty "
                    "parameters remain trainable."
                ),
            ),
        )

    def for_epoch(self, epoch: int) -> PhaseSpec:
        epoch = int(epoch)

        for phase in self.phases:
            if phase.contains(epoch):
                return phase

        raise ValueError(
            f"Epoch must be in [1, {self.TOTAL_EPOCHS}], received {epoch}"
        )

    def apply(self, model: nn.Module, epoch: int) -> PhaseSpec:
        phase = self.for_epoch(epoch)

        if not hasattr(model, "set_encoder_trainable"):
            raise TypeError(
                "Model must implement set_encoder_trainable(bool)"
            )

        model.set_encoder_trainable(phase.encoder_trainable)

        # The caller normally invokes model.train() before this method.
        # Keep a frozen encoder in evaluation mode so dropout and batch-like
        # statistics cannot change during the heads-only phase.
        if not phase.encoder_trainable:
            encoder = getattr(model, "encoder", None)
            if encoder is None:
                raise TypeError(
                    "Model must expose an encoder module for phase 3"
                )
            encoder.eval()

        return phase

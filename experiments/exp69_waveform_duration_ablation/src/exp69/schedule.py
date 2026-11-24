from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import Config


@dataclass(frozen=True)
class Phase:
    name: str
    encoder_trainable: bool


def phase_for_epoch(epoch: int, config: Config) -> Phase:
    if epoch <= int(config.phase1_epochs):
        return Phase("phase1_warmup_joint", True)
    if epoch <= int(config.phase2_end_epoch):
        return Phase("phase2_full_joint", True)
    return Phase("phase3_heads_only", False)


class Exp69LRScheduler:
    """Step-wise EXP 6.1-compatible learning-rate schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, config: Config) -> None:
        if len(optimizer.param_groups) != 2:
            raise ValueError("EXP 6.9 optimizer must have encoder and task groups")
        self.optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.config = config
        self.phase1_steps = int(config.phase1_epochs) * self.steps_per_epoch
        self.phase2_steps = (
            int(config.phase2_end_epoch) - int(config.phase1_epochs)
        ) * self.steps_per_epoch

    def learning_rates(self, optimizer_step: int, epoch: int) -> tuple[float, float]:
        phase = phase_for_epoch(epoch, self.config)
        if not phase.encoder_trainable:
            return 0.0, float(self.config.phase3_task_lr)

        step = max(int(optimizer_step), 0)
        if step < self.phase1_steps:
            if self.phase1_steps <= 1:
                value = float(self.config.peak_lr)
            else:
                fraction = step / float(self.phase1_steps - 1)
                value = float(self.config.warmup_start_lr) + fraction * (
                    float(self.config.peak_lr) - float(self.config.warmup_start_lr)
                )
            return value, value

        phase2_position = min(max(step - self.phase1_steps, 0), max(self.phase2_steps - 1, 0))
        if self.phase2_steps <= 1:
            value = float(self.config.cosine_end_lr)
        else:
            fraction = phase2_position / float(self.phase2_steps - 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
            value = float(self.config.cosine_end_lr) + cosine * (
                float(self.config.peak_lr) - float(self.config.cosine_end_lr)
            )
        return value, value

    def apply(self, optimizer_step: int, epoch: int) -> tuple[float, float]:
        encoder_lr, task_lr = self.learning_rates(optimizer_step, epoch)
        self.optimizer.param_groups[0]["lr"] = encoder_lr
        self.optimizer.param_groups[1]["lr"] = task_lr
        return encoder_lr, task_lr

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.optim import AdamW, Optimizer


@dataclass(frozen=True)
class OptimizerConfig:
    peak_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerConfig:
    total_epochs: int = 40
    warmup_epochs: int = 5
    joint_cosine_end_epoch: int = 35
    warmup_start_learning_rate: float = 1.0e-5
    peak_learning_rate: float = 1.0e-4
    cosine_minimum_learning_rate: float = 5.0e-6
    heads_only_learning_rate: float = 2.0e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_optimizer(
    model,
    config: OptimizerConfig | None = None,
) -> AdamW:
    config = config or OptimizerConfig()

    encoder_parameters = list(model.encoder_parameters())
    task_parameters = list(model.task_parameters())

    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    task_ids = {id(parameter) for parameter in task_parameters}

    if encoder_ids & task_ids:
        raise RuntimeError(
            "Encoder and task optimizer groups overlap"
        )

    model_parameter_ids = {
        id(parameter) for parameter in model.parameters()
    }

    if encoder_ids | task_ids != model_parameter_ids:
        missing = len(
            model_parameter_ids - (encoder_ids | task_ids)
        )
        extra = len(
            (encoder_ids | task_ids) - model_parameter_ids
        )
        raise RuntimeError(
            "Optimizer groups do not exactly cover model parameters: "
            f"missing={missing}, extra={extra}"
        )

    return AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.peak_learning_rate,
                "weight_decay": config.weight_decay,
                "group_name": "encoder",
            },
            {
                "params": task_parameters,
                "lr": config.peak_learning_rate,
                "weight_decay": config.weight_decay,
                "group_name": "tasks",
            },
        ],
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )


class ThreePhaseLRScheduler:
    """Optimizer-step scheduler matching the 5 + 30 + 5 plan."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        optimizer_steps_per_epoch: int,
        config: SchedulerConfig | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or SchedulerConfig()
        self.optimizer_steps_per_epoch = int(
            optimizer_steps_per_epoch
        )

        if self.optimizer_steps_per_epoch <= 0:
            raise ValueError(
                "optimizer_steps_per_epoch must be positive"
            )

        if self.config.total_epochs != 40:
            raise ValueError("EXP 6.1 total_epochs must remain 40")

        if self.config.warmup_epochs != 5:
            raise ValueError("EXP 6.1 warmup_epochs must remain 5")

        if self.config.joint_cosine_end_epoch != 35:
            raise ValueError(
                "EXP 6.1 joint cosine phase must end at epoch 35"
            )

        group_names = {
            group.get("group_name")
            for group in self.optimizer.param_groups
        }

        if group_names != {"encoder", "tasks"}:
            raise ValueError(
                "Optimizer must contain encoder and tasks groups"
            )

        self.warmup_steps = (
            self.config.warmup_epochs
            * self.optimizer_steps_per_epoch
        )
        self.joint_steps = (
            (
                self.config.joint_cosine_end_epoch
                - self.config.warmup_epochs
            )
            * self.optimizer_steps_per_epoch
        )
        self.total_steps = (
            self.config.total_epochs
            * self.optimizer_steps_per_epoch
        )
        self.heads_only_start_step = (
            self.config.joint_cosine_end_epoch
            * self.optimizer_steps_per_epoch
        )
        self.step_number = 0
        self._apply_learning_rates()

    @staticmethod
    def _linear_interpolate(
        start: float,
        end: float,
        fraction: float,
    ) -> float:
        return start + (end - start) * fraction

    def _joint_learning_rate(self, step_number: int) -> float:
        if step_number < self.warmup_steps:
            denominator = max(self.warmup_steps - 1, 1)
            fraction = min(
                max(step_number / denominator, 0.0),
                1.0,
            )
            return self._linear_interpolate(
                self.config.warmup_start_learning_rate,
                self.config.peak_learning_rate,
                fraction,
            )

        if step_number < self.heads_only_start_step:
            cosine_step = step_number - self.warmup_steps
            denominator = max(self.joint_steps - 1, 1)
            fraction = min(
                max(cosine_step / denominator, 0.0),
                1.0,
            )
            cosine_factor = 0.5 * (
                1.0 + math.cos(math.pi * fraction)
            )
            return (
                self.config.cosine_minimum_learning_rate
                + (
                    self.config.peak_learning_rate
                    - self.config.cosine_minimum_learning_rate
                )
                * cosine_factor
            )

        return self.config.heads_only_learning_rate

    def learning_rates_for_step(
        self,
        step_number: int,
    ) -> dict[str, float]:
        joint_learning_rate = self._joint_learning_rate(
            int(step_number)
        )

        if step_number >= self.heads_only_start_step:
            return {
                "encoder": 0.0,
                "tasks": self.config.heads_only_learning_rate,
            }

        return {
            "encoder": joint_learning_rate,
            "tasks": joint_learning_rate,
        }

    def _apply_learning_rates(self) -> None:
        learning_rates = self.learning_rates_for_step(
            self.step_number
        )

        for group in self.optimizer.param_groups:
            name = group["group_name"]
            group["lr"] = learning_rates[name]

    def step(self) -> None:
        if self.step_number >= self.total_steps:
            raise RuntimeError(
                "Learning-rate scheduler stepped beyond "
                "the configured 40 epochs"
            )

        self.step_number += 1
        self._apply_learning_rates()

    def current_learning_rates(self) -> dict[str, float]:
        return {
            str(group["group_name"]): float(group["lr"])
            for group in self.optimizer.param_groups
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "optimizer_steps_per_epoch": (
                self.optimizer_steps_per_epoch
            ),
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stored_steps_per_epoch = int(
            state["optimizer_steps_per_epoch"]
        )

        if (
            stored_steps_per_epoch
            != self.optimizer_steps_per_epoch
        ):
            raise ValueError(
                "Checkpoint optimizer_steps_per_epoch differs: "
                f"checkpoint={stored_steps_per_epoch}, "
                f"current={self.optimizer_steps_per_epoch}"
            )

        stored_config = SchedulerConfig(
            **state["config"]
        )

        if stored_config != self.config:
            raise ValueError(
                "Checkpoint scheduler configuration differs"
            )

        self.step_number = int(state["step_number"])

        if not 0 <= self.step_number <= self.total_steps:
            raise ValueError(
                f"Invalid restored scheduler step: "
                f"{self.step_number}"
            )

        self._apply_learning_rates()

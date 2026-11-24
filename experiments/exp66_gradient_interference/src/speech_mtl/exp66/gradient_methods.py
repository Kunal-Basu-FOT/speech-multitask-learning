"""Gradient diagnostics, PCGrad projection, and a stable GradNorm controller."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import nn

from .model import TASKS

GradientList = list[torch.Tensor | None]


def collect_parameter_gradients(
    losses: dict[str, torch.Tensor],
    parameters: list[nn.Parameter],
    scales: dict[str, torch.Tensor] | None = None,
) -> dict[str, GradientList]:
    result: dict[str, GradientList] = {}
    for task in TASKS:
        objective = losses[task]
        if scales is not None:
            objective = scales[task] * objective
        gradients = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        result[task] = [None if grad is None else grad.detach() for grad in gradients]
    return result


def _dot(left: GradientList, right: GradientList) -> torch.Tensor:
    value = None
    for first, second in zip(left, right, strict=True):
        if first is None or second is None:
            continue
        term = (first * second).sum()
        value = term if value is None else value + term
    if value is None:
        raise RuntimeError("Gradient lists have no overlapping tensors")
    return value


def _norm_squared(gradients: GradientList) -> torch.Tensor:
    value = None
    for gradient in gradients:
        if gradient is None:
            continue
        term = gradient.square().sum()
        value = term if value is None else value + term
    if value is None:
        raise RuntimeError("Gradient list is empty")
    return value


def gradient_norm(gradients: GradientList) -> float:
    return float(torch.sqrt(_norm_squared(gradients).clamp_min(0.0)).item())


def cosine(left: GradientList, right: GradientList) -> float:
    numerator = _dot(left, right)
    denominator = torch.sqrt(_norm_squared(left) * _norm_squared(right)).clamp_min(1e-20)
    return float((numerator / denominator).item())


def pairwise_statistics(gradients: dict[str, GradientList]) -> list[dict]:
    rows: list[dict] = []
    for first_index, first in enumerate(TASKS):
        for second in TASKS[first_index + 1 :]:
            value = cosine(gradients[first], gradients[second])
            rows.append(
                {
                    "task_a": first,
                    "task_b": second,
                    "cosine": value,
                    "conflict": int(value < 0.0),
                    "norm_a": gradient_norm(gradients[first]),
                    "norm_b": gradient_norm(gradients[second]),
                }
            )
    return rows


def _clone(gradients: GradientList) -> GradientList:
    return [None if value is None else value.clone() for value in gradients]


def pcgrad_project(
    task_gradients: dict[str, GradientList], *, seed: int
) -> tuple[dict[str, GradientList], GradientList, int]:
    """Apply deterministic PCGrad to shared-encoder task gradients.

    Each task gradient is projected against the original gradients of the other
    tasks in a deterministic shuffled order. The resulting task gradients are
    summed, matching the scale of the standard summed multitask objective.
    """
    originals = {task: task_gradients[task] for task in TASKS}
    projected = {task: _clone(task_gradients[task]) for task in TASKS}
    rng = random.Random(int(seed))
    projection_count = 0
    for task in TASKS:
        others = [candidate for candidate in TASKS if candidate != task]
        rng.shuffle(others)
        current = projected[task]
        for other in others:
            reference = originals[other]
            dot_value = _dot(current, reference)
            norm_value = _norm_squared(reference).clamp_min(1e-20)
            if float(dot_value.item()) < 0.0:
                coefficient = dot_value / norm_value
                for index, (current_tensor, reference_tensor) in enumerate(
                    zip(current, reference, strict=True)
                ):
                    if current_tensor is None or reference_tensor is None:
                        continue
                    current[index] = current_tensor - coefficient * reference_tensor
                projection_count += 1
    combined: GradientList = []
    for parameter_index in range(len(next(iter(projected.values())))):
        available = [
            projected[task][parameter_index]
            for task in TASKS
            if projected[task][parameter_index] is not None
        ]
        combined.append(None if not available else torch.stack(available, dim=0).sum(dim=0))
    return projected, combined, projection_count


def assign_gradients(parameters: list[nn.Parameter], gradients: GradientList) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = None if gradient is None else gradient.to(parameter.dtype)


class GradNormController(nn.Module):
    """Three-task GradNorm controller using shared-embedding gradient norms.

    The task weights are positive and normalized to sum to three. Gradient norms
    are measured with respect to the shared 192-dimensional utterance embedding,
    which is an explicit low-cost proxy for the shared encoder gradient magnitude.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(len(TASKS)))
        self.alpha = float(alpha)
        self.register_buffer("initial_losses", torch.full((len(TASKS),), float("nan")))

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0) * float(len(TASKS))

    def initialize_if_needed(self, raw_losses: dict[str, torch.Tensor]) -> None:
        if torch.isnan(self.initial_losses).any():
            values = torch.stack([raw_losses[task].detach() for task in TASKS])
            self.initial_losses.copy_(values.clamp_min(1e-8))

    def base_embedding_gradient_norms(
        self, raw_losses: dict[str, torch.Tensor], embedding: torch.Tensor
    ) -> torch.Tensor:
        norms = []
        for task in TASKS:
            gradient = torch.autograd.grad(
                raw_losses[task],
                embedding,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            norms.append(gradient.detach().norm(p=2).clamp_min(1e-12))
        return torch.stack(norms)

    def balancing_loss(
        self, raw_losses: dict[str, torch.Tensor], base_norms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.initialize_if_needed(raw_losses)
        weights = self.weights()
        weighted_norms = weights * base_norms.detach()
        current_losses = torch.stack([raw_losses[task].detach() for task in TASKS])
        loss_ratios = current_losses / self.initial_losses.clamp_min(1e-8)
        inverse_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
        target = weighted_norms.detach().mean() * inverse_rates.pow(self.alpha)
        loss = torch.abs(weighted_norms - target.detach()).sum()
        return loss, weighted_norms.detach(), target.detach()

    def normalized_weights_list(self) -> list[float]:
        return [float(value) for value in self.weights().detach().cpu().tolist()]


@dataclass
class DiagnosticAccumulator:
    samples: list[dict]

    @classmethod
    def create(cls) -> "DiagnosticAccumulator":
        return cls(samples=[])

    def add(
        self,
        *,
        epoch: int,
        batch: int,
        phase: str,
        pre_rows: list[dict],
        post_rows: list[dict] | None = None,
        projection_count: int = 0,
    ) -> None:
        post_map = {}
        if post_rows is not None:
            post_map = {
                (row["task_a"], row["task_b"]): row["cosine"] for row in post_rows
            }
        for row in pre_rows:
            key = (row["task_a"], row["task_b"])
            self.samples.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "phase": phase,
                    **row,
                    "cosine_post": post_map.get(key, ""),
                    "projection_count_step": projection_count,
                }
            )

    def epoch_summary(self, epoch: int, phase: str) -> list[dict]:
        rows = [row for row in self.samples if int(row["epoch"]) == int(epoch)]
        result: list[dict] = []
        for first_index, first in enumerate(TASKS):
            for second in TASKS[first_index + 1 :]:
                selected = [
                    row for row in rows if row["task_a"] == first and row["task_b"] == second
                ]
                if not selected:
                    continue
                pre = [float(row["cosine"]) for row in selected]
                post = [
                    float(row["cosine_post"])
                    for row in selected
                    if row["cosine_post"] != ""
                ]
                result.append(
                    {
                        "epoch": epoch,
                        "phase": phase,
                        "task_a": first,
                        "task_b": second,
                        "samples": len(selected),
                        "mean_cosine_pre": sum(pre) / len(pre),
                        "conflict_rate_pre": sum(value < 0.0 for value in pre) / len(pre),
                        "mean_cosine_post": "" if not post else sum(post) / len(post),
                        "conflict_rate_post": "" if not post else sum(value < 0.0 for value in post) / len(post),
                        "mean_norm_a": sum(float(row["norm_a"]) for row in selected) / len(selected),
                        "mean_norm_b": sum(float(row["norm_b"]) for row in selected) / len(selected),
                        "mean_projection_count_step": sum(
                            int(row["projection_count_step"]) for row in selected
                        ) / len(selected),
                    }
                )
        return result

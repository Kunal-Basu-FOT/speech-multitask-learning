from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class HomoscedasticTaskWeighting(nn.Module):
    """Learned uncertainty weighting for language, dialect, and gender.

    For each task with loss L and learned log variance s:

        weighted_loss = exp(-s) * L + s

    Log variances are clamped only in the computation graph, not modified
    in place, preserving optimizer state and checkpoint reproducibility.
    """

    TASKS = ("language", "dialect", "gender")

    def __init__(
        self,
        *,
        initial_log_variance: float = 0.0,
        minimum_log_variance: float = -6.0,
        maximum_log_variance: float = 6.0,
    ) -> None:
        super().__init__()

        if minimum_log_variance >= maximum_log_variance:
            raise ValueError(
                "minimum_log_variance must be smaller than "
                "maximum_log_variance"
            )

        self.minimum_log_variance = float(minimum_log_variance)
        self.maximum_log_variance = float(maximum_log_variance)

        self.log_variances = nn.ParameterDict(
            {
                task: nn.Parameter(
                    torch.tensor(
                        float(initial_log_variance),
                        dtype=torch.float32,
                    )
                )
                for task in self.TASKS
            }
        )

    def forward(
        self,
        losses: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = sorted(set(self.TASKS) - set(losses))
        extra = sorted(set(losses) - set(self.TASKS))

        if missing or extra:
            raise ValueError(
                f"Loss mapping mismatch; missing={missing}, extra={extra}"
            )

        weighted_terms: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None

        for task in self.TASKS:
            task_loss = losses[task]

            if task_loss.ndim != 0:
                raise ValueError(
                    f"{task} loss must be a scalar tensor, "
                    f"received {tuple(task_loss.shape)}"
                )

            clamped_log_variance = torch.clamp(
                self.log_variances[task],
                min=self.minimum_log_variance,
                max=self.maximum_log_variance,
            )

            term = (
                torch.exp(-clamped_log_variance) * task_loss
                + clamped_log_variance
            )

            weighted_terms[task] = term
            total = term if total is None else total + term

        if total is None:
            raise RuntimeError("No task losses were supplied")

        return total, weighted_terms

    def values(self) -> dict[str, float]:
        return {
            task: float(parameter.detach().cpu().item())
            for task, parameter in self.log_variances.items()
        }

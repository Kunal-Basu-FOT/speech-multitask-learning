from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MultitaskLossOutput:
    total: torch.Tensor
    raw_losses: dict[str, torch.Tensor]
    weighted_terms: dict[str, torch.Tensor]
    dialect_losses_by_language: dict[str, torch.Tensor]
    dialect_examples: int


def _validate_target_shape(
    name: str,
    target: torch.Tensor,
    batch_size: int,
) -> None:
    if target.shape != (batch_size,):
        raise ValueError(
            f"{name} target must have shape ({batch_size},), "
            f"received {tuple(target.shape)}"
        )

    if target.dtype != torch.long:
        raise ValueError(
            f"{name} target must use torch.long, received {target.dtype}"
        )


def compute_oracle_dialect_loss(
    *,
    dialect_logits: dict[str, torch.Tensor],
    dialect_batch_indices: dict[str, torch.Tensor],
    dialect_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    if set(dialect_logits) != set(dialect_batch_indices):
        raise ValueError(
            "Dialect logit and routing-index language sets differ"
        )

    total_loss_sum: torch.Tensor | None = None
    total_examples = 0
    losses_by_language: dict[str, torch.Tensor] = {}

    for language in sorted(dialect_logits):
        logits = dialect_logits[language]
        indices = dialect_batch_indices[language]

        if logits.ndim != 2:
            raise ValueError(
                f"{language} dialect logits must be 2-D, "
                f"received {tuple(logits.shape)}"
            )

        if indices.ndim != 1:
            raise ValueError(
                f"{language} routing indices must be 1-D, "
                f"received {tuple(indices.shape)}"
            )

        if int(logits.shape[0]) != int(indices.numel()):
            raise ValueError(
                f"{language}: routed example count and logits differ"
            )

        selected_targets = dialect_targets.index_select(0, indices)

        if torch.any(selected_targets < 0) or torch.any(
            selected_targets >= logits.shape[1]
        ):
            raise ValueError(
                f"{language}: dialect target outside local head range"
            )

        language_loss_sum = F.cross_entropy(
            logits,
            selected_targets,
            reduction="sum",
        )

        count = int(indices.numel())
        losses_by_language[language] = language_loss_sum / count
        total_examples += count
        total_loss_sum = (
            language_loss_sum
            if total_loss_sum is None
            else total_loss_sum + language_loss_sum
        )

    if total_loss_sum is None or total_examples == 0:
        raise ValueError("No dialect examples were routed")

    return (
        total_loss_sum / total_examples,
        losses_by_language,
        total_examples,
    )


def compute_multitask_loss(
    *,
    model,
    output,
    language_targets: torch.Tensor,
    dialect_targets: torch.Tensor,
    gender_targets: torch.Tensor,
) -> MultitaskLossOutput:
    batch_size = int(output.language_logits.shape[0])

    _validate_target_shape(
        "language",
        language_targets,
        batch_size,
    )
    _validate_target_shape(
        "dialect",
        dialect_targets,
        batch_size,
    )
    _validate_target_shape(
        "gender",
        gender_targets,
        batch_size,
    )

    if output.language_logits.shape != (batch_size, 8):
        raise ValueError(
            "Language logits must have shape "
            f"({batch_size}, 8), received "
            f"{tuple(output.language_logits.shape)}"
        )

    if output.gender_logits.shape != (batch_size, 2):
        raise ValueError(
            "Gender logits must have shape "
            f"({batch_size}, 2), received "
            f"{tuple(output.gender_logits.shape)}"
        )

    language_loss = F.cross_entropy(
        output.language_logits,
        language_targets,
    )
    gender_loss = F.cross_entropy(
        output.gender_logits,
        gender_targets,
    )

    (
        dialect_loss,
        dialect_losses_by_language,
        dialect_examples,
    ) = compute_oracle_dialect_loss(
        dialect_logits=output.dialect_logits,
        dialect_batch_indices=output.dialect_batch_indices,
        dialect_targets=dialect_targets,
    )

    if dialect_examples != batch_size:
        raise RuntimeError(
            f"Dialect routing covered {dialect_examples} of "
            f"{batch_size} examples"
        )

    raw_losses = {
        "language": language_loss,
        "dialect": dialect_loss,
        "gender": gender_loss,
    }

    total, weighted_terms = model.task_weighting(raw_losses)

    if total.ndim != 0 or not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite uncertainty-weighted multitask loss"
        )

    return MultitaskLossOutput(
        total=total,
        raw_losses=raw_losses,
        weighted_terms=weighted_terms,
        dialect_losses_by_language=dialect_losses_by_language,
        dialect_examples=dialect_examples,
    )

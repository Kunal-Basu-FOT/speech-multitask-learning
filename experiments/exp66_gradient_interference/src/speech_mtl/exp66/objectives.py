from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import TASKS, HardSharingMTLModel


def compute_raw_losses(model: HardSharingMTLModel, batch: dict, output: dict) -> dict[str, torch.Tensor]:
    lid = F.cross_entropy(output["lid_logits"], batch["lid"])
    gender = F.cross_entropy(output["gender_logits"], batch["gender"])
    dialect_sum = output["lid_logits"].new_zeros(())
    dialect_count = 0
    for _, (indices, logits) in output["dialect_logits"].items():
        dialect_sum = dialect_sum + F.cross_entropy(
            logits, batch["dialect"][indices], reduction="sum"
        )
        dialect_count += int(indices.numel())
    if dialect_count == 0:
        raise RuntimeError("No dialect examples were routed")
    dialect = dialect_sum / dialect_count
    return {"lid": lid, "dialect": dialect, "gender": gender}


def uncertainty_terms(
    model: HardSharingMTLModel, raw_losses: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    terms: dict[str, torch.Tensor] = {}
    for task in TASKS:
        value = model.log_vars[task].clamp(-6.0, 6.0)
        terms[task] = torch.exp(-value) * raw_losses[task] + value
    return terms


def uncertainty_total(
    model: HardSharingMTLModel, raw_losses: dict[str, torch.Tensor]
) -> torch.Tensor:
    terms = uncertainty_terms(model, raw_losses)
    return sum(terms.values())


def detached_uncertainty_scales(model: HardSharingMTLModel) -> dict[str, torch.Tensor]:
    return {
        task: torch.exp(-model.log_vars[task].clamp(-6.0, 6.0)).detach()
        for task in TASKS
    }


def weighted_total(
    raw_losses: dict[str, torch.Tensor], weights: torch.Tensor
) -> torch.Tensor:
    return sum(weights[index] * raw_losses[task] for index, task in enumerate(TASKS))

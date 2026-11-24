"""Active-task loss computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import ReducedMTLModel


def compute_losses(model: ReducedMTLModel, batch: dict, output: dict) -> dict:
    raw: dict[str, torch.Tensor] = {}
    if "lid" in model.condition.active_tasks:
        raw["lid"] = F.cross_entropy(output["lid_logits"], batch["lid"])
    if "dialect" in model.condition.active_tasks:
        total = output["embedding"].new_zeros(())
        count = 0
        for _lang, (indices, logits) in output["dialect_logits"].items():
            targets = batch["dialect"][indices]
            total = total + F.cross_entropy(logits, targets, reduction="sum")
            count += int(targets.numel())
        if count == 0:
            raise RuntimeError("No dialect examples in batch")
        raw["dialect"] = total / count

    if len(raw) == 1:
        total_loss = next(iter(raw.values()))
        weighted = dict(raw)
    else:
        total_loss = output["embedding"].new_zeros(())
        weighted = {}
        for task, task_loss in raw.items():
            effective = model.log_vars[task].clamp(-6.0, 6.0)
            term = torch.exp(-effective) * task_loss + effective
            weighted[task] = term
            total_loss = total_loss + term
    return {"total": total_loss, "raw": raw, "weighted": weighted}

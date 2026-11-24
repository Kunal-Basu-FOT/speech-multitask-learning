from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_losses(model, batch: dict, output: dict) -> dict:
    raw = {
        "lid": F.cross_entropy(output["lid_logits"], batch["lid"]),
        "gender": F.cross_entropy(output["gender_logits"], batch["gender"]),
    }

    dialect_sum = output["lid_logits"].new_zeros(())
    dialect_examples = 0
    for _, (indices, logits) in output["dialect_logits"].items():
        dialect_sum = dialect_sum + F.cross_entropy(
            logits,
            batch["dialect"][indices],
            reduction="sum",
        )
        dialect_examples += int(indices.numel())
    if dialect_examples == 0:
        raise RuntimeError("No dialect examples were routed to local heads")
    raw["dialect"] = dialect_sum / dialect_examples

    weighted = {}
    total = output["lid_logits"].new_zeros(())
    for task, task_loss in raw.items():
        log_variance = model.log_vars[task].clamp(-6.0, 6.0)
        term = torch.exp(-log_variance) * task_loss + log_variance
        weighted[task] = term
        total = total + term

    return {"total": total, "raw": raw, "weighted": weighted}

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import load_config
from .labels import LabelSpace
from .losses import compute_losses
from .model import CONDITION_NAME, HierarchicalLIDDialectModel


def synthetic_labels() -> LabelSpace:
    return LabelSpace.from_dict(
        {
            "languages": ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"],
            "gender_values": ["0", "1"],
            "dialect_values_by_language": {
                "bh": ["0", "1", "2"],
                "bn": ["0", "1", "2", "3", "4"],
                "ch": ["0", "1", "2", "3"],
                "kn": ["0", "1", "2", "3", "4"],
                "mg": ["0", "1", "2", "3"],
                "mr": ["0", "1", "2", "3"],
                "mt": ["0", "1", "2", "3"],
                "te": ["0", "1", "2", "3"],
            },
        }
    )


def synthetic_batch(labels: LabelSpace, device: torch.device) -> dict:
    batch_size = len(labels.languages)
    frames = 80
    return {
        "features": torch.randn(batch_size, frames, 768, device=device),
        "relative_lengths": torch.ones(batch_size, device=device),
        "valid_mask": torch.ones(
            batch_size, frames, dtype=torch.bool, device=device
        ),
        "frame_lengths": torch.full(
            (batch_size,), frames, dtype=torch.long, device=device
        ),
        "lid": torch.arange(batch_size, dtype=torch.long, device=device),
        "gender": torch.arange(batch_size, device=device) % 2,
        "dialect": torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 3],
            dtype=torch.long,
            device=device,
        ),
        "language": list(labels.languages),
        "utt_id": [str(index) for index in range(batch_size)],
        "filepath": ["synthetic"] * batch_size,
        "dataset_index": torch.arange(batch_size, device=device),
    }


def _dialect_only_loss(output: dict, batch: dict) -> torch.Tensor:
    total = output["lid_logits"].new_zeros(())
    count = 0
    for _, (indices, logits) in output["dialect_logits"].items():
        total = total + F.cross_entropy(
            logits,
            batch["dialect"][indices],
            reduction="sum",
        )
        count += int(indices.numel())
    return total / count


def run(config_path: str, device: torch.device | None = None) -> dict[str, int]:
    cfg = load_config(config_path)
    labels = synthetic_labels()
    target_device = device or torch.device("cpu")

    model = HierarchicalLIDDialectModel(cfg.model, labels).to(target_device)
    batch = synthetic_batch(labels, target_device)
    output = model(batch)

    assert output["embedding"].shape == (8, 192)
    assert output["lid_logits"].shape == (8, 8)
    assert output["lid_probabilities"].shape == (8, 8)
    assert output["dialect_input"].shape == (8, 200)
    assert output["gender_logits"].shape == (8, 2)
    assert torch.allclose(
        output["lid_probabilities"].sum(dim=-1),
        torch.ones(8, device=target_device),
        atol=1e-5,
    )
    assert not output["dialect_condition"].requires_grad

    # Verify the defining stop-gradient contract. Dialect loss may update the
    # shared encoder and dialect heads, but never the LID head through the
    # posterior-conditioning branch.
    model.zero_grad(set_to_none=True)
    dialect_loss = _dialect_only_loss(output, batch)
    dialect_loss.backward()
    lid_gradients = [parameter.grad for parameter in model.lid_head.parameters()]
    assert all(
        gradient is None or torch.count_nonzero(gradient).item() == 0
        for gradient in lid_gradients
    ), "Dialect loss leaked gradients into the LID head"
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.dialect_heads.parameters()
    )

    model.zero_grad(set_to_none=True)
    output = model(batch)
    losses = compute_losses(model, batch, output)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.lid_head.parameters())

    counts = model.parameter_counts
    assert 7_700_000 <= counts["total"] <= 7_800_000, counts
    assert counts["total"] == counts["trainable"], counts
    print("PARAMETER_COUNTS", counts)
    print("EXP67_STOP_GRADIENT_VALIDATION_PASS")
    print("EXP67_SYNTHETIC_VALIDATION_PASS")
    return counts

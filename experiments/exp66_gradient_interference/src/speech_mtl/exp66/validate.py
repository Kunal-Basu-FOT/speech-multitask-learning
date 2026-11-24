from __future__ import annotations

import torch

from .config import load_config
from .gradient_methods import (
    GradNormController,
    assign_gradients,
    collect_parameter_gradients,
    cosine,
    pairwise_statistics,
    pcgrad_project,
)
from .labels import LabelSpace
from .model import CONDITIONS, TASKS, HardSharingMTLModel
from .objectives import (
    compute_raw_losses,
    detached_uncertainty_scales,
    uncertainty_total,
    weighted_total,
)


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
        "valid_mask": torch.ones(batch_size, frames, dtype=torch.bool, device=device),
        "frame_lengths": torch.full((batch_size,), frames, dtype=torch.long, device=device),
        "lid": torch.arange(batch_size, dtype=torch.long, device=device),
        "gender": torch.arange(batch_size, device=device) % 2,
        "dialect": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long, device=device),
        "language": list(labels.languages),
        "utt_id": [str(index) for index in range(batch_size)],
        "filepath": ["synthetic"] * batch_size,
        "dataset_index": torch.arange(batch_size, device=device),
    }


def _validate_explicit_pcgrad_geometry(device: torch.device) -> None:
    gradients = {
        "lid": [torch.tensor([1.0, 0.0], device=device)],
        "dialect": [torch.tensor([-1.0, 1.0], device=device)],
        "gender": [torch.tensor([0.0, 1.0], device=device)],
    }
    assert cosine(gradients["lid"], gradients["dialect"]) < 0.0
    projected, combined, projections = pcgrad_project(gradients, seed=1337)
    assert projections > 0
    assert combined[0] is not None
    for first_index, first in enumerate(TASKS):
        for second in TASKS[first_index + 1 :]:
            assert cosine(projected[first], gradients[second]) >= -1.000001
    print("EXP66_PCGRAD_GEOMETRY_VALIDATION_PASS")


def run(config_path: str, device: torch.device | None = None) -> dict[str, int]:
    cfg = load_config(config_path)
    labels = synthetic_labels()
    target_device = device or torch.device("cpu")
    _validate_explicit_pcgrad_geometry(target_device)

    expected_total = 7_723_170
    observed_counts = None
    for condition in CONDITIONS:
        model = HardSharingMTLModel(cfg.model, labels).to(target_device)
        model.configure_condition(condition)
        batch = synthetic_batch(labels, target_device)
        output = model(batch)
        assert output["embedding"].shape == (8, 192)
        assert output["lid_logits"].shape == (8, 8)
        assert output["gender_logits"].shape == (8, 2)
        assert set(output["dialect_logits"]) == set(labels.languages)
        raw_losses = compute_raw_losses(model, batch, output)
        assert set(raw_losses) == set(TASKS)
        assert all(torch.isfinite(value) for value in raw_losses.values())

        encoder_params = list(model.encoder_parameters())
        if condition == "standard_logged":
            gradients = collect_parameter_gradients(
                raw_losses,
                encoder_params,
                scales=detached_uncertainty_scales(model),
            )
            assert len(pairwise_statistics(gradients)) == 3
            total = uncertainty_total(model, raw_losses)
            total.backward()
        elif condition == "pcgrad":
            gradients = collect_parameter_gradients(
                raw_losses,
                encoder_params,
                scales=detached_uncertainty_scales(model),
            )
            projected, combined, _ = pcgrad_project(gradients, seed=1337)
            assert len(pairwise_statistics(projected)) == 3
            total = uncertainty_total(model, raw_losses)
            total.backward()
            assign_gradients(encoder_params, combined)
            assert any(parameter.grad is not None for parameter in encoder_params)
        else:
            controller = GradNormController(cfg.training.gradnorm_alpha).to(target_device)
            base_norms = controller.base_embedding_gradient_norms(raw_losses, output["embedding"])
            balance_loss, weighted_norms, target = controller.balancing_loss(raw_losses, base_norms)
            assert base_norms.shape == (3,)
            assert weighted_norms.shape == (3,)
            assert target.shape == (3,)
            balance_loss.backward()
            assert controller.log_weights.grad is not None
            weights = controller.weights().detach()
            assert torch.all(weights > 0)
            assert torch.allclose(weights.sum(), torch.tensor(3.0, device=target_device), atol=1e-5)
            total = weighted_total(raw_losses, weights)
            total.backward()

        counts = model.parameter_counts
        assert counts["total"] == expected_total, counts
        if condition == "gradnorm":
            assert counts["trainable"] == expected_total - 3, counts
        else:
            assert counts["trainable"] == expected_total, counts
        assert any(parameter.grad is not None for parameter in model.lid_head.parameters())
        assert any(parameter.grad is not None for parameter in model.dialect_heads.parameters())
        assert any(parameter.grad is not None for parameter in model.gender_head.parameters())
        observed_counts = counts
        print(f"EXP66_{condition.upper()}_SYNTHETIC_PASS")

    assert observed_counts is not None
    print("PARAMETER_COUNTS", observed_counts)
    print("EXP66_SYNTHETIC_VALIDATION_PASS")
    print("EXP66_VALIDATION_PASS")
    return observed_counts

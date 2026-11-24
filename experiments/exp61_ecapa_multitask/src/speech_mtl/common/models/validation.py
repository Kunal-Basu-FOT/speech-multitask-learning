from __future__ import annotations

from typing import Any

import torch

from .audit import audit_model_parameters
from .factory import build_exp61_model


def validate_model_forward() -> dict[str, Any]:
    """Run a deterministic CPU parameter and forward-contract check."""

    torch.manual_seed(1337)

    model = build_exp61_model()
    model.eval()

    report = audit_model_parameters(model, enforce_exact=True)

    languages = ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"]
    batch_size = len(languages)
    maximum_frames = 64
    frame_lengths = torch.tensor(
        [64, 61, 58, 55, 52, 49, 46, 43],
        dtype=torch.long,
    )
    relative_lengths = frame_lengths.to(torch.float32) / maximum_frames

    valid_mask = (
        torch.arange(maximum_frames).unsqueeze(0)
        < frame_lengths.unsqueeze(1)
    )

    features = torch.randn(
        batch_size,
        maximum_frames,
        768,
        dtype=torch.float32,
    )
    features = features.masked_fill(
        ~valid_mask.unsqueeze(-1),
        0.0,
    )

    with torch.no_grad():
        output = model(
            features=features,
            relative_lengths=relative_lengths,
            oracle_languages=languages,
            valid_mask=valid_mask,
        )

    if output.embeddings.shape != (batch_size, 192):
        raise RuntimeError(
            f"Unexpected embedding shape: {tuple(output.embeddings.shape)}"
        )

    if output.language_logits.shape != (batch_size, 8):
        raise RuntimeError(
            "Unexpected language-logit shape: "
            f"{tuple(output.language_logits.shape)}"
        )

    if output.gender_logits.shape != (batch_size, 2):
        raise RuntimeError(
            "Unexpected gender-logit shape: "
            f"{tuple(output.gender_logits.shape)}"
        )

    expected_dialect_classes = {
        "bh": 3,
        "bn": 5,
        "ch": 4,
        "kn": 5,
        "mg": 4,
        "mr": 4,
        "mt": 4,
        "te": 4,
    }

    for language, number_of_classes in expected_dialect_classes.items():
        logits = output.dialect_logits[language]
        indices = output.dialect_batch_indices[language]

        if logits.shape != (1, number_of_classes):
            raise RuntimeError(
                f"Unexpected {language} dialect shape: "
                f"{tuple(logits.shape)}"
            )

        if indices.shape != (1,):
            raise RuntimeError(
                f"Unexpected {language} routing-index shape"
            )

    losses = {
        "language": output.language_logits.mean().square(),
        "dialect": sum(
            tensor.mean().square()
            for tensor in output.dialect_logits.values()
        ),
        "gender": output.gender_logits.mean().square(),
    }

    total_loss, weighted_terms = model.task_weighting(losses)

    if total_loss.ndim != 0 or not torch.isfinite(total_loss):
        raise RuntimeError("Uncertainty-weighted loss is invalid")

    if set(weighted_terms) != {"language", "dialect", "gender"}:
        raise RuntimeError("Unexpected uncertainty-weighting task set")

    model.set_encoder_trainable(False)
    frozen_encoder_parameters = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    )

    if frozen_encoder_parameters != 0:
        raise RuntimeError("Encoder freeze operation failed")

    trainable_task_parameters = sum(
        parameter.numel()
        for parameter in model.task_parameters()
        if parameter.requires_grad
    )

    if trainable_task_parameters <= 0:
        raise RuntimeError("Task parameters were unexpectedly frozen")

    model.set_encoder_trainable(True)

    return {
        "status": "PASS",
        "parameter_report": report.to_dict(),
        "forward_contract": {
            "batch_size": batch_size,
            "maximum_frames": maximum_frames,
            "feature_dimension": 768,
            "embedding_shape": list(output.embeddings.shape),
            "language_logits_shape": list(
                output.language_logits.shape
            ),
            "gender_logits_shape": list(
                output.gender_logits.shape
            ),
            "dialect_shapes": {
                language: list(tensor.shape)
                for language, tensor
                in output.dialect_logits.items()
            },
        },
        "oracle_routing_covered_examples": sum(
            int(indices.numel())
            for indices
            in output.dialect_batch_indices.values()
        ),
        "task_log_variances": model.task_weighting.values(),
        "phase3_encoder_freeze_check": "PASS",
    }

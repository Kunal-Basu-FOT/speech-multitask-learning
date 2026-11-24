from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from speech_mtl.common.models.audit import audit_model_parameters
from speech_mtl.common.models.factory import build_exp61_model

from .amp import AmpController
from .checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from .losses import compute_multitask_loss
from .optimization import (
    SchedulerConfig,
    ThreePhaseLRScheduler,
    build_optimizer,
)
from .phases import PhaseController
from .state import TrainingState


def validate_training_contract() -> dict[str, Any]:
    torch.manual_seed(1337)

    model = build_exp61_model()
    parameter_report = audit_model_parameters(
        model,
        enforce_exact=True,
    )

    phase_controller = PhaseController()
    phase_boundaries = {
        epoch: phase_controller.for_epoch(epoch).name
        for epoch in (1, 5, 6, 35, 36, 40)
    }

    expected_boundaries = {
        1: "phase1_warmup_joint",
        5: "phase1_warmup_joint",
        6: "phase2_full_joint",
        35: "phase2_full_joint",
        36: "phase3_heads_only",
        40: "phase3_heads_only",
    }

    if phase_boundaries != expected_boundaries:
        raise RuntimeError(
            f"Phase boundary mismatch: {phase_boundaries}"
        )

    optimizer = build_optimizer(model)
    scheduler = ThreePhaseLRScheduler(
        optimizer,
        optimizer_steps_per_epoch=3,
        config=SchedulerConfig(),
    )
    amp_controller = AmpController(
        device="cpu",
        enabled=False,
    )

    initial_lrs = scheduler.current_learning_rates()

    if not (
        initial_lrs["encoder"] == 1.0e-5
        and initial_lrs["tasks"] == 1.0e-5
    ):
        raise RuntimeError(
            f"Unexpected initial learning rates: {initial_lrs}"
        )

    languages = ["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"]
    frame_lengths = torch.tensor(
        [48, 47, 46, 45, 44, 43, 42, 41],
        dtype=torch.long,
    )
    maximum_frames = 48
    relative_lengths = (
        frame_lengths.to(torch.float32) / maximum_frames
    )
    valid_mask = (
        torch.arange(maximum_frames).unsqueeze(0)
        < frame_lengths.unsqueeze(1)
    )
    features = torch.randn(8, maximum_frames, 768)
    features = features.masked_fill(
        ~valid_mask.unsqueeze(-1),
        0.0,
    )

    language_targets = torch.tensor(
        [1, 0, 7, 5, 2, 4, 3, 6],
        dtype=torch.long,
    )
    dialect_targets = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3],
        dtype=torch.long,
    )
    gender_targets = torch.tensor(
        [0, 1, 0, 1, 0, 1, 0, 1],
        dtype=torch.long,
    )

    model.train()
    phase_controller.apply(model, 1)

    output = model(
        features=features,
        relative_lengths=relative_lengths,
        oracle_languages=languages,
        valid_mask=valid_mask,
    )

    loss_output = compute_multitask_loss(
        model=model,
        output=output,
        language_targets=language_targets,
        dialect_targets=dialect_targets,
        gender_targets=gender_targets,
    )

    optimizer.zero_grad(set_to_none=True)
    loss_output.total.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    scheduler.step()

    if not torch.isfinite(gradient_norm):
        raise RuntimeError("Gradient norm is non-finite")

    state = TrainingState(
        epoch=1,
        batch_in_epoch=1,
        global_batch=1,
        optimizer_step=1,
        phase_name="phase1_warmup_joint",
        best_dev_metric=0.5,
        best_epoch=1,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = (
            Path(temporary_directory) / "roundtrip.pt"
        )

        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_controller=amp_controller,
            state=state,
            extra={"validation": True},
        )

        restored_model = build_exp61_model()
        restored_optimizer = build_optimizer(
            restored_model
        )
        restored_scheduler = ThreePhaseLRScheduler(
            restored_optimizer,
            optimizer_steps_per_epoch=3,
            config=SchedulerConfig(),
        )
        restored_amp = AmpController(
            device="cpu",
            enabled=False,
        )

        restored_state, restored_extra = (
            load_training_checkpoint(
                checkpoint_path,
                model=restored_model,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                amp_controller=restored_amp,
                map_location="cpu",
                restore_rng=True,
            )
        )

        if restored_state.to_dict() != state.to_dict():
            raise RuntimeError(
                "TrainingState checkpoint roundtrip failed"
            )

        if restored_extra != {"validation": True}:
            raise RuntimeError(
                "Checkpoint extra metadata roundtrip failed"
            )

        for original, restored in zip(
            model.parameters(),
            restored_model.parameters(),
        ):
            if not torch.equal(original, restored):
                raise RuntimeError(
                    "Model checkpoint roundtrip failed"
                )

    model.train()
    phase3 = phase_controller.apply(model, 36)

    if phase3.encoder_trainable:
        raise RuntimeError(
            "Phase 3 encoder should be frozen"
        )

    if any(
        parameter.requires_grad
        for parameter in model.encoder.parameters()
    ):
        raise RuntimeError(
            "Phase 3 encoder freeze failed"
        )

    if not all(
        parameter.requires_grad
        for parameter in model.task_parameters()
    ):
        raise RuntimeError(
            "Phase 3 task parameters must remain trainable"
        )

    phase3_start_step = (
        scheduler.config.joint_cosine_end_epoch
        * scheduler.optimizer_steps_per_epoch
    )
    phase3_lrs = scheduler.learning_rates_for_step(
        phase3_start_step
    )

    if phase3_lrs != {
        "encoder": 0.0,
        "tasks": 2.0e-5,
    }:
        raise RuntimeError(
            f"Unexpected phase-3 learning rates: {phase3_lrs}"
        )

    return {
        "status": "PASS",
        "parameter_total": parameter_report.total,
        "phase_boundaries": {
            str(epoch): name
            for epoch, name in phase_boundaries.items()
        },
        "initial_learning_rates": initial_lrs,
        "learning_rates_after_one_step": (
            scheduler.current_learning_rates()
        ),
        "phase3_learning_rates": phase3_lrs,
        "losses": {
            task: float(loss.detach().cpu().item())
            for task, loss
            in loss_output.raw_losses.items()
        },
        "weighted_total_loss": float(
            loss_output.total.detach().cpu().item()
        ),
        "gradient_norm": float(
            gradient_norm.detach().cpu().item()
        ),
        "checkpoint_roundtrip": "PASS",
        "phase3_freeze": "PASS",
    }

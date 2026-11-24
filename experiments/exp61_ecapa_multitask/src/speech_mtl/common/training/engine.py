from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .losses import compute_multitask_loss
from .phases import PhaseController
from .state import TrainingState


@dataclass(frozen=True)
class EpochTrainingReport:
    epoch: int
    phase: str
    optimizer_steps: int
    examples: int
    mean_total_loss: float
    mean_language_loss: float
    mean_dialect_loss: float
    mean_gender_loss: float
    encoder_learning_rate: float
    task_learning_rate: float
    mean_gradient_norm: float
    amp_overflow_skips: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, Any]:
    tensor_keys = (
        "features",
        "relative_lengths",
        "valid_mask",
        "lid",
        "dialect",
        "gender",
    )

    moved = dict(batch)

    for key in tensor_keys:
        value = batch[key]
        moved[key] = value.to(
            device=device,
            non_blocking=non_blocking,
        )

    return moved


def train_one_epoch(
    *,
    model,
    dataloader,
    optimizer,
    scheduler,
    amp_controller,
    device: torch.device | str,
    epoch: int,
    training_state: TrainingState,
    phase_controller: PhaseController | None = None,
    gradient_accumulation_steps: int = 1,
    maximum_gradient_norm: float = 1.0,
    progress_writer=None,
    progress_interval_batches: int = 50,
    non_blocking: bool = True,
    show_progress_bar: bool = True,
) -> EpochTrainingReport:
    device = torch.device(device)
    phase_controller = phase_controller or PhaseController()

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive"
        )

    if maximum_gradient_norm <= 0:
        raise ValueError(
            "maximum_gradient_norm must be positive"
        )

    total_batches = len(dataloader)

    if total_batches <= 0:
        raise ValueError("Training dataloader is empty")

    model.train()
    phase = phase_controller.apply(model, epoch)

    training_state.epoch = int(epoch)
    training_state.batch_in_epoch = 0
    training_state.phase_name = phase.name

    optimizer.zero_grad(set_to_none=True)

    weighted_total_loss = 0.0
    weighted_task_losses = {
        "language": 0.0,
        "dialect": 0.0,
        "gender": 0.0,
    }
    examples_seen = 0
    optimizer_steps_this_epoch = 0
    gradient_norm_sum = 0.0
    gradient_norm_count = 0
    amp_overflow_skips = 0
    consecutive_amp_overflows = 0
    maximum_total_amp_overflows = 100
    maximum_consecutive_amp_overflows = 20
    started = time.monotonic()

    iterator = tqdm(
        dataloader,
        total=total_batches,
        desc=f"epoch {epoch:02d} {phase.name}",
        dynamic_ncols=True,
        disable=not show_progress_bar,
        leave=True,
    )

    for batch_index, raw_batch in enumerate(
        iterator,
        start=1,
    ):
        batch = _to_device(
            raw_batch,
            device,
            non_blocking=non_blocking,
        )
        batch_size = int(batch["features"].shape[0])

        group_start_zero_based = (
            ((batch_index - 1) // gradient_accumulation_steps)
            * gradient_accumulation_steps
        )
        group_size = min(
            gradient_accumulation_steps,
            total_batches - group_start_zero_based,
        )

        with amp_controller.autocast():
            output = model(
                features=batch["features"],
                relative_lengths=batch["relative_lengths"],
                oracle_languages=batch["language"],
                valid_mask=batch["valid_mask"],
            )

            loss_output = compute_multitask_loss(
                model=model,
                output=output,
                language_targets=batch["lid"],
                dialect_targets=batch["dialect"],
                gender_targets=batch["gender"],
            )

            scaled_loss = loss_output.total / group_size

        amp_controller.backward(scaled_loss)

        is_group_end = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == total_batches
        )

        if is_group_end:
            amp_controller.unscale_(optimizer)

            scale_before = float(
                amp_controller.scaler.get_scale()
            )
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=maximum_gradient_norm,
                error_if_nonfinite=False,
            )

            gradient_norm_value = float(
                gradient_norm.detach().cpu().item()
                if isinstance(gradient_norm, torch.Tensor)
                else gradient_norm
            )

            if math.isfinite(gradient_norm_value):
                amp_controller.step(optimizer)
                amp_controller.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                optimizer_steps_this_epoch += 1
                training_state.optimizer_step += 1
                gradient_norm_sum += gradient_norm_value
                gradient_norm_count += 1
                consecutive_amp_overflows = 0
            else:
                amp_controller.step(optimizer)
                amp_controller.update()
                optimizer.zero_grad(set_to_none=True)

                amp_overflow_skips += 1
                consecutive_amp_overflows += 1
                scale_after = float(
                    amp_controller.scaler.get_scale()
                )

                iterator.write(
                    "AMP_OVERFLOW_SKIPPED "
                    f"epoch={epoch} batch={batch_index}/{total_batches} "
                    f"scale={scale_before:.1f}->{scale_after:.1f} "
                    f"total_skips={amp_overflow_skips}"
                )

                if progress_writer is not None:
                    progress_writer.write(
                        {
                            "status": "AMP_OVERFLOW_SKIPPED",
                            "epoch": epoch,
                            "phase": phase.name,
                            "batch": batch_index,
                            "batches": total_batches,
                            "global_batch": (
                                training_state.global_batch
                            ),
                            "optimizer_step": (
                                training_state.optimizer_step
                            ),
                            "gradient_norm": gradient_norm_value,
                            "scale_before": scale_before,
                            "scale_after": scale_after,
                            "amp_overflow_skips": (
                                amp_overflow_skips
                            ),
                            "consecutive_amp_overflows": (
                                consecutive_amp_overflows
                            ),
                        }
                    )

                if (
                    amp_overflow_skips
                    > maximum_total_amp_overflows
                    or consecutive_amp_overflows
                    > maximum_consecutive_amp_overflows
                ):
                    raise FloatingPointError(
                        "Repeated AMP gradient overflows exceeded "
                        "the safety threshold: "
                        f"total={amp_overflow_skips}, "
                        f"consecutive={consecutive_amp_overflows}"
                    )

        loss_value = float(
            loss_output.total.detach().cpu().item()
        )

        if not math.isfinite(loss_value):
            raise FloatingPointError(
                "Non-finite training loss"
            )

        weighted_total_loss += loss_value * batch_size

        for task, task_loss in loss_output.raw_losses.items():
            weighted_task_losses[task] += (
                float(task_loss.detach().cpu().item())
                * batch_size
            )

        examples_seen += batch_size
        training_state.batch_in_epoch = batch_index
        training_state.global_batch += 1

        running_mean = weighted_total_loss / examples_seen
        current_lrs = scheduler.current_learning_rates()

        iterator.set_postfix(
            loss=f"{running_mean:.4f}",
            lr=f"{current_lrs['tasks']:.2e}",
            opt_step=training_state.optimizer_step,
            amp_skip=amp_overflow_skips,
        )

        should_write_progress = (
            progress_writer is not None
            and (
                batch_index % progress_interval_batches == 0
                or batch_index == total_batches
            )
        )

        if should_write_progress:
            progress_writer.write(
                {
                    "status": "TRAINING",
                    "epoch": epoch,
                    "phase": phase.name,
                    "batch": batch_index,
                    "batches": total_batches,
                    "global_batch": training_state.global_batch,
                    "optimizer_step": (
                        training_state.optimizer_step
                    ),
                    "amp_overflow_skips": amp_overflow_skips,
                    "examples_seen_in_epoch": examples_seen,
                    "mean_total_loss": running_mean,
                    "learning_rates": current_lrs,
                    "elapsed_seconds": (
                        time.monotonic() - started
                    ),
                }
            )

    elapsed_seconds = time.monotonic() - started
    current_lrs = scheduler.current_learning_rates()

    return EpochTrainingReport(
        epoch=epoch,
        phase=phase.name,
        optimizer_steps=optimizer_steps_this_epoch,
        examples=examples_seen,
        mean_total_loss=weighted_total_loss / examples_seen,
        mean_language_loss=(
            weighted_task_losses["language"] / examples_seen
        ),
        mean_dialect_loss=(
            weighted_task_losses["dialect"] / examples_seen
        ),
        mean_gender_loss=(
            weighted_task_losses["gender"] / examples_seen
        ),
        encoder_learning_rate=current_lrs["encoder"],
        task_learning_rate=current_lrs["tasks"],
        mean_gradient_norm=(
            gradient_norm_sum / gradient_norm_count
            if gradient_norm_count
            else 0.0
        ),
        amp_overflow_skips=amp_overflow_skips,
        elapsed_seconds=elapsed_seconds,
    )

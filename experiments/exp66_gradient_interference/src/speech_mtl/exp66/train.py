"""EXP 6.6 training: standard MTL, PCGrad, and GradNorm."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math
import time

import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, dump_config_snapshot
from .data import build_dataset, build_loader
from .gradient_methods import (
    DiagnosticAccumulator,
    GradNormController,
    assign_gradients,
    collect_parameter_gradients,
    pairwise_statistics,
    pcgrad_project,
)
from .labels import LabelSpace
from .metrics import PredictionAccumulator
from .model import CONDITIONS, TASKS, HardSharingMTLModel
from .objectives import (
    compute_raw_losses,
    detached_uncertainty_scales,
    uncertainty_total,
    weighted_total,
)
from .utils import atomic_json_dump, seed_everything, sha256_json


@dataclass
class TrainingPaths:
    root: Path
    checkpoints: Path
    history: Path
    progress: Path
    gradient_samples: Path
    gradient_summary: Path
    gradnorm_weights: Path

    @classmethod
    def make(cls, cfg: ExperimentConfig, domain: str, condition: str) -> "TrainingPaths":
        root = cfg.paths.run_root / domain / condition
        paths = cls(
            root=root,
            checkpoints=root / "checkpoints",
            history=root / "history.csv",
            progress=root / "progress.json",
            gradient_samples=root / "gradient_samples.csv",
            gradient_summary=root / "gradient_epoch_summary.csv",
            gradnorm_weights=root / "gradnorm_weights.csv",
        )
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
        return paths


class FrozenSchedule:
    def __init__(self, cfg: ExperimentConfig, steps_per_epoch: int):
        self.cfg = cfg
        self.steps_per_epoch = int(steps_per_epoch)

    def lr_for(self, epoch: int, successful_step: int) -> tuple[float, float]:
        t = self.cfg.training
        if epoch <= t.phase1_epochs:
            total = t.phase1_epochs * self.steps_per_epoch
            position = (epoch - 1) * self.steps_per_epoch + successful_step
            alpha = min(1.0, max(0.0, position / max(1, total - 1)))
            lr = t.warmup_start_lr + alpha * (t.peak_lr - t.warmup_start_lr)
            return lr, lr
        phase2_end = t.phase1_epochs + t.phase2_epochs
        if epoch <= phase2_end:
            total = t.phase2_epochs * self.steps_per_epoch
            position = (epoch - t.phase1_epochs - 1) * self.steps_per_epoch + successful_step
            alpha = min(1.0, max(0.0, position / max(1, total - 1)))
            lr = t.cosine_min_lr + 0.5 * (t.peak_lr - t.cosine_min_lr) * (
                1.0 + math.cos(math.pi * alpha)
            )
            return lr, lr
        return 0.0, t.phase3_task_lr

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int, successful_step: int) -> tuple[float, float]:
        encoder_lr, task_lr = self.lr_for(epoch, successful_step)
        optimizer.param_groups[0]["lr"] = encoder_lr
        optimizer.param_groups[1]["lr"] = task_lr
        return encoder_lr, task_lr


@contextmanager
def _singleton_batchnorm_safe(model: nn.Module, batch_size: int):
    if batch_size != 1:
        yield
        return
    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    states = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, state in zip(modules, states, strict=True):
            module.train(state)


def _to_device(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in (
        "features", "frame_lengths", "relative_lengths", "valid_mask",
        "lid", "dialect", "gender", "dataset_index",
    ):
        result[key] = result[key].to(device, non_blocking=True)
    return result


def _append_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _amp_components(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    return active, torch.amp.GradScaler("cuda", enabled=active)


def _evaluation_total(
    model: HardSharingMTLModel,
    raw_losses: dict[str, torch.Tensor],
    condition: str,
    gradnorm_controller: GradNormController | None,
) -> torch.Tensor:
    if condition == "gradnorm":
        if gradnorm_controller is None:
            raise RuntimeError("GradNorm controller is required")
        return weighted_total(raw_losses, gradnorm_controller.weights().detach())
    return uncertainty_total(model, raw_losses)


def evaluate_loader(
    model: HardSharingMTLModel,
    loader,
    device: torch.device,
    amp_enabled: bool,
    condition: str,
    gradnorm_controller: GradNormController | None,
    description: str,
) -> tuple[dict[str, Any], float]:
    model.eval()
    accumulator = PredictionAccumulator(model.labels)
    weighted_loss = 0.0
    examples = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False):
            batch = _to_device(batch, device)
            with _singleton_batchnorm_safe(model, len(batch["utt_id"])):
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(batch)
                    raw_losses = compute_raw_losses(model, batch, output)
                    total = _evaluation_total(
                        model, raw_losses, condition, gradnorm_controller
                    )
            batch_size = len(batch["utt_id"])
            weighted_loss += float(total.item()) * batch_size
            examples += batch_size
            accumulator.update(batch, output)
    metrics, _ = accumulator.finalize()
    return metrics, weighted_loss / max(1, examples)


def train_condition(
    cfg: ExperimentConfig,
    labels: LabelSpace,
    domain: str,
    condition: str,
    device: torch.device,
    *,
    resume: bool = True,
) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    seed_everything(cfg.training.seed)
    paths = TrainingPaths.make(cfg, domain, condition)
    complete_path = paths.root / "TRAINING_COMPLETE.json"
    best_path = paths.checkpoints / "best.pt"
    if resume and complete_path.is_file() and best_path.is_file():
        print(f"TRAINING_ALREADY_COMPLETE domain={domain} condition={condition}", flush=True)
        return best_path

    run_contract = {
        "experiment": "exp66",
        "domain": domain,
        "condition": condition,
        "active_tasks": list(TASKS),
        "gradient_diagnostic_scope": "shared_input_norm_and_full_ecapa_encoder",
        "gradient_diagnostic_interval": (
            1 if condition == "pcgrad" else cfg.training.gradient_diagnostic_interval
        ),
        "pcgrad_scope": "shared_encoder_parameters_only",
        "gradnorm_scope": "task_weights_from_shared_embedding_gradient_norms",
        "config": cfg.config_fingerprint_payload,
        "labels": labels.to_dict(),
    }
    contract_hash = sha256_json(run_contract)
    atomic_json_dump(
        run_contract | {"contract_sha256": contract_hash},
        paths.root / "RUN_CONTRACT.json",
    )
    dump_config_snapshot(cfg, paths.root / "CONFIG_SNAPSHOT.json")

    train_set = build_dataset(cfg, domain, "train", labels)
    dev_set = build_dataset(cfg, domain, "dev", labels)
    train_loader, train_sampler = build_loader(cfg, train_set, train=True)
    dev_loader, _ = build_loader(cfg, dev_set, train=False)

    model = HardSharingMTLModel(cfg.model, labels).to(device)
    model.configure_condition(condition)
    encoder_params = [parameter for parameter in model.encoder_parameters()]
    task_params = [parameter for parameter in model.task_parameters(condition)]
    if not encoder_params or not task_params:
        raise RuntimeError("Both optimizer groups must be non-empty")
    if {id(value) for value in encoder_params} & {id(value) for value in task_params}:
        raise RuntimeError("Optimizer groups overlap")
    trainable_ids = {id(value) for value in model.parameters() if value.requires_grad}
    if {id(value) for value in encoder_params + task_params} != trainable_ids:
        raise RuntimeError("Optimizer groups do not cover trainable parameters exactly")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.training.warmup_start_lr, "name": "encoder"},
            {"params": task_params, "lr": cfg.training.warmup_start_lr, "name": "tasks"},
        ],
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        eps=cfg.training.eps,
    )
    gradnorm_controller = None
    gradnorm_optimizer = None
    if condition == "gradnorm":
        gradnorm_controller = GradNormController(cfg.training.gradnorm_alpha).to(device)
        gradnorm_optimizer = torch.optim.Adam(
            gradnorm_controller.parameters(), lr=cfg.training.gradnorm_weight_lr
        )

    amp_enabled, scaler = _amp_components(device, cfg.training.amp)
    schedule = FrozenSchedule(cfg, len(train_loader))
    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0
    total_amp_skips = 0
    global_successful_steps = 0

    last_path = paths.checkpoints / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_checkpoint(last_path, device)
        if checkpoint["contract_sha256"] != contract_hash:
            raise RuntimeError("Checkpoint run contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        total_amp_skips = int(checkpoint.get("total_amp_skips", 0))
        global_successful_steps = int(checkpoint.get("global_successful_steps", 0))
        if condition == "gradnorm":
            assert gradnorm_controller is not None and gradnorm_optimizer is not None
            gradnorm_controller.load_state_dict(checkpoint["gradnorm_controller"])
            gradnorm_optimizer.load_state_dict(checkpoint["gradnorm_optimizer"])

    history_fields = [
        "epoch", "phase", "train_loss", "dev_loss", "selection_score",
        "lid_accuracy", "lid_macro_f1", "dialect_sample_weighted_accuracy",
        "dialect_language_macro_accuracy", "dialect_language_macro_f1",
        "gender_accuracy", "gender_macro_f1", "encoder_lr", "task_lr",
        "amp_skips_epoch", "amp_skips_total", "gradient_samples_epoch",
        "pcgrad_projections_epoch", "gradnorm_lid_weight", "gradnorm_dialect_weight",
        "gradnorm_gender_weight", "epoch_seconds", "examples_per_second",
        "best_epoch", "best_score",
    ]
    gradient_sample_fields = [
        "epoch", "batch", "phase", "task_a", "task_b", "cosine", "conflict",
        "norm_a", "norm_b", "cosine_post", "projection_count_step",
    ]
    gradient_summary_fields = [
        "epoch", "phase", "task_a", "task_b", "samples", "mean_cosine_pre",
        "conflict_rate_pre", "mean_cosine_post", "conflict_rate_post",
        "mean_norm_a", "mean_norm_b", "mean_projection_count_step",
    ]
    gradnorm_fields = [
        "epoch", "phase", "lid_weight", "dialect_weight", "gender_weight",
        "initial_lid_loss", "initial_dialect_loss", "initial_gender_loss",
    ]
    if start_epoch == 1:
        for path in (
            paths.history,
            paths.gradient_samples,
            paths.gradient_summary,
            paths.gradnorm_weights,
        ):
            if path.is_file():
                path.unlink()

    phase2_end = cfg.training.phase1_epochs + cfg.training.phase2_epochs
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        phase = (
            "phase1_warmup_joint" if epoch <= cfg.training.phase1_epochs
            else "phase2_full_joint" if epoch <= phase2_end
            else "phase3_heads_only"
        )
        encoder_trainable = epoch <= phase2_end
        model.set_encoder_trainable(encoder_trainable)
        model.train()
        if not encoder_trainable:
            model.input_norm.eval()
            model.encoder.eval()
        train_sampler.set_epoch(epoch)
        accumulator = PredictionAccumulator(labels)
        diagnostics = DiagnosticAccumulator.create()
        train_loss_sum = 0.0
        train_examples = 0
        skips_epoch = 0
        consecutive_skips = 0
        projections_epoch = 0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        if gradnorm_optimizer is not None:
            gradnorm_optimizer.zero_grad(set_to_none=True)
        last_encoder_lr = 0.0
        last_task_lr = 0.0
        successful_steps_epoch = 0
        progress = tqdm(train_loader, desc=f"{domain} {condition} e{epoch:02d}")

        for batch_index, batch in enumerate(progress, start=1):
            last_encoder_lr, last_task_lr = schedule.apply(
                optimizer, epoch, successful_steps_epoch
            )
            batch = _to_device(batch, device)
            with _singleton_batchnorm_safe(model, len(batch["utt_id"])):
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(batch)
                    raw_losses = compute_raw_losses(model, batch, output)

            for task, value in raw_losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite {task} loss at epoch={epoch} batch={batch_index}"
                    )

            pre_rows = None
            post_rows = None
            projection_count = 0
            combined_gradients = None
            gradnorm_weights = None
            should_diagnose = encoder_trainable and (
                condition == "pcgrad"
                or batch_index % cfg.training.gradient_diagnostic_interval == 0
            )

            if condition == "pcgrad" and encoder_trainable:
                task_gradients = collect_parameter_gradients(
                    raw_losses,
                    encoder_params,
                    scales=detached_uncertainty_scales(model),
                )
                pre_rows = pairwise_statistics(task_gradients)
                projected, combined_gradients, projection_count = pcgrad_project(
                    task_gradients,
                    seed=cfg.training.seed + global_successful_steps,
                )
                post_rows = pairwise_statistics(projected)
                total = uncertainty_total(model, raw_losses)
            elif condition == "gradnorm":
                assert gradnorm_controller is not None
                if encoder_trainable:
                    base_norms = gradnorm_controller.base_embedding_gradient_norms(
                        raw_losses, output["embedding"]
                    )
                    balance_loss, _, _ = gradnorm_controller.balancing_loss(
                        raw_losses, base_norms
                    )
                    balance_loss.backward()
                gradnorm_weights = gradnorm_controller.weights().detach()
                total = weighted_total(raw_losses, gradnorm_weights)
                if should_diagnose:
                    task_gradients = collect_parameter_gradients(raw_losses, encoder_params)
                    pre_rows = pairwise_statistics(task_gradients)
            else:
                total = uncertainty_total(model, raw_losses)
                if should_diagnose:
                    task_gradients = collect_parameter_gradients(
                        raw_losses,
                        encoder_params,
                        scales=detached_uncertainty_scales(model),
                    )
                    pre_rows = pairwise_statistics(task_gradients)

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite total loss at epoch={epoch} batch={batch_index}"
                )

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if combined_gradients is not None:
                assign_gradients(encoder_params, combined_gradients)
            grad_norm = nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=cfg.training.grad_clip_norm,
                error_if_nonfinite=False,
            )
            finite_gradients = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
            if not finite_gradients:
                if not amp_enabled:
                    raise RuntimeError(
                        f"Non-finite gradient norm without AMP epoch={epoch} batch={batch_index}"
                    )
                old_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if gradnorm_optimizer is not None:
                    gradnorm_optimizer.zero_grad(set_to_none=True)
                skips_epoch += 1
                total_amp_skips += 1
                consecutive_skips += 1
                print(
                    "AMP_OVERFLOW_SKIPPED "
                    f"condition={condition} epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"scale={old_scale}->{float(scaler.get_scale())} total={total_amp_skips}",
                    flush=True,
                )
                if skips_epoch > cfg.training.max_amp_skips_per_epoch:
                    raise RuntimeError("Exceeded maximum AMP skips in one epoch")
                if consecutive_skips > cfg.training.max_consecutive_amp_skips:
                    raise RuntimeError("Exceeded maximum consecutive AMP skips")
                continue

            scaler.step(optimizer)
            scaler.update()
            if gradnorm_optimizer is not None and encoder_trainable:
                gradnorm_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.zero_grad(set_to_none=True)

            successful_steps_epoch += 1
            global_successful_steps += 1
            consecutive_skips = 0
            projections_epoch += int(projection_count)
            if pre_rows is not None:
                diagnostics.add(
                    epoch=epoch,
                    batch=batch_index,
                    phase=phase,
                    pre_rows=pre_rows,
                    post_rows=post_rows,
                    projection_count=projection_count,
                )

            batch_size = len(batch["utt_id"])
            train_loss_sum += float(total.item()) * batch_size
            train_examples += batch_size
            accumulator.update(batch, output)
            postfix = {
                "loss": f"{train_loss_sum / max(1, train_examples):.4f}",
                "lr": f"{last_task_lr:.2e}",
                "amp_skip": skips_epoch,
            }
            if condition == "pcgrad":
                postfix["proj"] = projections_epoch
            if condition == "gradnorm" and gradnorm_controller is not None:
                weights_now = gradnorm_controller.normalized_weights_list()
                postfix["w"] = "/".join(f"{value:.2f}" for value in weights_now)
            progress.set_postfix(**postfix)

            if batch_index % cfg.training.progress_every_batches == 0:
                atomic_json_dump(
                    {
                        "status": "training",
                        "domain": domain,
                        "condition": condition,
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(train_loader),
                        "successful_steps_epoch": successful_steps_epoch,
                        "gradient_samples_epoch": len(diagnostics.samples),
                        "pcgrad_projections_epoch": projections_epoch,
                        "amp_skips_epoch": skips_epoch,
                        "amp_skips_total": total_amp_skips,
                        "timestamp": time.time(),
                    },
                    paths.progress,
                )

        train_metrics, _ = accumulator.finalize()
        train_loss = train_loss_sum / max(1, train_examples)
        dev_metrics, dev_loss = evaluate_loader(
            model,
            dev_loader,
            device,
            amp_enabled,
            condition,
            gradnorm_controller,
            f"dev {domain} {condition}",
        )
        score = float(dev_metrics["three_task_mean_macro_f1"])
        elapsed = time.time() - start_time
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch

        controller_state = None
        controller_optimizer_state = None
        current_gradnorm_weights = ["", "", ""]
        if gradnorm_controller is not None:
            controller_state = gradnorm_controller.state_dict()
            assert gradnorm_optimizer is not None
            controller_optimizer_state = gradnorm_optimizer.state_dict()
            current_gradnorm_weights = gradnorm_controller.normalized_weights_list()

        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "total_amp_skips": total_amp_skips,
            "global_successful_steps": global_successful_steps,
            "contract_sha256": contract_hash,
            "domain": domain,
            "condition": condition,
            "dev_metrics": dev_metrics,
            "parameter_counts": model.parameter_counts,
            "gradnorm_controller": controller_state,
            "gradnorm_optimizer": controller_optimizer_state,
            "gradient_diagnostic_interval": (
                1 if condition == "pcgrad" else cfg.training.gradient_diagnostic_interval
            ),
        }
        save_checkpoint(last_path, checkpoint_payload)
        if is_best:
            save_checkpoint(best_path, checkpoint_payload)
            atomic_json_dump(
                dev_metrics | {"selection_score": score, "epoch": epoch},
                paths.root / "best_dev_metrics.json",
            )

        _append_rows(paths.gradient_samples, diagnostics.samples, gradient_sample_fields)
        epoch_summary = diagnostics.epoch_summary(epoch, phase)
        _append_rows(paths.gradient_summary, epoch_summary, gradient_summary_fields)
        if gradnorm_controller is not None:
            initial = gradnorm_controller.initial_losses.detach().cpu().tolist()
            _append_rows(
                paths.gradnorm_weights,
                [
                    {
                        "epoch": epoch,
                        "phase": phase,
                        "lid_weight": current_gradnorm_weights[0],
                        "dialect_weight": current_gradnorm_weights[1],
                        "gender_weight": current_gradnorm_weights[2],
                        "initial_lid_loss": initial[0],
                        "initial_dialect_loss": initial[1],
                        "initial_gender_loss": initial[2],
                    }
                ],
                gradnorm_fields,
            )

        history_row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "selection_score": score,
            "lid_accuracy": dev_metrics["lid_accuracy"],
            "lid_macro_f1": dev_metrics["lid_macro_f1"],
            "dialect_sample_weighted_accuracy": dev_metrics["dialect_sample_weighted_accuracy"],
            "dialect_language_macro_accuracy": dev_metrics["dialect_language_macro_accuracy"],
            "dialect_language_macro_f1": dev_metrics["dialect_language_macro_f1"],
            "gender_accuracy": dev_metrics["gender_accuracy"],
            "gender_macro_f1": dev_metrics["gender_macro_f1"],
            "encoder_lr": last_encoder_lr,
            "task_lr": last_task_lr,
            "amp_skips_epoch": skips_epoch,
            "amp_skips_total": total_amp_skips,
            "gradient_samples_epoch": len(diagnostics.samples),
            "pcgrad_projections_epoch": projections_epoch,
            "gradnorm_lid_weight": current_gradnorm_weights[0],
            "gradnorm_dialect_weight": current_gradnorm_weights[1],
            "gradnorm_gender_weight": current_gradnorm_weights[2],
            "epoch_seconds": elapsed,
            "examples_per_second": train_examples / max(elapsed, 1e-9),
            "best_epoch": best_epoch,
            "best_score": best_score,
        }
        _append_rows(paths.history, [history_row], history_fields)
        atomic_json_dump(
            {
                "status": "epoch_complete",
                "domain": domain,
                "condition": condition,
                "phase": phase,
                "epoch": epoch,
                "epochs": cfg.training.epochs,
                "selection_score": score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "gradient_samples_epoch": len(diagnostics.samples),
                "pcgrad_projections_epoch": projections_epoch,
                "amp_skips_total": total_amp_skips,
                "timestamp": time.time(),
            },
            paths.progress,
        )

    atomic_json_dump(
        {
            "status": "complete",
            "domain": domain,
            "condition": condition,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "checkpoint": str(best_path),
            "gradient_samples": str(paths.gradient_samples),
            "gradient_epoch_summary": str(paths.gradient_summary),
            "timestamp": time.time(),
        },
        complete_path,
    )
    print(f"EXP66_TRAINING_PASS domain={domain} condition={condition}", flush=True)
    return best_path

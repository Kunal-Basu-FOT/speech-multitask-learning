"""Formal training loop for one condition and one feature domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import csv
import json
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
from .labels import LabelSpace
from .losses import compute_losses
from .metrics import PredictionAccumulator
from .model import HierarchicalLIDDialectModel, CONDITION_NAME
from .utils import atomic_json_dump, seed_everything, sha256_json


@dataclass
class TrainingPaths:
    root: Path
    checkpoints: Path
    history: Path
    progress: Path

    @classmethod
    def make(cls, cfg: ExperimentConfig, domain: str, condition: str) -> "TrainingPaths":
        root = cfg.paths.run_root / domain / condition
        paths = cls(
            root=root,
            checkpoints=root / "checkpoints",
            history=root / "history.csv",
            progress=root / "progress.json",
        )
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
        return paths


class FrozenSchedule:
    def __init__(self, cfg: ExperimentConfig, steps_per_epoch: int):
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.optimizer_steps = 0

    def lr_for(self, epoch: int, step_in_epoch: int) -> tuple[float, float]:
        t = self.cfg.training
        if epoch <= t.phase1_epochs:
            total = t.phase1_epochs * self.steps_per_epoch
            position = (epoch - 1) * self.steps_per_epoch + step_in_epoch
            alpha = min(1.0, max(0.0, position / max(1, total - 1)))
            lr = t.warmup_start_lr + alpha * (t.peak_lr - t.warmup_start_lr)
            return lr, lr
        phase2_end = t.phase1_epochs + t.phase2_epochs
        if epoch <= phase2_end:
            total = t.phase2_epochs * self.steps_per_epoch
            position = (epoch - t.phase1_epochs - 1) * self.steps_per_epoch + step_in_epoch
            alpha = min(1.0, max(0.0, position / max(1, total - 1)))
            lr = t.cosine_min_lr + 0.5 * (t.peak_lr - t.cosine_min_lr) * (
                1.0 + math.cos(math.pi * alpha)
            )
            return lr, lr
        return 0.0, t.phase3_task_lr

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int, step_in_epoch: int) -> tuple[float, float]:
        encoder_lr, task_lr = self.lr_for(epoch, step_in_epoch)
        optimizer.param_groups[0]["lr"] = encoder_lr
        optimizer.param_groups[1]["lr"] = task_lr
        return encoder_lr, task_lr




@contextmanager
def _singleton_batchnorm_safe(model: nn.Module, batch_size: int):
    """Use running BN statistics for unavoidable singleton training batches.

    Full utterances longer than the frame budget are intentionally retained as
    singleton batches. ECAPA's post-pooling BatchNorm cannot estimate batch
    statistics from B=1, so only BatchNorm modules are temporarily switched to
    evaluation mode for that forward pass. Dropout and all non-BN modules keep
    their normal training state, and affine BN parameters remain differentiable.
    """
    if batch_size != 1:
        yield
        return
    modules = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    states = [m.training for m in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, state in zip(modules, states, strict=True):
            module.train(state)


def _to_device(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in ("features", "frame_lengths", "relative_lengths", "valid_mask", "lid", "dialect", "gender", "dataset_index"):
        result[key] = result[key].to(device, non_blocking=True)
    return result


def _amp_components(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=active)
    return active, scaler


def _append_history(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_loader(
    model: HierarchicalLIDDialectModel,
    loader,
    device: torch.device,
    amp_enabled: bool,
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
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(batch)
                    losses = compute_losses(model, batch, output)
            batch_size = len(batch["utt_id"])
            weighted_loss += float(losses["total"].item()) * batch_size
            examples += batch_size
            accumulator.update(batch, output)
    metrics, _ = accumulator.finalize()
    return metrics, weighted_loss / max(1, examples)


def train_condition(
    cfg: ExperimentConfig,
    labels: LabelSpace,
    domain: str,
    device: torch.device,
    *,
    resume: bool = True,
) -> Path:
    seed_everything(cfg.training.seed)
    paths = TrainingPaths.make(cfg, domain, CONDITION_NAME)
    complete_path = paths.root / "TRAINING_COMPLETE.json"
    best_path = paths.checkpoints / "best.pt"
    if resume and complete_path.is_file() and best_path.is_file():
        print(f"TRAINING_ALREADY_COMPLETE domain={domain} condition={CONDITION_NAME}", flush=True)
        return best_path
    run_contract = {
        "domain": domain,
        "condition": CONDITION_NAME,
        "active_tasks": ("lid", "dialect", "gender"),
        "config": cfg.config_fingerprint_payload,
        "labels": labels.to_dict(),
    }
    contract_hash = sha256_json(run_contract)
    atomic_json_dump(run_contract | {"contract_sha256": contract_hash}, paths.root / "RUN_CONTRACT.json")
    dump_config_snapshot(cfg, paths.root / "CONFIG_SNAPSHOT.json")

    train_set = build_dataset(cfg, domain, "train", labels)
    dev_set = build_dataset(cfg, domain, "dev", labels)
    train_loader, train_sampler = build_loader(cfg, train_set, train=True)
    dev_loader, _ = build_loader(cfg, dev_set, train=False)

    model = HierarchicalLIDDialectModel(cfg.model, labels).to(device)
    encoder_params = list(model.encoder_parameters())
    task_params = list(model.task_parameters())
    if not encoder_params or not task_params:
        raise RuntimeError("Both encoder and task optimizer groups must be non-empty")
    if {id(x) for x in encoder_params} & {id(x) for x in task_params}:
        raise RuntimeError("Optimizer groups overlap")
    if {id(x) for x in encoder_params + task_params} != {id(x) for x in model.parameters()}:
        raise RuntimeError("Optimizer groups do not cover every model parameter exactly once")
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.training.warmup_start_lr, "name": "encoder"},
            {"params": task_params, "lr": cfg.training.warmup_start_lr, "name": "tasks"},
        ],
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        eps=cfg.training.eps,
    )
    amp_enabled, scaler = _amp_components(device, cfg.training.amp)
    schedule = FrozenSchedule(cfg, len(train_loader))
    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0
    total_amp_skips = 0

    last_path = paths.checkpoints / "last.pt"
    if resume and last_path.is_file():
        checkpoint = load_checkpoint(last_path, device)
        if checkpoint["contract_sha256"] != contract_hash:
            raise RuntimeError("Checkpoint run contract does not match current configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        total_amp_skips = int(checkpoint.get("total_amp_skips", 0))

    history_fields = [
        "epoch", "phase", "train_loss", "dev_loss", "selection_score",
        "lid_accuracy", "lid_macro_f1", "dialect_sample_weighted_accuracy",
        "dialect_language_macro_accuracy", "dialect_language_macro_f1", "gender_accuracy", "gender_macro_f1",
        "encoder_lr", "task_lr", "amp_skips_epoch", "amp_skips_total",
        "epoch_seconds", "examples_per_second", "best_epoch", "best_score",
    ]
    if paths.history.is_file() and start_epoch == 1:
        paths.history.unlink()

    phase2_end = cfg.training.phase1_epochs + cfg.training.phase2_epochs
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        phase = (
            "phase1_warmup_joint" if epoch <= cfg.training.phase1_epochs
            else "phase2_full_joint" if epoch <= phase2_end
            else "phase3_heads_only"
        )
        model.set_encoder_trainable(epoch <= phase2_end)
        model.train()
        if epoch > phase2_end:
            model.input_norm.eval()
            model.encoder.eval()
        train_sampler.set_epoch(epoch)
        accumulator = PredictionAccumulator(labels)
        train_loss_sum = 0.0
        train_examples = 0
        skips_epoch = 0
        consecutive_skips = 0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        last_encoder_lr = 0.0
        last_task_lr = 0.0
        progress = tqdm(train_loader, desc=f"{domain} {CONDITION_NAME} e{epoch:02d}")
        successful_steps_in_epoch = 0
        for batch_index, batch in enumerate(progress, start=1):
            # LR progression is tied to successful optimizer updates, not raw batches.
            # A recoverable AMP overflow therefore does not advance the schedule.
            last_encoder_lr, last_task_lr = schedule.apply(
                optimizer, epoch, successful_steps_in_epoch
            )
            batch = _to_device(batch, device)
            with _singleton_batchnorm_safe(model, len(batch["utt_id"])):
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(batch)
                    losses = compute_losses(model, batch, output)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}: {losses['total'].item()}"
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.training.grad_clip_norm,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                if not amp_enabled:
                    raise RuntimeError(
                        f"Non-finite gradient norm without AMP at epoch={epoch}, batch={batch_index}"
                    )
                old_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                new_scale = float(scaler.get_scale())
                optimizer.zero_grad(set_to_none=True)
                skips_epoch += 1
                total_amp_skips += 1
                consecutive_skips += 1
                print(
                    "AMP_OVERFLOW_SKIPPED "
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"scale={old_scale}->{new_scale} total_skips={total_amp_skips}",
                    flush=True,
                )
                if skips_epoch > cfg.training.max_amp_skips_per_epoch:
                    raise RuntimeError("Exceeded maximum AMP skips in one epoch")
                if consecutive_skips > cfg.training.max_consecutive_amp_skips:
                    raise RuntimeError("Exceeded maximum consecutive AMP skips")
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            successful_steps_in_epoch += 1
            consecutive_skips = 0
            batch_size = len(batch["utt_id"])
            train_loss_sum += float(losses["total"].item()) * batch_size
            train_examples += batch_size
            accumulator.update(batch, output)
            progress.set_postfix(
                loss=f"{train_loss_sum/max(1, train_examples):.4f}",
                lr=f"{last_task_lr:.2e}",
                amp_skip=skips_epoch,
            )
            if batch_index % cfg.training.progress_every_batches == 0:
                atomic_json_dump(
                    {
                        "status": "training",
                        "domain": domain,
                        "condition": CONDITION_NAME,
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(train_loader),
                        "amp_skips_epoch": skips_epoch,
                        "amp_skips_total": total_amp_skips,
                        "timestamp": time.time(),
                    },
                    paths.progress,
                )

        train_metrics, _ = accumulator.finalize()
        train_loss = train_loss_sum / max(1, train_examples)
        dev_metrics, dev_loss = evaluate_loader(
            model, dev_loader, device, amp_enabled, f"dev {domain} {CONDITION_NAME}"
        )
        score = float(dev_metrics["three_task_mean_macro_f1"])
        elapsed = time.time() - start_time
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
        checkpoint_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "total_amp_skips": total_amp_skips,
            "contract_sha256": contract_hash,
            "domain": domain,
            "condition": CONDITION_NAME,
            "dev_metrics": dev_metrics,
            "parameter_counts": model.parameter_counts,
        }
        save_checkpoint(last_path, checkpoint_payload)
        if is_best:
            save_checkpoint(paths.checkpoints / "best.pt", checkpoint_payload)
            atomic_json_dump(dev_metrics | {"selection_score": score, "epoch": epoch}, paths.root / "best_dev_metrics.json")
        history_row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "selection_score": score,
            "lid_accuracy": dev_metrics.get("lid_accuracy", ""),
            "lid_macro_f1": dev_metrics.get("lid_macro_f1", ""),
            "dialect_sample_weighted_accuracy": dev_metrics.get("dialect_sample_weighted_accuracy", ""),
            "dialect_language_macro_accuracy": dev_metrics.get("dialect_language_macro_accuracy", ""),
            "dialect_language_macro_f1": dev_metrics.get("dialect_language_macro_f1", ""),
            "gender_accuracy": dev_metrics.get("gender_accuracy", ""),
            "gender_macro_f1": dev_metrics.get("gender_macro_f1", ""),
            "encoder_lr": last_encoder_lr,
            "task_lr": last_task_lr,
            "amp_skips_epoch": skips_epoch,
            "amp_skips_total": total_amp_skips,
            "epoch_seconds": elapsed,
            "examples_per_second": train_examples / max(elapsed, 1e-9),
            "best_epoch": best_epoch,
            "best_score": best_score,
        }
        _append_history(paths.history, history_row, history_fields)
        atomic_json_dump(
            {
                "status": "epoch_complete",
                "domain": domain,
                "condition": CONDITION_NAME,
                "phase": phase,
                "epoch": epoch,
                "epochs": cfg.training.epochs,
                "selection_score": score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "amp_skips_total": total_amp_skips,
                "timestamp": time.time(),
            },
            paths.progress,
        )

    complete = paths.root / "TRAINING_COMPLETE.json"
    atomic_json_dump(
        {
            "status": "complete",
            "domain": domain,
            "condition": CONDITION_NAME,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "checkpoint": str(paths.checkpoints / "best.pt"),
            "timestamp": time.time(),
        },
        complete,
    )
    return paths.checkpoints / "best.pt"

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .config import Config
from .data import build_dataloader, move_batch
from .evaluate import evaluate_loader
from .metrics import PredictionAccumulator
from .model import LogMelProjectedMultitaskECAPA, parameter_report
from .schedule import Exp69LRScheduler, phase_for_epoch
from .utils import atomic_json, seed_everything, sha256_file, utc_now


HISTORY_COLUMNS = [
    "epoch",
    "phase",
    "train_loss",
    "dev_loss",
    "selection_score",
    "lid_accuracy",
    "lid_macro_f1",
    "dialect_sample_weighted_accuracy",
    "dialect_language_macro_accuracy",
    "dialect_language_macro_f1",
    "gender_accuracy",
    "gender_macro_f1",
    "encoder_lr",
    "task_lr",
    "amp_skips_epoch",
    "amp_skips_total",
    "optimizer_steps",
    "epoch_seconds",
    "examples_per_second",
    "best_epoch",
    "best_score",
]


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _contract(config: Config, domain: str, duration_name: str, steps_per_epoch: int) -> dict[str, Any]:
    cache_index = config.path("logmel_cache_root") / "train" / "index.csv"
    return {
        "experiment": "EXP6.9",
        "domain": domain,
        "duration": duration_name,
        "seed": int(config.seed),
        "epochs": int(config.epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "feature_dimension": int(config.feature_dimension),
        "projected_dimension": int(config.projected_dimension),
        "embedding_dimension": int(config.embedding_dimension),
        "max_batch_frames": int(config.max_batch_frames),
        "max_batch_size": int(config.max_batch_size),
        "number_of_buckets": int(config.number_of_buckets),
        "train_index_sha256": sha256_file(cache_index),
        "model": "80-bin log-Mel -> Linear(80,768) -> EXP6.1 compact ECAPA-TDNN",
        "one_bit_quantizer": "+1 if x>0 else -1",
    }


def _write_history_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _history_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _train_epoch(
    *,
    config: Config,
    model: LogMelProjectedMultitaskECAPA,
    loader,
    sampler,
    dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: Exp69LRScheduler,
    scaler,
    device: torch.device,
    epoch: int,
    optimizer_step: int,
    amp_skips_total: int,
) -> tuple[dict[str, Any], int, int, float, float]:
    from speech_mtl.common.training.losses import compute_multitask_loss
    from speech_mtl.exp61.metrics import EpochMetricAccumulator

    phase = phase_for_epoch(epoch, config)
    model.train()
    model.set_encoder_trainable(phase.encoder_trainable)
    sampler.set_epoch(epoch)
    dataset.set_epoch(epoch)

    accumulator = PredictionAccumulator(dict(config.dialect_class_counts))
    dialect_helper = EpochMetricAccumulator()
    loss_sum = 0.0
    examples = 0
    amp_skips_epoch = 0
    started = time.monotonic()
    last_encoder_lr = 0.0
    last_task_lr = 0.0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"train {epoch:02d}/{int(config.epochs)}",
        dynamic_ncols=True,
        leave=True,
    )
    for raw_batch in iterator:
        batch = move_batch(raw_batch, device)
        batch_size = int(batch["features"].shape[0])
        last_encoder_lr, last_task_lr = scheduler.apply(optimizer_step, epoch)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(config.use_amp and device.type == "cuda"),
        ):
            output = model(
                features=batch["features"],
                relative_lengths=batch["relative_lengths"],
                oracle_languages=batch["language"],
                valid_mask=batch["valid_mask"],
            )
            loss_output = compute_multitask_loss(
                model=model.base,
                output=output,
                language_targets=batch["lid"],
                dialect_targets=batch["dialect"],
                gender_targets=batch["gender"],
            )
            loss = loss_output.total

        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config.gradient_clip_norm),
            error_if_nonfinite=False,
        )
        finite_grad = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())

        if finite_grad:
            optimizer_step += 1
        else:
            # GradScaler has already skipped optimizer.step and reduced its scale.
            amp_skips_epoch += 1
            amp_skips_total += 1
            print(
                "AMP_OVERFLOW_SKIPPED "
                f"epoch={epoch} scale={old_scale}->{new_scale} "
                f"total_skips={amp_skips_total}",
                flush=True,
            )

        dialect_predictions = dialect_helper.update(output=output, batch=batch)
        accumulator.update(
            output=output,
            batch=batch,
            dialect_predictions=dialect_predictions,
        )
        loss_sum += float(loss.detach().cpu().item()) * batch_size
        examples += batch_size
        iterator.set_postfix(
            loss=f"{loss_sum / examples:.4f}",
            lr=f"{last_task_lr:.2e}",
            amp_skip=amp_skips_epoch,
        )

    elapsed = time.monotonic() - started
    metrics = accumulator.finalize()
    metrics.update(
        {
            "mean_total_loss": loss_sum / examples,
            "elapsed_seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "amp_skips_epoch": amp_skips_epoch,
        }
    )
    return metrics, optimizer_step, amp_skips_total, last_encoder_lr, last_task_lr


def train_condition(
    config: Config,
    *,
    domain: str,
    duration_name: str,
    device: torch.device,
) -> dict[str, Any]:
    run_root = config.path("run_root") / domain / duration_name
    checkpoints = run_root / "checkpoints"
    history_path = run_root / "history.csv"
    completion_path = run_root / "TRAINING_COMPLETE.json"
    last_path = checkpoints / "last.pt"
    best_path = checkpoints / "best.pt"

    if completion_path.is_file():
        marker = json.loads(completion_path.read_text(encoding="utf-8"))
        print(f"TRAINING_SKIP_COMPLETE domain={domain} duration={duration_name}", flush=True)
        return marker

    seed_everything(int(config.seed))
    train_loader, train_sampler, train_dataset = build_dataloader(
        config,
        split="train",
        domain=domain,
        duration_name=duration_name,
        training=True,
    )
    dev_loader, dev_sampler, dev_dataset = build_dataloader(
        config,
        split="dev",
        domain=domain,
        duration_name=duration_name,
        training=False,
    )
    steps_per_epoch = len(train_loader)
    contract = _contract(config, domain, duration_name, steps_per_epoch)
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json(run_root / "RUN_CONTRACT.json", contract)
    atomic_json(run_root / "CONFIG_SNAPSHOT.json", config.raw)

    model = LogMelProjectedMultitaskECAPA(
        int(config.feature_dimension), int(config.projected_dimension)
    ).to(device)
    report = parameter_report(model)
    atomic_json(run_root / "MODEL_PARAMETER_REPORT.json", report)

    encoder_parameters = model.encoder_parameters()
    task_parameters = model.task_parameters()
    encoder_ids = {id(value) for value in encoder_parameters}
    task_ids = {id(value) for value in task_parameters}
    if encoder_ids & task_ids:
        raise RuntimeError("Encoder and task optimizer groups overlap")
    if encoder_ids | task_ids != {id(value) for value in model.parameters()}:
        raise RuntimeError("Optimizer groups do not cover all parameters")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": float(config.warmup_start_lr), "name": "encoder"},
            {"params": task_parameters, "lr": float(config.warmup_start_lr), "name": "tasks"},
        ],
        betas=(float(config.beta1), float(config.beta2)),
        eps=float(config.epsilon),
        weight_decay=float(config.weight_decay),
    )
    scaler = _make_scaler(bool(config.use_amp and device.type == "cuda"))
    scheduler = Exp69LRScheduler(optimizer, steps_per_epoch, config)

    state = {
        "completed_epoch": 0,
        "optimizer_step": 0,
        "best_score": float("-inf"),
        "best_epoch": 0,
        "amp_skips_total": 0,
    }
    existing_history = _history_rows(history_path)
    if last_path.is_file():
        state = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_contract=contract,
        )
        checkpoint_epoch = int(state["completed_epoch"])
        if len(existing_history) == checkpoint_epoch + 1:
            # A crash may occur after the history row is atomically appended but
            # before last.pt is atomically replaced. Drop only that uncommitted row
            # and restart the epoch from the last complete checkpoint.
            retained = existing_history[:checkpoint_epoch]
            temporary = history_path.with_name(f".{history_path.name}.repair")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
                writer.writeheader()
                writer.writerows(retained)
            temporary.replace(history_path)
            existing_history = retained
            print(
                f"HISTORY_AHEAD_REPAIRED retained_epochs={checkpoint_epoch}",
                flush=True,
            )
        if len(existing_history) != checkpoint_epoch:
            raise RuntimeError(
                "History/checkpoint epoch mismatch: "
                f"history={len(existing_history)} checkpoint={checkpoint_epoch}"
            )
        print(
            f"RESUME domain={domain} duration={duration_name} "
            f"from_epoch={int(state['completed_epoch']) + 1}",
            flush=True,
        )

        # Repair the narrow crash window after last.pt is committed but before
        # a newly improved best.pt is atomically replaced.
        if int(state.get("best_epoch", 0)) == checkpoint_epoch and checkpoint_epoch > 0:
            best_saved_epoch = -1
            if best_path.is_file():
                try:
                    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
                    best_saved_epoch = int(
                        best_payload.get("training_state", {}).get("completed_epoch", -1)
                    )
                except Exception:
                    best_saved_epoch = -1
            if best_saved_epoch != checkpoint_epoch:
                shutil.copy2(last_path, best_path)
                if "best_dev_metrics" in state:
                    atomic_json(run_root / "best_dev_metrics.json", state["best_dev_metrics"])
                print(
                    f"BEST_CHECKPOINT_REPAIRED epoch={checkpoint_epoch}",
                    flush=True,
                )
    elif existing_history:
        raise RuntimeError("History exists without last.pt; refusing ambiguous resume")

    for epoch in range(int(state["completed_epoch"]) + 1, int(config.epochs) + 1):
        epoch_started = time.monotonic()
        train_metrics, optimizer_step, amp_skips_total, encoder_lr, task_lr = _train_epoch(
            config=config,
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            dataset=train_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            optimizer_step=int(state["optimizer_step"]),
            amp_skips_total=int(state["amp_skips_total"]),
        )
        dev_sampler.set_epoch(0)
        dev_dataset.set_epoch(0)
        dev_metrics = evaluate_loader(
            config=config,
            model=model,
            loader=dev_loader,
            device=device,
            description=f"dev {domain}/{duration_name} e{epoch}",
            predictions_path=None,
        )
        score = float(dev_metrics["three_task_mean_macro_f1"])
        improved = score > float(state["best_score"])
        if improved:
            state["best_score"] = score
            state["best_epoch"] = epoch
            state["best_dev_metrics"] = dev_metrics

        state.update(
            {
                "completed_epoch": epoch,
                "optimizer_step": optimizer_step,
                "amp_skips_total": amp_skips_total,
                "phase": phase_for_epoch(epoch, config).name,
            }
        )
        epoch_seconds = time.monotonic() - epoch_started
        row = {
            "epoch": epoch,
            "phase": phase_for_epoch(epoch, config).name,
            "train_loss": train_metrics["mean_total_loss"],
            "dev_loss": dev_metrics["mean_total_loss"],
            "selection_score": score,
            "lid_accuracy": dev_metrics["language"]["accuracy"],
            "lid_macro_f1": dev_metrics["language"]["macro_f1"],
            "dialect_sample_weighted_accuracy": dev_metrics["dialect"]["sample_weighted_accuracy"],
            "dialect_language_macro_accuracy": dev_metrics["dialect"]["language_macro_accuracy"],
            "dialect_language_macro_f1": dev_metrics["dialect"]["language_macro_f1"],
            "gender_accuracy": dev_metrics["gender"]["accuracy"],
            "gender_macro_f1": dev_metrics["gender"]["macro_f1"],
            "encoder_lr": encoder_lr,
            "task_lr": task_lr,
            "amp_skips_epoch": train_metrics["amp_skips_epoch"],
            "amp_skips_total": amp_skips_total,
            "optimizer_steps": optimizer_step,
            "epoch_seconds": epoch_seconds,
            "examples_per_second": train_metrics["examples_per_second"],
            "best_epoch": state["best_epoch"],
            "best_score": state["best_score"],
        }
        # Commit order is intentional: history first, then atomic last.pt.
        # If interrupted between them, resume removes the single uncommitted
        # history row and restarts that epoch from the previous checkpoint.
        _write_history_row(history_path, row)
        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_state=state,
            contract=contract,
        )
        if improved:
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                training_state=state,
                contract=contract,
            )
            atomic_json(run_root / "best_dev_metrics.json", dev_metrics)
        atomic_json(
            run_root / "progress.json",
            {
                "status": "running",
                "domain": domain,
                "duration": duration_name,
                "epoch": epoch,
                "epochs": int(config.epochs),
                "best_epoch": state["best_epoch"],
                "best_score": state["best_score"],
                "optimizer_step": optimizer_step,
                "amp_skips_total": amp_skips_total,
                "updated_utc": utc_now(),
            },
        )
        print(
            f"EPOCH_COMPLETE domain={domain} duration={duration_name} "
            f"epoch={epoch}/{int(config.epochs)} score={score:.6f} "
            f"best={float(state['best_score']):.6f}",
            flush=True,
        )

    if len(_history_rows(history_path)) != int(config.epochs):
        raise RuntimeError("Final history does not contain exactly 40 epochs")
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Final checkpoints are missing")

    marker = {
        "status": "TRAINING_COMPLETE",
        "created_utc": utc_now(),
        "domain": domain,
        "duration": duration_name,
        "epochs": int(config.epochs),
        "train_examples": int(config.expected_rows["train"]),
        "dev_examples": int(config.expected_rows["dev"]),
        "best_epoch": int(state["best_epoch"]),
        "best_dev_metric": float(state["best_score"]),
        "optimizer_steps": int(state["optimizer_step"]),
        "amp_skips_total": int(state["amp_skips_total"]),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": str(history_path),
        "model_parameters": report,
    }
    atomic_json(completion_path, marker)
    print(f"EXP69_TRAINING_PASS domain={domain} duration={duration_name}", flush=True)
    return marker

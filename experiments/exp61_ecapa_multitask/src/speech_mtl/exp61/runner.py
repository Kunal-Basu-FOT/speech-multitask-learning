from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from speech_mtl.common.analysis.confusion_plots import (
    save_composite_confusion_figure,
)
from speech_mtl.common.analysis.cross_precision import (
    save_cross_precision_summary,
)
from speech_mtl.common.analysis.curves import (
    save_training_curves,
)
from speech_mtl.common.analysis.report_tables import (
    save_cross_precision_table,
    save_experimental_configuration_table,
    save_matched_results_table,
    save_per_language_dialect_table,
    save_runtime_table,
)
from speech_mtl.common.models.audit import (
    audit_model_parameters,
)
from speech_mtl.common.models.factory import build_exp61_model
from speech_mtl.common.training.amp import AmpController
from speech_mtl.common.training.checkpointing import (
    CheckpointManager,
    load_training_checkpoint,
)
from speech_mtl.common.training.optimization import (
    OptimizerConfig,
    SchedulerConfig,
    ThreePhaseLRScheduler,
    build_optimizer,
)
from speech_mtl.common.training.progress import ProgressWriter
from speech_mtl.common.training.state import TrainingState
from speech_mtl.common.utils.atomic_io import (
    atomic_write_json,
    sha256_file,
)
from speech_mtl.common.utils.logging_utils import (
    configure_logging,
)
from speech_mtl.common.utils.reproducibility import (
    seed_everything,
)
from speech_mtl.common.utils.runtime import (
    collect_runtime_metadata,
)

from .config import Exp61Config, VALID_DOMAINS, VALID_SPLITS
from .data import build_exp61_dataloader
from .epoch import train_epoch_with_metrics
from .evaluation import evaluate_split_with_loss
from .history import EpochHistoryWriter, history_row
from .preflight import run_exp61_preflight


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_sha256(config: Exp61Config) -> str:
    return hashlib.sha256(
        config.canonical_json().encode("utf-8")
    ).hexdigest()


def _domain_run_dir(
    config: Exp61Config,
    domain: str,
) -> Path:
    return config.run_root / domain


def _checkpoint_dir(
    config: Exp61Config,
    domain: str,
) -> Path:
    return _domain_run_dir(config, domain) / "checkpoints"


def _history_path(
    config: Exp61Config,
    domain: str,
) -> Path:
    return _domain_run_dir(config, domain) / "history.csv"


def _evaluation_dir(
    config: Exp61Config,
    train_domain: str,
    eval_domain: str,
    split: str,
) -> Path:
    return (
        config.results_root
        / "evaluations"
        / f"{train_domain}_to_{eval_domain}"
        / split
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_scheduler(
    config: Exp61Config,
    optimizer,
    *,
    optimizer_steps_per_epoch: int,
) -> ThreePhaseLRScheduler:
    return ThreePhaseLRScheduler(
        optimizer,
        optimizer_steps_per_epoch=(
            optimizer_steps_per_epoch
        ),
        config=SchedulerConfig(
            total_epochs=40,
            warmup_epochs=5,
            joint_cosine_end_epoch=35,
            warmup_start_learning_rate=(
                config.warmup_start_learning_rate
            ),
            peak_learning_rate=config.peak_learning_rate,
            cosine_minimum_learning_rate=(
                config.cosine_minimum_learning_rate
            ),
            heads_only_learning_rate=(
                config.heads_only_learning_rate
            ),
        ),
    )


def train_domain(
    config: Exp61Config,
    *,
    domain: str,
    device: torch.device,
) -> dict[str, Any]:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown training domain: {domain}")

    run_dir = _domain_run_dir(config, domain)
    checkpoint_manager = CheckpointManager(
        _checkpoint_dir(config, domain)
    )
    progress_writer = ProgressWriter(
        run_dir / "progress.json"
    )
    history_writer = EpochHistoryWriter(
        _history_path(config, domain)
    )
    completion_path = run_dir / "TRAINING_COMPLETE.json"
    logger = configure_logging(
        run_dir / "train.log",
        logger_name=f"speech_mtl.exp61.{domain}",
    )

    if (
        completion_path.is_file()
        and checkpoint_manager.has_best()
        and checkpoint_manager.has_last()
    ):
        logger.info(
            "Training already complete for %s; skipping.",
            domain,
        )
        return _load_json(completion_path)

    seed_report = seed_everything(
        config.seed,
        deterministic_algorithms=False,
    )

    train_loader, train_sampler, train_dataset = (
        build_exp61_dataloader(
            config,
            domain=domain,
            split="train",
            training=True,
        )
    )
    dev_loader, _, dev_dataset = (
        build_exp61_dataloader(
            config,
            domain=domain,
            split="dev",
            training=False,
        )
    )

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader)
        / config.gradient_accumulation_steps
    )

    model = build_exp61_model().to(device)
    parameter_report = audit_model_parameters(
        model,
        enforce_exact=True,
    )

    optimizer = build_optimizer(
        model,
        OptimizerConfig(
            peak_learning_rate=config.peak_learning_rate,
            weight_decay=config.weight_decay,
        ),
    )
    scheduler = _build_scheduler(
        config,
        optimizer,
        optimizer_steps_per_epoch=(
            optimizer_steps_per_epoch
        ),
    )
    amp_controller = AmpController(
        device=device,
        enabled=config.amp_enabled,
    )

    config_hash = _config_sha256(config)
    train_index_hash = sha256_file(
        config.cache_root / domain / "train" / "index.csv"
    )
    dev_index_hash = sha256_file(
        config.cache_root / domain / "dev" / "index.csv"
    )

    state = TrainingState()
    start_epoch = 1

    if checkpoint_manager.has_last():
        state, checkpoint_extra = load_training_checkpoint(
            checkpoint_manager.last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_controller=amp_controller,
            map_location=device,
            restore_rng=True,
        )

        expected_extra = {
            "domain": domain,
            "config_sha256": config_hash,
            "train_index_sha256": train_index_hash,
            "dev_index_sha256": dev_index_hash,
            "optimizer_steps_per_epoch": (
                optimizer_steps_per_epoch
            ),
        }

        for key, expected in expected_extra.items():
            actual = checkpoint_extra.get(key)

            if actual != expected:
                raise RuntimeError(
                    f"Checkpoint contract mismatch for {key}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

        start_epoch = state.epoch + 1
        logger.info(
            "Resuming %s from epoch %d; next epoch=%d.",
            domain,
            state.epoch,
            start_epoch,
        )

    completed_history_epochs = (
        history_writer.completed_epochs()
    )

    expected_completed_epochs = set(
        range(1, start_epoch)
    )

    if completed_history_epochs != expected_completed_epochs:
        raise RuntimeError(
            "History/checkpoint mismatch: "
            f"history={sorted(completed_history_epochs)}, "
            f"expected={sorted(expected_completed_epochs)}"
        )

    run_metadata = {
        "created_utc": _utc_now(),
        "domain": domain,
        "config_sha256": config_hash,
        "seed": seed_report,
        "runtime": collect_runtime_metadata(),
        "parameter_report": parameter_report.to_dict(),
        "train_sampler": train_sampler.diagnostics(),
        "optimizer_steps_per_epoch": (
            optimizer_steps_per_epoch
        ),
        "train_index_sha256": train_index_hash,
        "dev_index_sha256": dev_index_hash,
    }
    atomic_write_json(
        run_dir / "RUN_METADATA.json",
        run_metadata,
    )

    logger.info(
        "Starting domain=%s | train=%d | dev=%d | "
        "batches/epoch=%d | optimizer_steps/epoch=%d.",
        domain,
        len(train_dataset),
        len(dev_dataset),
        len(train_loader),
        optimizer_steps_per_epoch,
    )

    for epoch in range(start_epoch, config.total_epochs + 1):
        train_sampler.set_epoch(epoch)

        train_report = train_epoch_with_metrics(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_controller=amp_controller,
            device=device,
            epoch=epoch,
            training_state=state,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            maximum_gradient_norm=(
                config.maximum_gradient_norm
            ),
            progress_writer=progress_writer,
            progress_interval_batches=(
                config.progress_interval_batches
            ),
            show_progress_bar=True,
        )

        dev_report = evaluate_split_with_loss(
            model=model,
            dataloader=dev_loader,
            device=device,
            use_amp=config.amp_enabled,
            predictions_path=None,
            show_progress_bar=True,
        )

        selection_metric = float(
            dev_report["summary"][
                "selection_metric"
            ]["value"]
        )
        improved = selection_metric > state.best_dev_metric

        if improved:
            state.best_dev_metric = selection_metric
            state.best_epoch = epoch

        row = history_row(train_report, dev_report)
        history_writer.append(row)

        checkpoint_extra = {
            "domain": domain,
            "config_sha256": config_hash,
            "train_index_sha256": train_index_hash,
            "dev_index_sha256": dev_index_hash,
            "optimizer_steps_per_epoch": (
                optimizer_steps_per_epoch
            ),
            "epoch_train_report": train_report,
            "epoch_dev_report": dev_report,
        }

        checkpoint_manager.save_last(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            amp_controller=amp_controller,
            state=state,
            extra=checkpoint_extra,
        )

        if improved:
            checkpoint_manager.save_best(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                amp_controller=amp_controller,
                state=state,
                extra=checkpoint_extra,
            )
            atomic_write_json(
                run_dir / "best_dev_metrics.json",
                {
                    "epoch": epoch,
                    "selection_metric": selection_metric,
                    "dev_report": dev_report,
                    "checkpoint": str(
                        checkpoint_manager.best_path
                    ),
                },
            )

        progress_writer.write(
            {
                "status": "EPOCH_COMPLETE",
                "domain": domain,
                "epoch": epoch,
                "phase": state.phase_name,
                "best_epoch": state.best_epoch,
                "best_dev_metric": state.best_dev_metric,
                "train": train_report,
                "dev": dev_report,
            }
        )

        logger.info(
            "Epoch %d complete | train_loss=%.6f | "
            "dev_loss=%.6f | dev_selection=%.6f | "
            "best=%.6f at epoch %d.",
            epoch,
            train_report["mean_total_loss"],
            dev_report["mean_total_loss"],
            selection_metric,
            state.best_dev_metric,
            state.best_epoch,
        )

    state.completed = True

    checkpoint_manager.save_last(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_controller=amp_controller,
        state=state,
        extra={
            "domain": domain,
            "config_sha256": config_hash,
            "train_index_sha256": train_index_hash,
            "dev_index_sha256": dev_index_hash,
            "optimizer_steps_per_epoch": (
                optimizer_steps_per_epoch
            ),
            "training_complete": True,
            "training_completed_utc": _utc_now(),
        },
    )

    completion = {
        "status": "TRAINING_COMPLETE",
        "created_utc": _utc_now(),
        "domain": domain,
        "epochs": config.total_epochs,
        "best_epoch": state.best_epoch,
        "best_dev_metric": state.best_dev_metric,
        "best_checkpoint": str(
            checkpoint_manager.best_path
        ),
        "last_checkpoint": str(
            checkpoint_manager.last_path
        ),
        "history": str(_history_path(config, domain)),
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "optimizer_steps": state.optimizer_step,
    }
    atomic_write_json(completion_path, completion)
    progress_writer.write(completion)

    train_dataset.close()
    dev_dataset.close()

    logger.info(
        "TRAINING_COMPLETE domain=%s best_epoch=%d "
        "best_dev_metric=%.6f",
        domain,
        state.best_epoch,
        state.best_dev_metric,
    )

    return completion


def evaluate_condition(
    config: Exp61Config,
    *,
    train_domain: str,
    eval_domain: str,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    if train_domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown train_domain: {train_domain}"
        )

    if eval_domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown eval_domain: {eval_domain}"
        )

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    if split == "train":
        raise ValueError(
            "Final queue evaluation is restricted to dev/test"
        )

    output_dir = _evaluation_dir(
        config,
        train_domain,
        eval_domain,
        split,
    )
    result_path = output_dir / "evaluation.json"
    marker_path = output_dir / "EVALUATION_COMPLETE.json"
    checkpoint_path = (
        _checkpoint_dir(config, train_domain) / "best.pt"
    )

    if result_path.is_file() and marker_path.is_file():
        marker = _load_json(marker_path)

        if marker.get("checkpoint_sha256") == sha256_file(
            checkpoint_path
        ):
            return _load_json(result_path)

    model = build_exp61_model().to(device)
    state, extra = load_training_checkpoint(
        checkpoint_path,
        model=model,
        map_location=device,
        restore_rng=False,
    )

    loader, _, dataset = build_exp61_dataloader(
        config,
        domain=eval_domain,
        split=split,
        training=False,
    )

    predictions_path = (
        output_dir / "predictions.csv"
        if split == "test"
        else None
    )

    report = evaluate_split_with_loss(
        model=model,
        dataloader=loader,
        device=device,
        use_amp=config.amp_enabled,
        predictions_path=predictions_path,
        show_progress_bar=True,
    )
    report.update(
        {
            "status": "EVALUATION_COMPLETE",
            "created_utc": _utc_now(),
            "train_domain": train_domain,
            "eval_domain": eval_domain,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(
                checkpoint_path
            ),
            "checkpoint_epoch": state.epoch,
            "checkpoint_best_epoch": state.best_epoch,
            "checkpoint_best_dev_metric": (
                state.best_dev_metric
            ),
            "checkpoint_extra_domain": extra.get("domain"),
            "eval_index_sha256": sha256_file(
                config.cache_root
                / eval_domain
                / split
                / "index.csv"
            ),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, report)
    atomic_write_json(
        marker_path,
        {
            "status": "EVALUATION_COMPLETE",
            "created_utc": _utc_now(),
            "result": str(result_path),
            "checkpoint_sha256": report[
                "checkpoint_sha256"
            ],
            "examples": report["examples"],
        },
    )

    dataset.close()
    return report


def generate_report(
    config: Exp61Config,
) -> dict[str, Any]:
    one_history = pd.read_csv(
        _history_path(config, "one_bit")
    )
    sixteen_history = pd.read_csv(
        _history_path(config, "sixteen_bit")
    )

    def evaluation(
        train_domain: str,
        eval_domain: str,
    ) -> dict[str, Any]:
        return _load_json(
            _evaluation_dir(
                config,
                train_domain,
                eval_domain,
                "test",
            )
            / "evaluation.json"
        )

    one_to_one = evaluation("one_bit", "one_bit")
    one_to_sixteen = evaluation(
        "one_bit",
        "sixteen_bit",
    )
    sixteen_to_sixteen = evaluation(
        "sixteen_bit",
        "sixteen_bit",
    )
    sixteen_to_one = evaluation(
        "sixteen_bit",
        "one_bit",
    )

    figures = config.report_root / "figures"
    tables = config.report_root / "tables"

    generated: dict[str, str] = {}

    generated.update(
        {
            "figure_01_one_bit_loss": (
                save_training_curves(
                    one_history,
                    loss_output=(
                        figures
                        / "figure_01_one_bit_loss.png"
                    ),
                    accuracy_output=(
                        figures
                        / "figure_02_one_bit_accuracy.png"
                    ),
                    title_prefix="1-bit model",
                )["loss_figure"]
            ),
            "figure_02_one_bit_accuracy": str(
                figures / "figure_02_one_bit_accuracy.png"
            ),
        }
    )

    generated[
        "figure_03_one_bit_matched_confusion"
    ] = save_composite_confusion_figure(
        one_to_one["confusion_matrices"],
        output_path=(
            figures
            / "figure_03_one_bit_matched_confusion.png"
        ),
        title="1-bit model evaluated on 1-bit speech",
    )

    sixteen_curves = save_training_curves(
        sixteen_history,
        loss_output=(
            figures / "figure_04_sixteen_bit_loss.png"
        ),
        accuracy_output=(
            figures / "figure_05_sixteen_bit_accuracy.png"
        ),
        title_prefix="16-bit model",
    )
    generated[
        "figure_04_sixteen_bit_loss"
    ] = sixteen_curves["loss_figure"]
    generated[
        "figure_05_sixteen_bit_accuracy"
    ] = sixteen_curves["accuracy_figure"]

    generated[
        "figure_06_sixteen_bit_matched_confusion"
    ] = save_composite_confusion_figure(
        sixteen_to_sixteen["confusion_matrices"],
        output_path=(
            figures
            / "figure_06_sixteen_bit_matched_confusion.png"
        ),
        title="16-bit model evaluated on 16-bit speech",
    )

    cross_evaluations = {
        "1-bit → 1-bit": one_to_one,
        "1-bit → 16-bit": one_to_sixteen,
        "16-bit → 16-bit": sixteen_to_sixteen,
        "16-bit → 1-bit": sixteen_to_one,
    }

    generated[
        "figure_07_cross_precision_summary"
    ] = save_cross_precision_summary(
        cross_evaluations,
        output_path=(
            figures
            / "figure_07_cross_precision_summary.png"
        ),
    )

    generated[
        "appendix_a_one_to_sixteen_confusion"
    ] = save_composite_confusion_figure(
        one_to_sixteen["confusion_matrices"],
        output_path=(
            figures
            / "appendix_a_one_to_sixteen_confusion.png"
        ),
        title="1-bit model evaluated on 16-bit speech",
    )
    generated[
        "appendix_b_sixteen_to_one_confusion"
    ] = save_composite_confusion_figure(
        sixteen_to_one["confusion_matrices"],
        output_path=(
            figures
            / "appendix_b_sixteen_to_one_confusion.png"
        ),
        title="16-bit model evaluated on 1-bit speech",
    )

    generated[
        "table_01_experimental_configuration"
    ] = save_experimental_configuration_table(
        config.raw,
        output_path=(
            tables
            / "table_01_experimental_configuration.csv"
        ),
    )
    generated[
        "table_02_matched_results"
    ] = save_matched_results_table(
        one_bit_evaluation=one_to_one,
        sixteen_bit_evaluation=sixteen_to_sixteen,
        output_path=(
            tables / "table_02_matched_results.csv"
        ),
    )
    generated[
        "table_03_per_language_dialect"
    ] = save_per_language_dialect_table(
        one_bit_evaluation=one_to_one,
        sixteen_bit_evaluation=sixteen_to_sixteen,
        output_path=(
            tables
            / "table_03_per_language_dialect.csv"
        ),
    )
    generated[
        "table_04_cross_precision"
    ] = save_cross_precision_table(
        cross_evaluations,
        output_path=(
            tables / "table_04_cross_precision.csv"
        ),
    )

    runtime_rows = []

    for name, report in (
        ("one_bit_to_one_bit_test", one_to_one),
        ("one_bit_to_sixteen_bit_test", one_to_sixteen),
        (
            "sixteen_bit_to_sixteen_bit_test",
            sixteen_to_sixteen,
        ),
        ("sixteen_bit_to_one_bit_test", sixteen_to_one),
    ):
        runtime_rows.append(
            {
                "run": name,
                "examples": report["examples"],
                "elapsed_seconds": report[
                    "elapsed_seconds"
                ],
                "examples_per_second": report[
                    "examples_per_second"
                ],
            }
        )

    generated["table_05_runtime"] = save_runtime_table(
        runtime_rows,
        output_path=(
            tables / "table_05_runtime.csv"
        ),
    )

    missing = [
        path
        for path in generated.values()
        if not Path(path).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Report generation missed artifacts: {missing}"
        )

    completion = {
        "status": "REPORT_COMPLETE",
        "created_utc": _utc_now(),
        "main_figures": 7,
        "appendix_figures": 2,
        "tables": 5,
        "artifacts": generated,
    }
    atomic_write_json(
        config.report_root / "REPORT_COMPLETE.json",
        completion,
    )
    return completion


def run_full_queue(
    config: Exp61Config,
    *,
    device: torch.device,
) -> dict[str, Any]:
    queue_progress = ProgressWriter(
        config.run_root / "QUEUE_PROGRESS.json"
    )
    queue_progress.write(
        {
            "status": "STARTING",
            "created_utc": _utc_now(),
        }
    )

    preflight = run_exp61_preflight(config)
    atomic_write_json(
        config.run_root / "PREFLIGHT.json",
        preflight,
    )
    queue_progress.write(
        {
            "status": "PREFLIGHT_COMPLETE",
            "created_utc": _utc_now(),
        }
    )

    train_domain(
        config,
        domain="one_bit",
        device=device,
    )
    queue_progress.write(
        {
            "status": "ONE_BIT_TRAINING_COMPLETE",
            "created_utc": _utc_now(),
        }
    )

    evaluate_condition(
        config,
        train_domain="one_bit",
        eval_domain="one_bit",
        split="dev",
        device=device,
    )
    evaluate_condition(
        config,
        train_domain="one_bit",
        eval_domain="one_bit",
        split="test",
        device=device,
    )
    evaluate_condition(
        config,
        train_domain="one_bit",
        eval_domain="sixteen_bit",
        split="test",
        device=device,
    )

    train_domain(
        config,
        domain="sixteen_bit",
        device=device,
    )
    queue_progress.write(
        {
            "status": "SIXTEEN_BIT_TRAINING_COMPLETE",
            "created_utc": _utc_now(),
        }
    )

    evaluate_condition(
        config,
        train_domain="sixteen_bit",
        eval_domain="sixteen_bit",
        split="dev",
        device=device,
    )
    evaluate_condition(
        config,
        train_domain="sixteen_bit",
        eval_domain="sixteen_bit",
        split="test",
        device=device,
    )
    evaluate_condition(
        config,
        train_domain="sixteen_bit",
        eval_domain="one_bit",
        split="test",
        device=device,
    )

    report = generate_report(config)

    completion = {
        "status": "EXPERIMENT_COMPLETE",
        "created_utc": _utc_now(),
        "report": report,
        "one_bit_training": str(
            _domain_run_dir(config, "one_bit")
            / "TRAINING_COMPLETE.json"
        ),
        "sixteen_bit_training": str(
            _domain_run_dir(config, "sixteen_bit")
            / "TRAINING_COMPLETE.json"
        ),
    }
    atomic_write_json(
        config.run_root / "EXPERIMENT_COMPLETE.json",
        completion,
    )
    queue_progress.write(completion)
    return completion

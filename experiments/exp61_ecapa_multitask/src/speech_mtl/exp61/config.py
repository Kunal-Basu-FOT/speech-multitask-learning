from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech_mtl.common.utils.config import get_required, load_yaml


VALID_DOMAINS = ("one_bit", "sixteen_bit")
VALID_SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class Exp61Config:
    project_root: Path
    cache_root: Path
    run_root: Path
    results_root: Path
    report_root: Path
    physical_gpu: int
    seed: int
    feature_dimension: int
    total_epochs: int
    max_batch_frames: int
    max_batch_size: int
    number_of_buckets: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    gradient_accumulation_steps: int
    maximum_gradient_norm: float
    amp_enabled: bool
    progress_interval_batches: int
    peak_learning_rate: float
    weight_decay: float
    warmup_start_learning_rate: float
    cosine_minimum_learning_rate: float
    heads_only_learning_rate: float
    expected_split_counts: dict[str, int]
    raw: dict[str, Any]

    def canonical_json(self) -> str:
        return json.dumps(
            self.raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def load_exp61_config(path: Path | str) -> Exp61Config:
    raw = load_yaml(path)

    project_root = Path(get_required(raw, "paths.project_root"))
    cache_root = Path(get_required(raw, "paths.cache_root"))
    run_root = Path(get_required(raw, "paths.run_root"))
    results_root = Path(get_required(raw, "paths.results_root"))
    report_root = Path(get_required(raw, "paths.report_root"))

    config = Exp61Config(
        project_root=project_root,
        cache_root=cache_root,
        run_root=run_root,
        results_root=results_root,
        report_root=report_root,
        physical_gpu=int(get_required(raw, "runtime.physical_gpu")),
        seed=int(get_required(raw, "runtime.seed")),
        feature_dimension=int(
            get_required(raw, "model.feature_dimension")
        ),
        total_epochs=int(get_required(raw, "training.total_epochs")),
        max_batch_frames=int(
            get_required(raw, "data.max_batch_frames")
        ),
        max_batch_size=int(
            get_required(raw, "data.max_batch_size")
        ),
        number_of_buckets=int(
            get_required(raw, "data.number_of_buckets")
        ),
        num_workers=int(get_required(raw, "data.num_workers")),
        pin_memory=bool(get_required(raw, "data.pin_memory")),
        persistent_workers=bool(
            get_required(raw, "data.persistent_workers")
        ),
        prefetch_factor=int(
            get_required(raw, "data.prefetch_factor")
        ),
        gradient_accumulation_steps=int(
            get_required(
                raw,
                "training.gradient_accumulation_steps",
            )
        ),
        maximum_gradient_norm=float(
            get_required(raw, "training.maximum_gradient_norm")
        ),
        amp_enabled=bool(get_required(raw, "training.amp_enabled")),
        progress_interval_batches=int(
            get_required(
                raw,
                "training.progress_interval_batches",
            )
        ),
        peak_learning_rate=float(
            get_required(raw, "optimizer.peak_learning_rate")
        ),
        weight_decay=float(
            get_required(raw, "optimizer.weight_decay")
        ),
        warmup_start_learning_rate=float(
            get_required(
                raw,
                "scheduler.warmup_start_learning_rate",
            )
        ),
        cosine_minimum_learning_rate=float(
            get_required(
                raw,
                "scheduler.cosine_minimum_learning_rate",
            )
        ),
        heads_only_learning_rate=float(
            get_required(
                raw,
                "scheduler.heads_only_learning_rate",
            )
        ),
        expected_split_counts={
            split: int(
                get_required(
                    raw,
                    f"data.expected_split_counts.{split}",
                )
            )
            for split in VALID_SPLITS
        },
        raw=raw,
    )

    _validate_config(config)
    return config


def _validate_config(config: Exp61Config) -> None:
    if config.physical_gpu != 1:
        raise ValueError(
            "EXP 6.1 is frozen to physical GPU 1 on WTC9"
        )

    if config.feature_dimension != 768:
        raise ValueError(
            "EXP 6.1 feature_dimension must remain 768"
        )

    if config.total_epochs != 40:
        raise ValueError("EXP 6.1 total_epochs must remain 40")

    if config.seed != 1337:
        raise ValueError("EXP 6.1 seed must remain 1337")

    if config.expected_split_counts != {
        "train": 146_359,
        "dev": 11_507,
        "test": 17_550,
    }:
        raise ValueError(
            "Cache-aligned split counts differ from the frozen contract"
        )

    positive_integer_fields = {
        "max_batch_frames": config.max_batch_frames,
        "max_batch_size": config.max_batch_size,
        "number_of_buckets": config.number_of_buckets,
        "gradient_accumulation_steps": (
            config.gradient_accumulation_steps
        ),
        "progress_interval_batches": (
            config.progress_interval_batches
        ),
    }

    for name, value in positive_integer_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    if config.maximum_gradient_norm <= 0:
        raise ValueError(
            "maximum_gradient_norm must be positive"
        )

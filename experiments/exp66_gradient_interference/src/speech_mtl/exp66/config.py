from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import yaml


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    cache_root: Path
    run_root: Path
    results_root: Path
    report_root: Path


@dataclass(frozen=True)
class DataConfig:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    max_batch_frames: int
    max_batch_size: int
    number_of_buckets: int


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    embedding_dim: int
    head_hidden: int
    head_dropout: float
    dropout: float
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    dilations: tuple[int, ...]
    groups: tuple[int, ...]
    attention_channels: int
    res2net_scale: int
    se_channels: int
    global_context: bool


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    phase1_epochs: int
    phase2_epochs: int
    phase3_epochs: int
    warmup_start_lr: float
    peak_lr: float
    cosine_min_lr: float
    phase3_task_lr: float
    weight_decay: float
    grad_clip_norm: float
    amp: bool
    max_amp_skips_per_epoch: int
    max_consecutive_amp_skips: int
    betas: tuple[float, float]
    eps: float
    progress_every_batches: int
    gradient_diagnostic_interval: int
    gradnorm_alpha: float
    gradnorm_weight_lr: float


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @property
    def config_fingerprint_payload(self) -> dict:
        return json.loads(
            json.dumps(
                self,
                default=lambda value: str(value)
                if isinstance(value, Path)
                else value.__dict__,
            )
        )


def load_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    paths = payload["paths"]
    data = payload["data"]
    model = payload["model"]
    training = payload["training"]
    return ExperimentConfig(
        paths=PathsConfig(
            project_root=Path(paths["project_root"]),
            cache_root=Path(paths["cache_root"]),
            run_root=Path(paths["run_root"]),
            results_root=Path(paths["results_root"]),
            report_root=Path(paths["report_root"]),
        ),
        data=DataConfig(**data),
        model=ModelConfig(
            input_dim=int(model["input_dim"]),
            embedding_dim=int(model["embedding_dim"]),
            head_hidden=int(model["head_hidden"]),
            head_dropout=float(model["head_dropout"]),
            dropout=float(model["dropout"]),
            channels=tuple(int(x) for x in model["channels"]),
            kernels=tuple(int(x) for x in model["kernels"]),
            dilations=tuple(int(x) for x in model["dilations"]),
            groups=tuple(int(x) for x in model["groups"]),
            attention_channels=int(model["attention_channels"]),
            res2net_scale=int(model["res2net_scale"]),
            se_channels=int(model["se_channels"]),
            global_context=bool(model["global_context"]),
        ),
        training=TrainingConfig(
            **{**training, "betas": tuple(float(x) for x in training["betas"])}
        ),
    )


def dump_config_snapshot(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cfg.config_fingerprint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

"""Strict configuration loading for reduced EXP 6.2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os

try:
    import yaml
except ImportError as exc:  # pragma: no cover - checked by environment validation
    raise RuntimeError("PyYAML is required. Install it in env_speech_mtl.") from exc


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    cache_root: Path
    run_root: Path
    results_root: Path
    report_root: Path
    exp61_results_root: Path


@dataclass(frozen=True)
class DataConfig:
    feature_dim: int = 768
    max_batch_frames: int = 12000
    max_batch_size: int = 32
    number_of_buckets: int = 40
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 768
    embedding_dim: int = 192
    channels: tuple[int, ...] = (480, 480, 480, 480, 1440)
    kernels: tuple[int, ...] = (5, 3, 3, 3, 1)
    dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    groups: tuple[int, ...] = (1, 1, 1, 1, 1)
    res2net_scale: int = 8
    se_channels: int = 128
    attention_channels: int = 128
    global_context: bool = True
    dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    epochs: int = 40
    phase1_epochs: int = 5
    phase2_epochs: int = 30
    phase3_epochs: int = 5
    warmup_start_lr: float = 1e-5
    peak_lr: float = 1e-4
    cosine_min_lr: float = 5e-6
    phase3_task_lr: float = 2e-5
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    amp: bool = True
    max_amp_skips_per_epoch: int = 100
    max_consecutive_amp_skips: int = 20
    progress_every_batches: int = 50


@dataclass(frozen=True)
class EvaluationConfig:
    representative_dialect_language: str = "bn"
    save_all_dialect_confusions: bool = True
    figure_dpi: int = 300


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def config_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "cache_root": str(self.paths.cache_root),
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
        }


def _path(value: str, *, env_name: str | None = None) -> Path:
    if env_name and os.environ.get(env_name):
        value = os.environ[env_name]
    return Path(value).expanduser().resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    p = raw["paths"]
    paths = PathsConfig(
        project_root=_path(p["project_root"], env_name="EXP62_PROJECT_ROOT"),
        cache_root=_path(p["cache_root"], env_name="EXP62_CACHE_ROOT"),
        run_root=_path(p["run_root"], env_name="EXP62_RUN_ROOT"),
        results_root=_path(p["results_root"], env_name="EXP62_RESULTS_ROOT"),
        report_root=_path(p["report_root"], env_name="EXP62_REPORT_ROOT"),
        exp61_results_root=_path(p["exp61_results_root"], env_name="EXP61_RESULTS_ROOT"),
    )

    def merge(cls: type, section: str):
        values = raw.get(section, {}) or {}
        return cls(**values)

    cfg = ExperimentConfig(
        paths=paths,
        data=merge(DataConfig, "data"),
        model=merge(ModelConfig, "model"),
        training=merge(TrainingConfig, "training"),
        evaluation=merge(EvaluationConfig, "evaluation"),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    t = cfg.training
    if t.phase1_epochs + t.phase2_epochs + t.phase3_epochs != t.epochs:
        raise ValueError("phase1 + phase2 + phase3 must equal total epochs")
    if cfg.data.feature_dim != cfg.model.input_dim:
        raise ValueError("data.feature_dim must equal model.input_dim")
    if cfg.data.max_batch_frames <= 0 or cfg.data.max_batch_size <= 0:
        raise ValueError("batch budgets must be positive")
    if cfg.model.embedding_dim != 192:
        raise ValueError("EXP 6.2 control requires a 192-dimensional embedding")
    if t.seed != 1337:
        raise ValueError("EXP 6.2 frozen seed is 1337")


def dump_config_snapshot(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.config_fingerprint_payload, indent=2, sort_keys=True))

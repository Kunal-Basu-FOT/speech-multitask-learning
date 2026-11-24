from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .metrics import multitask_metrics
from .utils import atomic_json, atomic_text, utc_now


TASK_COLUMNS = {
    "language_macro_f1": ("language", "macro_f1"),
    "dialect_language_macro_f1": ("dialect", "language_macro_f1"),
    "gender_macro_f1": ("gender", "macro_f1"),
    "three_task_mean_macro_f1": ("three_task_mean_macro_f1",),
}


def _metric(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matched_metrics_path(config: Config, domain: str, duration: str) -> Path:
    return (
        config.path("results_root")
        / domain
        / duration
        / f"input_{domain}"
        / f"test_{duration}"
        / "metrics.json"
    )


def _cross_metrics_path(config: Config, domain: str, duration: str) -> Path:
    other = "sixteen_bit" if domain == "one_bit" else "one_bit"
    return (
        config.path("results_root")
        / domain
        / duration
        / f"input_{other}"
        / f"test_{duration}"
        / "metrics.json"
    )


def _duration_metrics_path(config: Config, domain: str, train_duration: str, test_duration: str) -> Path:
    return (
        config.path("results_root")
        / domain
        / train_duration
        / f"input_{domain}"
        / f"test_{test_duration}"
        / "metrics.json"
    )


def _reference_table(config: Config) -> pd.DataFrame:
    path = config.path("exp61_root") / "reports/ecapa_7p2m/exp61/tables/table_02_matched_results.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["domain"].isin(["1-bit", "16-bit"])].copy()
    if len(frame) != 2:
        raise RuntimeError(f"Unexpected EXP 6.1 reference table: {path}")
    return frame


def _matched_frame(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in config.domains:
        for duration in config.durations:
            payload = _load_json(_matched_metrics_path(config, domain, duration))
            rows.append(
                {
                    "domain": domain,
                    "duration": duration,
                    "language_accuracy": payload["language"]["accuracy"],
                    "language_macro_f1": payload["language"]["macro_f1"],
                    "dialect_sample_weighted_accuracy": payload["dialect"]["sample_weighted_accuracy"],
                    "dialect_language_macro_accuracy": payload["dialect"]["language_macro_accuracy"],
                    "dialect_language_macro_f1": payload["dialect"]["language_macro_f1"],
                    "gender_accuracy": payload["gender"]["accuracy"],
                    "gender_macro_f1": payload["gender"]["macro_f1"],
                    "three_task_mean_macro_f1": payload["three_task_mean_macro_f1"],
                    "examples": payload["examples"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "examples_per_second": payload["examples_per_second"],
                }
            )
    return pd.DataFrame(rows)


def _cross_frame(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in config.domains:
        other = "sixteen_bit" if domain == "one_bit" else "one_bit"
        for duration in config.durations:
            matched = _load_json(_matched_metrics_path(config, domain, duration))
            cross = _load_json(_cross_metrics_path(config, domain, duration))
            rows.append(
                {
                    "train_domain": domain,
                    "input_domain": other,
                    "duration": duration,
                    "matched_language_macro_f1": matched["language"]["macro_f1"],
                    "cross_language_macro_f1": cross["language"]["macro_f1"],
                    "language_delta_cross_minus_matched": cross["language"]["macro_f1"] - matched["language"]["macro_f1"],
                    "matched_dialect_language_macro_f1": matched["dialect"]["language_macro_f1"],
                    "cross_dialect_language_macro_f1": cross["dialect"]["language_macro_f1"],
                    "dialect_delta_cross_minus_matched": cross["dialect"]["language_macro_f1"] - matched["dialect"]["language_macro_f1"],
                    "matched_gender_macro_f1": matched["gender"]["macro_f1"],
                    "cross_gender_macro_f1": cross["gender"]["macro_f1"],
                    "gender_delta_cross_minus_matched": cross["gender"]["macro_f1"] - matched["gender"]["macro_f1"],
                    "matched_three_task_mean_macro_f1": matched["three_task_mean_macro_f1"],
                    "cross_three_task_mean_macro_f1": cross["three_task_mean_macro_f1"],
                    "overall_delta_cross_minus_matched": cross["three_task_mean_macro_f1"] - matched["three_task_mean_macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def _duration_transfer_frame(config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in config.domains:
        for train_duration in config.durations:
            for test_duration in config.durations:
                payload = _load_json(
                    _duration_metrics_path(config, domain, train_duration, test_duration)
                )
                rows.append(
                    {
                        "domain": domain,
                        "train_duration": train_duration,
                        "test_duration": test_duration,
                        "language_macro_f1": payload["language"]["macro_f1"],
                        "dialect_language_macro_f1": payload["dialect"]["language_macro_f1"],
                        "gender_macro_f1": payload["gender"]["macro_f1"],
                        "three_task_mean_macro_f1": payload["three_task_mean_macro_f1"],
                    }
                )
    return pd.DataFrame(rows)


def _configuration_frame(config: Config) -> pd.DataFrame:
    rows = []
    for domain in config.domains:
        for duration in config.durations:
            marker = _load_json(
                config.path("run_root") / domain / duration / "TRAINING_COMPLETE.json"
            )
            rows.append(
                {
                    "domain": domain,
                    "duration": duration,
                    "front_end": "80-bin log-Mel from quantized waveform",
                    "projection": "Linear(80,768)",
                    "encoder": "EXP6.1 compact ECAPA-TDNN",
                    "tasks": "LID + oracle-routed dialect + gender",
                    "epochs": marker["epochs"],
                    "train_examples": marker["train_examples"],
                    "dev_examples": marker["dev_examples"],
                    "best_epoch": marker["best_epoch"],
                    "best_dev_metric": marker["best_dev_metric"],
                    "amp_skips_total": marker["amp_skips_total"],
                    "total_parameters": marker["model_parameters"]["total"],
                    "frontend_projection_parameters": marker["model_parameters"]["frontend_projection"],
                }
            )
    return pd.DataFrame(rows)


def _full_frontend_frame(config: Config, matched: pd.DataFrame) -> pd.DataFrame:
    reference = _reference_table(config)
    rows: list[dict[str, Any]] = []
    for domain, reference_name in (("one_bit", "1-bit"), ("sixteen_bit", "16-bit")):
        logmel = matched[(matched.domain == domain) & (matched.duration == "full")].iloc[0]
        wav = reference[reference.domain == reference_name].iloc[0]
        rows.extend(
            [
                {
                    "domain": domain,
                    "front_end": "waveform_logmel80",
                    "language_accuracy": logmel.language_accuracy,
                    "language_macro_f1": logmel.language_macro_f1,
                    "dialect_language_macro_accuracy": logmel.dialect_language_macro_accuracy,
                    "dialect_language_macro_f1": logmel.dialect_language_macro_f1,
                    "gender_accuracy": logmel.gender_accuracy,
                    "gender_macro_f1": logmel.gender_macro_f1,
                    "three_task_mean_macro_f1": logmel.three_task_mean_macro_f1,
                },
                {
                    "domain": domain,
                    "front_end": "indicwav2vec2_layer11",
                    "language_accuracy": wav.language_accuracy,
                    "language_macro_f1": wav.language_macro_f1,
                    "dialect_language_macro_accuracy": wav.dialect_language_macro_accuracy,
                    "dialect_language_macro_f1": wav.dialect_language_macro_f1,
                    "gender_accuracy": wav.gender_accuracy,
                    "gender_macro_f1": wav.gender_macro_f1,
                    "three_task_mean_macro_f1": wav.three_task_mean_macro_f1,
                },
            ]
        )
    return pd.DataFrame(rows)


def _gain_frame(full_frontend: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "language_accuracy",
        "language_macro_f1",
        "dialect_language_macro_accuracy",
        "dialect_language_macro_f1",
        "gender_accuracy",
        "gender_macro_f1",
        "three_task_mean_macro_f1",
    ]
    for domain in ("one_bit", "sixteen_bit"):
        logmel = full_frontend[(full_frontend.domain == domain) & (full_frontend.front_end == "waveform_logmel80")].iloc[0]
        wav = full_frontend[(full_frontend.domain == domain) & (full_frontend.front_end == "indicwav2vec2_layer11")].iloc[0]
        for metric in metrics:
            baseline = float(logmel[metric])
            reference = float(wav[metric])
            error_baseline = 1.0 - baseline
            relative_error_reduction = (
                (error_baseline - (1.0 - reference)) / error_baseline
                if error_baseline > 0
                else float("nan")
            )
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "waveform_logmel80": baseline,
                    "indicwav2vec2_layer11": reference,
                    "absolute_gain": reference - baseline,
                    "percentage_point_gain": 100.0 * (reference - baseline),
                    "relative_error_reduction": relative_error_reduction,
                }
            )
    return pd.DataFrame(rows)


def _history_runtime_frame(config: Config) -> pd.DataFrame:
    rows = []
    for domain in config.domains:
        for duration in config.durations:
            history = pd.read_csv(config.path("run_root") / domain / duration / "history.csv")
            rows.append(
                {
                    "domain": domain,
                    "duration": duration,
                    "training_seconds": float(history["epoch_seconds"].sum()),
                    "mean_epoch_seconds": float(history["epoch_seconds"].mean()),
                    "mean_examples_per_second": float(history["examples_per_second"].mean()),
                    "amp_skips_total": int(history["amp_skips_total"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def _prediction_metrics(frame: pd.DataFrame, dialect_counts: dict[str, int]) -> dict[str, Any]:
    return multitask_metrics(
        true_lid=frame["true_language_index"].to_numpy(),
        pred_lid=frame["predicted_language_index"].to_numpy(),
        true_gender=frame["true_gender_index"].to_numpy(),
        pred_gender=frame["predicted_gender_index"].to_numpy(),
        true_dialect=frame["true_dialect_index"].to_numpy(),
        pred_dialect=frame["predicted_dialect_index"].to_numpy(),
        languages=frame["oracle_language"].astype(str).to_numpy(),
        dialect_class_counts=dialect_counts,
    )


def _locate_exp61_predictions(config: Config, domain: str, target_row: pd.Series) -> Path:
    root = config.path("exp61_root") / "results/ecapa_7p2m/exp61"
    candidates = list(root.rglob("predictions.csv"))
    scored: list[tuple[float, Path]] = []
    required = {
        "utt_id", "filepath", "oracle_language", "true_language_index",
        "predicted_language_index", "true_dialect_index", "predicted_dialect_index",
        "true_gender_index", "predicted_gender_index",
    }
    for path in candidates:
        try:
            frame = pd.read_csv(path, dtype={"utt_id": str, "oracle_language": str})
        except Exception:
            continue
        if len(frame) != int(config.expected_rows["test"]):
            continue
        if not required.issubset(frame.columns):
            continue
        metrics = _prediction_metrics(frame, dict(config.dialect_class_counts))
        score = (
            abs(metrics["language"]["accuracy"] - float(target_row.language_accuracy))
            + abs(metrics["gender"]["accuracy"] - float(target_row.gender_accuracy))
            + abs(metrics["dialect"]["language_macro_accuracy"] - float(target_row.dialect_language_macro_accuracy))
        )
        scored.append((score, path))
    if not scored:
        raise RuntimeError(f"No EXP 6.1 prediction candidate found for {domain}")
    scored.sort(key=lambda value: value[0])
    score, path = scored[0]
    if score > 1.0e-5:
        raise RuntimeError(
            f"Could not identify matched EXP 6.1 predictions for {domain}; best score={score} path={path}"
        )
    return path


def _bootstrap_frontend(config: Config, full_frontend: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.seed))
    reference = _reference_table(config)
    rows: list[dict[str, Any]] = []
    task_paths = {
        "language_macro_f1": ("language", "macro_f1"),
        "dialect_language_macro_f1": ("dialect", "language_macro_f1"),
        "gender_macro_f1": ("gender", "macro_f1"),
        "three_task_mean_macro_f1": ("three_task_mean_macro_f1",),
    }
    for domain, reference_name in (("one_bit", "1-bit"), ("sixteen_bit", "16-bit")):
        target = reference[reference.domain == reference_name].iloc[0]
        exp61_path = _locate_exp61_predictions(config, domain, target)
        logmel_path = (
            config.path("results_root")
            / domain
            / "full"
            / f"input_{domain}"
            / "test_full"
            / "predictions.csv"
        )
        left = pd.read_csv(exp61_path, dtype={"utt_id": str, "oracle_language": str})
        right = pd.read_csv(logmel_path, dtype={"utt_id": str, "oracle_language": str})
        if "speaker_id" not in left.columns:
            left["speaker_id"] = left["filepath"].map(lambda value: Path(str(value)).parent.name)
        merged = left.merge(
            right,
            on="utt_id",
            suffixes=("_wav2vec", "_logmel"),
            validate="one_to_one",
        )
        if len(merged) != int(config.expected_rows["test"]):
            raise RuntimeError("Paired prediction merge did not retain every test utterance")
        for truth in ("true_language_index", "true_dialect_index", "true_gender_index", "oracle_language"):
            if not np.array_equal(
                merged[f"{truth}_wav2vec"].astype(str).to_numpy(),
                merged[f"{truth}_logmel"].astype(str).to_numpy(),
            ):
                raise RuntimeError(f"Paired truth mismatch for {truth}")

        speaker_column = "speaker_id_wav2vec"
        language_column = "oracle_language_wav2vec"
        speaker_keys = (
            merged[language_column].astype(str)
            + ":"
            + merged[speaker_column].astype(str)
        ).to_numpy()
        groups = {
            key: np.flatnonzero(speaker_keys == key)
            for key in np.unique(speaker_keys)
        }
        speakers_by_language = {
            language: np.asarray(
                sorted(key for key in groups if key.startswith(f"{language}:")),
                dtype=object,
            )
            for language in config.dialect_class_counts
        }
        if any(len(values) == 0 for values in speakers_by_language.values()):
            raise RuntimeError("Speaker bootstrap could not find speakers for every language")

        differences = {task: [] for task in task_paths}
        replicates = int(config.bootstrap_replicates)
        for _ in range(replicates):
            sampled_keys: list[str] = []
            for language in config.dialect_class_counts:
                candidates = speakers_by_language[language]
                sampled_keys.extend(
                    str(value)
                    for value in rng.choice(candidates, size=len(candidates), replace=True)
                )
            indices = np.concatenate([groups[key] for key in sampled_keys])
            sample = merged.iloc[indices]
            common = pd.DataFrame(
                {
                    "true_language_index": sample["true_language_index_wav2vec"],
                    "true_dialect_index": sample["true_dialect_index_wav2vec"],
                    "true_gender_index": sample["true_gender_index_wav2vec"],
                    "oracle_language": sample["oracle_language_wav2vec"],
                }
            )
            wav_frame = common.copy()
            wav_frame["predicted_language_index"] = sample["predicted_language_index_wav2vec"].to_numpy()
            wav_frame["predicted_dialect_index"] = sample["predicted_dialect_index_wav2vec"].to_numpy()
            wav_frame["predicted_gender_index"] = sample["predicted_gender_index_wav2vec"].to_numpy()
            log_frame = common.copy()
            log_frame["predicted_language_index"] = sample["predicted_language_index_logmel"].to_numpy()
            log_frame["predicted_dialect_index"] = sample["predicted_dialect_index_logmel"].to_numpy()
            log_frame["predicted_gender_index"] = sample["predicted_gender_index_logmel"].to_numpy()
            wav_metrics = _prediction_metrics(wav_frame, dict(config.dialect_class_counts))
            log_metrics = _prediction_metrics(log_frame, dict(config.dialect_class_counts))
            for task, path in task_paths.items():
                differences[task].append(_metric(wav_metrics, path) - _metric(log_metrics, path))

        for task, values in differences.items():
            array = np.asarray(values, dtype=np.float64)
            rows.append(
                {
                    "domain": domain,
                    "metric": task,
                    "replicates": replicates,
                    "mean_difference_wav2vec_minus_logmel": float(array.mean()),
                    "ci95_lower": float(np.percentile(array, 2.5)),
                    "ci95_upper": float(np.percentile(array, 97.5)),
                    "fraction_positive": float(np.mean(array > 0.0)),
                    "exp61_predictions": str(exp61_path),
                    "logmel_predictions": str(logmel_path),
                }
            )
    return pd.DataFrame(rows)


def _save_main_figure(config: Config, matched: pd.DataFrame, full_frontend: pd.DataFrame, path: Path) -> None:
    labels = list(config.durations)
    x = np.arange(len(labels))
    panel_metrics = [
        ("language_macro_f1", "Language macro-F1"),
        ("dialect_language_macro_f1", "Dialect language-macro F1"),
        ("gender_macro_f1", "Gender macro-F1"),
        ("three_task_mean_macro_f1", "Three-task mean macro-F1"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, panel_metrics):
        for domain, label in (("one_bit", "1-bit log-Mel"), ("sixteen_bit", "16-bit log-Mel")):
            subset = matched[matched.domain == domain].set_index("duration").loc[labels]
            axis.plot(x, subset[metric].to_numpy() * 100.0, marker="o", label=label)
            reference_value = full_frontend[
                (full_frontend.domain == domain)
                & (full_frontend.front_end == "indicwav2vec2_layer11")
            ].iloc[0][metric]
            axis.scatter([x[-1]], [float(reference_value) * 100.0], marker="*", s=180, label=f"{domain.replace('_', '-')} IndicWav2Vec2 full")
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Maximum utterance duration")
        axis.set_ylabel("Score (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("EXP 6.9: duration sensitivity and full-length front-end comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_heatmap(config: Config, transfer: pd.DataFrame, domain: str, path: Path) -> None:
    labels = list(config.durations)
    subset = transfer[transfer.domain == domain]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row, train_duration in enumerate(labels):
        for column, test_duration in enumerate(labels):
            value = subset[
                (subset.train_duration == train_duration)
                & (subset.test_duration == test_duration)
            ].iloc[0].three_task_mean_macro_f1
            matrix[row, column] = float(value) * 100.0
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Test duration")
    axis.set_ylabel("Training duration")
    axis.set_title(f"{domain.replace('_', '-')} duration transfer: three-task mean macro-F1")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Score (%)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_gain_figure(gain: pd.DataFrame, path: Path) -> None:
    metrics = [
        "language_macro_f1",
        "dialect_language_macro_f1",
        "gender_macro_f1",
        "three_task_mean_macro_f1",
    ]
    labels = ["Language", "Dialect", "Gender", "Overall"]
    x = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for offset, domain in ((-width / 2, "one_bit"), (width / 2, "sixteen_bit")):
        subset = gain[(gain.domain == domain) & (gain.metric.isin(metrics))].set_index("metric").loc[metrics]
        axis.bar(x + offset, subset["percentage_point_gain"].to_numpy(), width, label=domain.replace("_", "-"))
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x, labels)
    axis.set_ylabel("IndicWav2Vec2 gain over log-Mel (percentage points)")
    axis.set_title("Full-utterance front-end gain")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_cross_figure(config: Config, cross: pd.DataFrame, path: Path) -> None:
    labels = list(config.durations)
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for domain in config.domains:
        subset = cross[cross.train_domain == domain].set_index("duration").loc[labels]
        axis.plot(x, subset["matched_three_task_mean_macro_f1"] * 100.0, marker="o", label=f"{domain.replace('_','-')} matched")
        axis.plot(x, subset["cross_three_task_mean_macro_f1"] * 100.0, marker="x", linestyle="--", label=f"{domain.replace('_','-')} cross")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Training and test duration cap")
    axis.set_ylabel("Three-task mean macro-F1 (%)")
    axis.set_title("Matched versus cross-precision robustness")
    axis.grid(True, alpha=0.3)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_runtime_figure(runtime: pd.DataFrame, path: Path) -> None:
    labels = [f"{row.domain}\n{row.duration}" for row in runtime.itertuples()]
    values = runtime["training_seconds"].to_numpy() / 3600.0
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.bar(np.arange(len(labels)), values)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylabel("Training time (hours)")
    axis.set_title("EXP 6.9 training runtime by condition")
    axis.grid(True, axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _save_confusion_composite(config: Config, domain: str, path: Path) -> None:
    payload = _load_json(_matched_metrics_path(config, domain, "full"))
    figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    axes = axes.flat
    matrices = [
        (np.asarray(payload["language"]["confusion_row_percent"]), "Language ID"),
        (np.asarray(payload["gender"]["confusion_row_percent"]), "Gender"),
    ]
    for language in config.dialect_class_counts:
        matrices.append(
            (
                np.asarray(payload["dialect"]["per_language"][language]["confusion_row_percent"]),
                f"Dialect: {language}",
            )
        )
    for axis, (matrix, title) in zip(axes, matrices):
        image = axis.imshow(matrix, vmin=0.0, vmax=100.0, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center", fontsize=7)
    for axis in list(axes)[len(matrices):]:
        axis.axis("off")
    figure.suptitle(f"{domain.replace('_','-')} full log-Mel matched confusion matrices (row %)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _analysis_text(
    config: Config,
    matched: pd.DataFrame,
    gain: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    lines = [
        "# EXP 6.9 — Automatically Generated Results Interpretation",
        "",
        "This document reports measured outputs only. It compares a waveform-derived 80-bin log-Mel front end against the completed full-utterance IndicWav2Vec2 layer-11 reference from EXP 6.1.",
        "",
    ]
    for domain in config.domains:
        subset = matched[matched.domain == domain].set_index("duration").loc[list(config.durations)]
        lines.append(f"## {domain.replace('_', '-')} duration trend")
        for metric, label in (
            ("language_macro_f1", "Language macro-F1"),
            ("dialect_language_macro_f1", "Dialect language-macro F1"),
            ("gender_macro_f1", "Gender macro-F1"),
            ("three_task_mean_macro_f1", "Three-task mean macro-F1"),
        ):
            values = subset[metric].to_numpy(dtype=float)
            best_index = int(np.argmax(values))
            deltas = np.diff(values) * 100.0
            lines.append(
                f"- **{label}:** best at `{config.durations[best_index]}` ({values[best_index]*100:.2f}%). "
                f"Changes 3→7, 7→9 and 9→full: {deltas[0]:+.2f}, {deltas[1]:+.2f}, {deltas[2]:+.2f} percentage points."
            )
        lines.append("")

    lines.append("## Full-length IndicWav2Vec2 versus log-Mel")
    for domain in config.domains:
        domain_gain = gain[gain.domain == domain].set_index("metric")
        for metric, label in (
            ("language_macro_f1", "Language macro-F1"),
            ("dialect_language_macro_f1", "Dialect language-macro F1"),
            ("gender_macro_f1", "Gender macro-F1"),
            ("three_task_mean_macro_f1", "Three-task mean macro-F1"),
        ):
            row = domain_gain.loc[metric]
            lines.append(
                f"- **{domain.replace('_','-')} {label}:** IndicWav2Vec2 minus log-Mel = "
                f"{row.percentage_point_gain:+.2f} percentage points."
            )
    lines.append("")
    lines.append("## Paired speaker-bootstrap evidence")
    for row in bootstrap.itertuples():
        lines.append(
            f"- `{row.domain}` / `{row.metric}`: mean difference {row.mean_difference_wav2vec_minus_logmel*100:+.2f} pp, "
            f"95% CI [{row.ci95_lower*100:+.2f}, {row.ci95_upper*100:+.2f}] pp, "
            f"positive in {row.fraction_positive*100:.1f}% of bootstrap samples."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A positive full-length gap supports the claim that the frozen pretrained IndicWav2Vec2 final-layer representation is more useful than this conventional log-Mel baseline for the same three-task ECAPA classifier. It does not isolate pretraining, contextualisation, the convolutional feature encoder, and the final transformer layer as separate causal factors.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(config: Config) -> dict[str, Any]:
    report_root = config.path("report_root")
    tables = report_root / "tables"
    figures = report_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    configuration = _configuration_frame(config)
    matched = _matched_frame(config)
    cross = _cross_frame(config)
    transfer = _duration_transfer_frame(config)
    full_frontend = _full_frontend_frame(config, matched)
    gain = _gain_frame(full_frontend)
    runtime = _history_runtime_frame(config)
    bootstrap = _bootstrap_frontend(config, full_frontend)

    outputs = {
        "table_01_experimental_configuration": tables / "table_01_experimental_configuration.csv",
        "table_02_matched_duration_results": tables / "table_02_matched_duration_results.csv",
        "table_03_full_frontend_comparison": tables / "table_03_full_frontend_comparison.csv",
        "table_04_indicwav2vec2_gain": tables / "table_04_indicwav2vec2_gain.csv",
        "table_05_cross_precision": tables / "table_05_cross_precision.csv",
        "table_06_duration_transfer": tables / "table_06_duration_transfer.csv",
        "table_07_runtime": tables / "table_07_runtime.csv",
        "table_08_paired_speaker_bootstrap": tables / "table_08_paired_speaker_bootstrap.csv",
    }
    for frame, key in (
        (configuration, "table_01_experimental_configuration"),
        (matched, "table_02_matched_duration_results"),
        (full_frontend, "table_03_full_frontend_comparison"),
        (gain, "table_04_indicwav2vec2_gain"),
        (cross, "table_05_cross_precision"),
        (transfer, "table_06_duration_transfer"),
        (runtime, "table_07_runtime"),
        (bootstrap, "table_08_paired_speaker_bootstrap"),
    ):
        frame.to_csv(outputs[key], index=False)

    figure_outputs = {
        "figure_01_duration_frontend_ablation": figures / "figure_01_duration_frontend_ablation.png",
        "figure_02_one_bit_duration_transfer": figures / "figure_02_one_bit_duration_transfer.png",
        "figure_03_sixteen_bit_duration_transfer": figures / "figure_03_sixteen_bit_duration_transfer.png",
        "figure_04_full_frontend_gain": figures / "figure_04_full_frontend_gain.png",
        "figure_05_cross_precision": figures / "figure_05_cross_precision.png",
        "figure_06_training_runtime": figures / "figure_06_training_runtime.png",
        "figure_07_one_bit_full_confusions": figures / "figure_07_one_bit_full_confusions.png",
        "figure_08_sixteen_bit_full_confusions": figures / "figure_08_sixteen_bit_full_confusions.png",
    }
    _save_main_figure(config, matched, full_frontend, figure_outputs["figure_01_duration_frontend_ablation"])
    _save_heatmap(config, transfer, "one_bit", figure_outputs["figure_02_one_bit_duration_transfer"])
    _save_heatmap(config, transfer, "sixteen_bit", figure_outputs["figure_03_sixteen_bit_duration_transfer"])
    _save_gain_figure(gain, figure_outputs["figure_04_full_frontend_gain"])
    _save_cross_figure(config, cross, figure_outputs["figure_05_cross_precision"])
    _save_runtime_figure(runtime, figure_outputs["figure_06_training_runtime"])
    _save_confusion_composite(config, "one_bit", figure_outputs["figure_07_one_bit_full_confusions"])
    _save_confusion_composite(config, "sixteen_bit", figure_outputs["figure_08_sixteen_bit_full_confusions"])

    analysis_path = report_root / "RESULTS_INTERPRETATION.md"
    atomic_text(analysis_path, _analysis_text(config, matched, gain, bootstrap))

    marker = {
        "status": "REPORT_COMPLETE",
        "created_utc": utc_now(),
        "tables": {key: str(path) for key, path in outputs.items()},
        "figures": {key: str(path) for key, path in figure_outputs.items()},
        "analysis": str(analysis_path),
        "table_count": len(outputs),
        "figure_count": len(figure_outputs),
    }
    atomic_json(report_root / "REPORT_COMPLETE.json", marker)
    print("EXP69_REPORT_COMPLETE", flush=True)
    return marker

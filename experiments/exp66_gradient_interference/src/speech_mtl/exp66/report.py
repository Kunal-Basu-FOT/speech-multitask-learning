from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import CONDITIONS, TASKS
from .utils import atomic_json_dump

DOMAINS = ("one_bit", "sixteen_bit")
TASK_METRICS = {
    "LID": "lid_macro_f1",
    "Dialect": "dialect_language_macro_f1",
    "Gender": "gender_macro_f1",
}
PAIR_LABELS = {
    ("lid", "dialect"): "LID–Dialect",
    ("lid", "gender"): "LID–Gender",
    ("dialect", "gender"): "Dialect–Gender",
}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _metric_path(cfg, train_domain: str, condition: str, test_domain: str) -> Path:
    return (
        cfg.paths.results_root
        / train_domain
        / condition
        / f"test_{test_domain}"
        / "metrics.json"
    )


def _load_domain_metrics(cfg, domain: str) -> dict[tuple[str, str], dict]:
    other = "sixteen_bit" if domain == "one_bit" else "one_bit"
    return {
        (condition, test_domain): _read_json(
            _metric_path(cfg, domain, condition, test_domain)
        )
        for condition in CONDITIONS
        for test_domain in (domain, other)
    }


def _aggregate_gradient_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_a"], row["task_b"])].append(row)
    result = []
    for pair, selected in grouped.items():
        weights = np.asarray([float(row["samples"]) for row in selected], dtype=float)
        if weights.sum() == 0:
            continue
        pre = np.asarray([float(row["mean_cosine_pre"]) for row in selected])
        conflict = np.asarray([float(row["conflict_rate_pre"]) for row in selected])
        norm_a = np.asarray([float(row["mean_norm_a"]) for row in selected])
        norm_b = np.asarray([float(row["mean_norm_b"]) for row in selected])
        post_rows = [row for row in selected if row.get("mean_cosine_post", "") != ""]
        aggregate = {
            "task_a": pair[0],
            "task_b": pair[1],
            "pair": PAIR_LABELS[pair],
            "samples": int(weights.sum()),
            "mean_cosine_pre": float(np.average(pre, weights=weights)),
            "conflict_rate_pre": float(np.average(conflict, weights=weights)),
            "mean_norm_a": float(np.average(norm_a, weights=weights)),
            "mean_norm_b": float(np.average(norm_b, weights=weights)),
            "mean_cosine_post": "",
            "conflict_rate_post": "",
        }
        if post_rows:
            post_weights = np.asarray([float(row["samples"]) for row in post_rows])
            aggregate["mean_cosine_post"] = float(
                np.average(
                    [float(row["mean_cosine_post"]) for row in post_rows],
                    weights=post_weights,
                )
            )
            aggregate["conflict_rate_post"] = float(
                np.average(
                    [float(row["conflict_rate_post"]) for row in post_rows],
                    weights=post_weights,
                )
            )
        result.append(aggregate)
    return result


def _plot_grouped_task_bars(metrics, domain: str, report_root: Path) -> None:
    task_names = list(TASK_METRICS)
    x = np.arange(len(task_names))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for index, condition in enumerate(CONDITIONS):
        values = [metrics[(condition, domain)][TASK_METRICS[task]] for task in task_names]
        axis.bar(x + (index - 1) * width, values, width, label=condition)
    axis.set_xticks(x, task_names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Macro-F1")
    axis.set_title(f"EXP 6.6 {domain}: matched task performance")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_01_matched_macro_f1_by_method.png", dpi=300)
    plt.close(fig)


def _plot_matched_cross(metrics, domain: str, report_root: Path) -> None:
    other = "sixteen_bit" if domain == "one_bit" else "one_bit"
    x = np.arange(len(CONDITIONS))
    width = 0.36
    matched = [metrics[(condition, domain)]["three_task_mean_macro_f1"] for condition in CONDITIONS]
    cross = [metrics[(condition, other)]["three_task_mean_macro_f1"] for condition in CONDITIONS]
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(x - width / 2, matched, width, label=f"matched test ({domain})")
    axis.bar(x + width / 2, cross, width, label=f"cross test ({other})")
    axis.set_xticks(x, CONDITIONS, rotation=15)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Three-task mean macro-F1")
    axis.set_title(f"EXP 6.6 {domain}: matched and cross-precision evaluation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_02_matched_vs_cross_precision.png", dpi=300)
    plt.close(fig)


def _plot_gradient_aggregate(aggregate_by_condition: dict, report_root: Path) -> None:
    pairs = list(PAIR_LABELS.values())
    x = np.arange(len(pairs))
    width = 0.24
    for metric_name, ylabel, filename, title in (
        (
            "conflict_rate_pre",
            "Fraction of sampled steps with cosine < 0",
            "figure_03_gradient_conflict_rate_by_method.png",
            "Pre-intervention gradient conflict rate",
        ),
        (
            "mean_cosine_pre",
            "Mean pairwise gradient cosine",
            "figure_04_mean_gradient_cosine_by_method.png",
            "Pre-intervention mean gradient alignment",
        ),
    ):
        fig, axis = plt.subplots(figsize=(8.5, 5.0))
        for condition_index, condition in enumerate(CONDITIONS):
            lookup = {row["pair"]: row for row in aggregate_by_condition[condition]}
            values = [float(lookup[pair][metric_name]) for pair in pairs]
            axis.bar(x + (condition_index - 1) * width, values, width, label=condition)
        axis.set_xticks(x, pairs)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if metric_name == "mean_cosine_pre":
            axis.axhline(0.0, linewidth=1)
        else:
            axis.set_ylim(0.0, 1.0)
        axis.legend()
        fig.tight_layout()
        fig.savefig(report_root / filename, dpi=300)
        plt.close(fig)


def _plot_gradient_epochs(condition: str, rows: list[dict], report_root: Path) -> None:
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for pair_key, pair_label in PAIR_LABELS.items():
        selected = [
            row for row in rows
            if row["task_a"] == pair_key[0] and row["task_b"] == pair_key[1]
        ]
        if not selected:
            continue
        selected.sort(key=lambda row: int(row["epoch"]))
        axis.plot(
            [int(row["epoch"]) for row in selected],
            [float(row["mean_cosine_pre"]) for row in selected],
            marker="o",
            markersize=2,
            label=pair_label,
        )
    axis.axhline(0.0, linewidth=1)
    axis.axvline(5.5, linestyle="--", linewidth=1)
    axis.axvline(35.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean sampled gradient cosine")
    axis.set_title(f"{condition}: pairwise gradient alignment over training")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / f"figure_gradient_epochs_{condition}.png", dpi=300)
    plt.close(fig)


def _plot_training_curves(cfg, domain: str, report_root: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for condition in CONDITIONS:
        rows = _read_csv(cfg.paths.run_root / domain / condition / "history.csv")
        if not rows:
            continue
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["selection_score"]) for row in rows],
            marker="o",
            markersize=2,
            label=condition,
        )
    axis.axvline(5.5, linestyle="--", linewidth=1)
    axis.axvline(35.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Development three-task mean macro-F1")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"EXP 6.6 {domain}: development selection curves")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_08_development_selection_curves.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for condition in CONDITIONS:
        rows = _read_csv(cfg.paths.run_root / domain / condition / "history.csv")
        if not rows:
            continue
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["train_loss"]) for row in rows],
            label=f"{condition} train",
        )
        axis.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["dev_loss"]) for row in rows],
            linestyle="--",
            label=f"{condition} dev",
        )
    axis.axvline(5.5, linestyle="--", linewidth=1)
    axis.axvline(35.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Objective loss")
    axis.set_title(f"EXP 6.6 {domain}: train and development objectives")
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(report_root / "figure_09_loss_curves.png", dpi=300)
    plt.close(fig)


def _plot_gradnorm(cfg, domain: str, report_root: Path) -> None:
    rows = _read_csv(cfg.paths.run_root / domain / "gradnorm" / "gradnorm_weights.csv")
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    epochs = [int(row["epoch"]) for row in rows]
    for task in TASKS:
        axis.plot(
            epochs,
            [float(row[f"{task}_weight"]) for row in rows],
            marker="o",
            markersize=2,
            label=task,
        )
    axis.axhline(1.0, linewidth=1)
    axis.axvline(5.5, linestyle="--", linewidth=1)
    axis.axvline(35.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized GradNorm task weight")
    axis.set_title(f"EXP 6.6 {domain}: GradNorm task weights")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_10_gradnorm_task_weights.png", dpi=300)
    plt.close(fig)


def _plot_pcgrad(cfg, domain: str, report_root: Path) -> None:
    rows = _read_csv(cfg.paths.run_root / domain / "pcgrad" / "history.csv")
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(
        [int(row["epoch"]) for row in rows],
        [int(float(row["pcgrad_projections_epoch"])) for row in rows],
    )
    axis.axvline(5.5, linestyle="--", linewidth=1)
    axis.axvline(35.5, linestyle="--", linewidth=1)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Pairwise projection operations")
    axis.set_title(f"EXP 6.6 {domain}: PCGrad projection count")
    fig.tight_layout()
    fig.savefig(report_root / "figure_11_pcgrad_projection_counts.png", dpi=300)
    plt.close(fig)


def generate_domain_report(cfg, domain: str) -> Path:
    if domain not in DOMAINS:
        raise ValueError(domain)
    other = "sixteen_bit" if domain == "one_bit" else "one_bit"
    report_root = cfg.paths.report_root / domain
    report_root.mkdir(parents=True, exist_ok=True)
    metrics = _load_domain_metrics(cfg, domain)

    summary_rows = []
    dialect_rows = []
    parameter_rows = []
    history_rows = []
    gradient_epoch_rows = []
    gradient_aggregate_rows = []
    aggregate_by_condition = {}
    for condition in CONDITIONS:
        for test_domain in (domain, other):
            values = metrics[(condition, test_domain)]
            summary_rows.append(
                {
                    "train_domain": domain,
                    "test_domain": test_domain,
                    "matched": test_domain == domain,
                    "condition": condition,
                    "lid_accuracy": values["lid_accuracy"],
                    "lid_macro_f1": values["lid_macro_f1"],
                    "dialect_sample_weighted_accuracy": values["dialect_sample_weighted_accuracy"],
                    "dialect_language_macro_accuracy": values["dialect_language_macro_accuracy"],
                    "dialect_language_macro_f1": values["dialect_language_macro_f1"],
                    "gender_accuracy": values["gender_accuracy"],
                    "gender_macro_f1": values["gender_macro_f1"],
                    "three_task_mean_macro_f1": values["three_task_mean_macro_f1"],
                    "best_epoch": values["best_epoch"],
                    "selection_score": values["selection_score"],
                }
            )
            for language, language_metrics in values["dialect_per_language"].items():
                dialect_rows.append(
                    {
                        "train_domain": domain,
                        "test_domain": test_domain,
                        "matched": test_domain == domain,
                        "condition": condition,
                        "language": language,
                        "accuracy": language_metrics["accuracy"],
                        "macro_f1": language_metrics["macro_f1"],
                        "support": sum(language_metrics["support"]),
                    }
                )
        counts = metrics[(condition, domain)]["parameter_counts"]
        parameter_rows.append({"condition": condition, **counts})
        condition_history = _read_csv(cfg.paths.run_root / domain / condition / "history.csv")
        for row in condition_history:
            history_rows.append({"condition": condition, **row})
        condition_gradients = _read_csv(
            cfg.paths.run_root / domain / condition / "gradient_epoch_summary.csv"
        )
        for row in condition_gradients:
            gradient_epoch_rows.append({"condition": condition, **row})
        aggregate = _aggregate_gradient_rows(condition_gradients)
        aggregate_by_condition[condition] = aggregate
        for row in aggregate:
            gradient_aggregate_rows.append({"condition": condition, **row})

    _write_csv(report_root / "table_01_matched_and_cross_precision.csv", summary_rows)
    _write_csv(report_root / "table_02_per_language_dialect.csv", dialect_rows)
    _write_csv(report_root / "table_03_gradient_conflict_aggregate.csv", gradient_aggregate_rows)
    _write_csv(report_root / "table_04_gradient_epoch_summary.csv", gradient_epoch_rows)
    _write_csv(report_root / "table_05_training_history.csv", history_rows)
    _write_csv(report_root / "table_06_parameter_counts.csv", parameter_rows)
    gradnorm_rows = _read_csv(cfg.paths.run_root / domain / "gradnorm" / "gradnorm_weights.csv")
    if gradnorm_rows:
        _write_csv(report_root / "table_07_gradnorm_weights.csv", gradnorm_rows)
    _write_csv(
        report_root / "table_08_method_configuration.csv",
        [
            {
                "condition": "standard_logged",
                "model": "hard-sharing ECAPA",
                "task_weighting": "learned homoscedastic uncertainty",
                "shared_gradient_update": "ordinary summed objective",
                "diagnostic_frequency": f"every {cfg.training.gradient_diagnostic_interval} batches",
            },
            {
                "condition": "pcgrad",
                "model": "hard-sharing ECAPA",
                "task_weighting": "learned homoscedastic uncertainty",
                "shared_gradient_update": "PCGrad on full shared encoder every optimizer step",
                "diagnostic_frequency": "every optimizer step",
            },
            {
                "condition": "gradnorm",
                "model": "hard-sharing ECAPA",
                "task_weighting": "GradNorm positive weights normalized to sum to three",
                "shared_gradient_update": "ordinary weighted sum",
                "diagnostic_frequency": f"every {cfg.training.gradient_diagnostic_interval} batches",
            },
        ],
    )

    _plot_grouped_task_bars(metrics, domain, report_root)
    _plot_matched_cross(metrics, domain, report_root)
    _plot_gradient_aggregate(aggregate_by_condition, report_root)
    for condition in CONDITIONS:
        _plot_gradient_epochs(
            condition,
            _read_csv(cfg.paths.run_root / domain / condition / "gradient_epoch_summary.csv"),
            report_root,
        )
    _plot_training_curves(cfg, domain, report_root)
    _plot_gradnorm(cfg, domain, report_root)
    _plot_pcgrad(cfg, domain, report_root)

    manifest = {
        "status": "complete",
        "domain": domain,
        "queue_semantics": (
            "This domain report was generated after all three methods completed "
            "training, matched evaluation, and cross-precision evaluation, and "
            "before the next domain was allowed to start."
        ),
        "conditions": list(CONDITIONS),
        "matched_test_domain": domain,
        "cross_test_domain": other,
        "tables": sorted(path.name for path in report_root.glob("table_*.csv")),
        "figures": sorted(path.name for path in report_root.glob("figure_*.png")),
    }
    atomic_json_dump(manifest, report_root / "DOMAIN_REPORT_COMPLETE.json")
    print(f"EXP66_DOMAIN_REPORT_PASS domain={domain}", flush=True)
    return report_root


def generate_final_report(cfg) -> Path:
    report_root = cfg.paths.report_root / "final"
    report_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for domain in DOMAINS:
        path = cfg.paths.report_root / domain / "table_01_matched_and_cross_precision.csv"
        rows = _read_csv(path)
        if not rows:
            raise FileNotFoundError(path)
        all_rows.extend(rows)
    _write_csv(report_root / "table_01_all_evaluation_cells.csv", all_rows)

    matched_lookup = {
        (row["train_domain"], row["condition"]): row
        for row in all_rows
        if row["matched"].lower() == "true"
    }
    gap_rows = []
    for condition in CONDITIONS:
        one = matched_lookup[("one_bit", condition)]
        sixteen = matched_lookup[("sixteen_bit", condition)]
        for task, metric in TASK_METRICS.items():
            gap_rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "one_bit_macro_f1": float(one[metric]),
                    "sixteen_bit_macro_f1": float(sixteen[metric]),
                    "sixteen_minus_one_bit": float(sixteen[metric]) - float(one[metric]),
                    "one_bit_retention_percent": 100.0 * float(one[metric]) / max(float(sixteen[metric]), 1e-12),
                }
            )
    _write_csv(report_root / "table_02_bit_depth_gaps.csv", gap_rows)

    x = np.arange(len(CONDITIONS))
    width = 0.36
    one_values = [float(matched_lookup[("one_bit", condition)]["three_task_mean_macro_f1"]) for condition in CONDITIONS]
    sixteen_values = [float(matched_lookup[("sixteen_bit", condition)]["three_task_mean_macro_f1"]) for condition in CONDITIONS]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - width / 2, one_values, width, label="1-bit matched")
    axis.bar(x + width / 2, sixteen_values, width, label="16-bit matched")
    axis.set_xticks(x, CONDITIONS, rotation=15)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Three-task mean macro-F1")
    axis.set_title("EXP 6.6 optimization methods across bit-depth endpoints")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_01_methods_across_bit_depths.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    task_names = list(TASK_METRICS)
    x = np.arange(len(task_names))
    width = 0.24
    for index, condition in enumerate(CONDITIONS):
        lookup = {
            row["task"]: float(row["sixteen_minus_one_bit"])
            for row in gap_rows
            if row["condition"] == condition
        }
        axis.bar(
            x + (index - 1) * width,
            [lookup[task] for task in task_names],
            width,
            label=condition,
        )
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x, task_names)
    axis.set_ylabel("16-bit minus 1-bit macro-F1")
    axis.set_title("EXP 6.6 task sensitivity to endpoint quantization")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_02_bit_depth_gap_by_task_and_method.png", dpi=300)
    plt.close(fig)

    atomic_json_dump(
        {
            "status": "complete",
            "domains": list(DOMAINS),
            "conditions": list(CONDITIONS),
            "tables": sorted(path.name for path in report_root.glob("table_*.csv")),
            "figures": sorted(path.name for path in report_root.glob("figure_*.png")),
        },
        report_root / "FINAL_REPORT_COMPLETE.json",
    )
    print("EXP66_FINAL_REPORT_PASS", flush=True)
    return report_root

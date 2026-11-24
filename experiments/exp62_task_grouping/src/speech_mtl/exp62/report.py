"""Aggregate reduced EXP 6.2 and prepare comparison figures with EXP 6.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .conditions import CONDITION_ORDER, DOMAINS
from .config import ExperimentConfig
from .utils import atomic_json_dump


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def collect_exp62(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_domain in DOMAINS:
        for condition in CONDITION_ORDER:
            for test_domain in DOMAINS:
                path = cfg.paths.results_root / train_domain / condition / f"test_{test_domain}" / "metrics.json"
                metrics = _read_json(path)
                parameter_counts = metrics.get("parameter_counts", {}) or {}
                rows.append(
                    {
                        "train_domain": train_domain,
                        "test_domain": test_domain,
                        "condition": condition,
                        "matched": train_domain == test_domain,
                        "lid_accuracy": metrics.get("lid_accuracy"),
                        "lid_macro_f1": metrics.get("lid_macro_f1"),
                        "dialect_sample_weighted_accuracy": metrics.get("dialect_sample_weighted_accuracy"),
                        "dialect_language_macro_accuracy": metrics.get("dialect_language_macro_accuracy"),
                        "dialect_language_macro_f1": metrics.get("dialect_language_macro_f1"),
                        "best_epoch": metrics.get("best_epoch"),
                        "selection_score": metrics.get("selection_score"),
                        "total_parameters": parameter_counts.get("total"),
                        "trainable_parameters_at_checkpoint": parameter_counts.get("trainable"),
                        "metrics_path": str(path),
                    }
                )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matched_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["train_domain"], r["condition"]): r for r in rows if r["matched"]}


def compute_affinity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _matched_lookup(rows)
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        lid_only = lookup[(domain, "lid_only")]
        dialect_only = lookup[(domain, "dialect_only")]
        joint = lookup[(domain, "lid_dialect")]
        for metric in ("accuracy", "macro_f1"):
            dialect_key = (
                "dialect_language_macro_accuracy"
                if metric == "accuracy"
                else "dialect_language_macro_f1"
            )
            output.extend(
                [
                    {
                        "domain": domain,
                        "metric": metric,
                        "direction": "A(LID | Dialect)",
                        "affinity": float(joint[f"lid_{metric}"]) - float(lid_only[f"lid_{metric}"]),
                    },
                    {
                        "domain": domain,
                        "metric": metric,
                        "direction": "A(Dialect | LID)",
                        "affinity": float(joint[dialect_key])
                        - float(dialect_only[dialect_key]),
                    },
                ]
            )
    return output


def plot_affinity(rows: list[dict[str, Any]], metric: str, path: Path, dpi: int) -> None:
    selected = [x for x in rows if x["metric"] == metric]
    labels = [f"{x['domain']}\n{x['direction']}" for x in selected]
    values = [x["affinity"] for x in selected]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(values)), values)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylabel(f"Joint − single ({metric.replace('_', ' ')})")
    ax.set_title("Directional LID–dialect task affinity")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_cross_change(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["train_domain"], r["condition"], r["test_domain"]): r for r in rows}
    output: list[dict[str, Any]] = []
    metrics = (
        "lid_accuracy", "lid_macro_f1",
        "dialect_language_macro_accuracy", "dialect_language_macro_f1",
    )
    for train_domain in DOMAINS:
        other = "sixteen_bit" if train_domain == "one_bit" else "one_bit"
        for condition in CONDITION_ORDER:
            matched = by_key[(train_domain, condition, train_domain)]
            cross = by_key[(train_domain, condition, other)]
            for metric in metrics:
                if matched.get(metric) is None:
                    continue
                output.append(
                    {
                        "train_domain": train_domain,
                        "test_direction": f"{train_domain}->{other}",
                        "condition": condition,
                        "metric": metric,
                        "matched_value": matched[metric],
                        "cross_value": cross[metric],
                        "cross_minus_matched": float(cross[metric]) - float(matched[metric]),
                    }
                )
    return output


def plot_cross_change(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    labels = [f"{r['test_direction']}\n{r['condition']}\n{r['metric'].replace('_macro','').replace('_accuracy',' acc').replace('_f1',' F1')}" for r in rows]
    values = [r["cross_minus_matched"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 0.75), 6))
    bars = ax.bar(range(len(values)), values)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(range(len(values)), labels, rotation=55, ha="right")
    ax.set_ylabel("Cross − matched")
    ax.set_title("Cross-precision performance change")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _normalize_exp61(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        if domain not in payload:
            raise ValueError(f"EXP 6.1 summary missing domain {domain}")
        source = payload[domain]
        required = {
            "lid_accuracy", "lid_macro_f1",
            "dialect_language_macro_accuracy", "dialect_language_macro_f1",
        }
        missing = required - set(source)
        if missing:
            raise ValueError(f"EXP 6.1 {domain} summary missing {sorted(missing)}")
        result[domain] = {key: float(source[key]) for key in required}
    return result


def plot_task_group_comparison(
    rows: list[dict[str, Any]],
    exp61: dict[str, dict[str, float]],
    domain: str,
    metric: str,
    path: Path,
    dpi: int,
) -> None:
    lookup = _matched_lookup(rows)
    suffix = "accuracy" if metric == "accuracy" else "macro_f1"
    dialect_key = (
        "dialect_language_macro_accuracy"
        if metric == "accuracy"
        else "dialect_language_macro_f1"
    )
    lid_values = [
        float(lookup[(domain, "lid_only")][f"lid_{suffix}"]),
        float(lookup[(domain, "lid_dialect")][f"lid_{suffix}"]),
        exp61[domain][f"lid_{suffix}"],
    ]
    dialect_values = [
        float(lookup[(domain, "dialect_only")][dialect_key]),
        float(lookup[(domain, "lid_dialect")][dialect_key]),
        exp61[domain][dialect_key],
    ]
    names = ["Single task", "LID + Dialect", "LID + Dialect + Gender\n(EXP 6.1)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, values, title in zip(axes, (lid_values, dialect_values), ("Language ID", "Dialect ID (language-macro)"), strict=True):
        bars = ax.bar(range(3), values)
        ax.set_xticks(range(3), names, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom")
    fig.suptitle(f"{domain}: task-group comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_report(cfg: ExperimentConfig, exp61_summary: Path | None = None) -> Path:
    report_root = cfg.paths.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    rows = collect_exp62(cfg)
    matched_parameter_rows = []
    seen_parameter_keys = set()
    for row in rows:
        if not row["matched"]:
            continue
        key = (row["train_domain"], row["condition"])
        if key in seen_parameter_keys:
            continue
        seen_parameter_keys.add(key)
        matched_parameter_rows.append(
            {
                "domain": row["train_domain"],
                "condition": row["condition"],
                "total_parameters": row["total_parameters"],
                "trainable_parameters_at_best_checkpoint": row["trainable_parameters_at_checkpoint"],
            }
        )
    _write_csv(matched_parameter_rows, report_root / "table_00_condition_parameter_counts.csv")
    _write_csv(rows, report_root / "table_01_all_matched_and_cross_metrics.csv")
    affinity = compute_affinity(rows)
    _write_csv(affinity, report_root / "table_02_directional_affinity.csv")
    plot_affinity(affinity, "macro_f1", report_root / "figure_01_affinity_macro_f1.png", cfg.evaluation.figure_dpi)
    plot_affinity(affinity, "accuracy", report_root / "figure_02_affinity_accuracy.png", cfg.evaluation.figure_dpi)
    cross = compute_cross_change(rows)
    _write_csv(cross, report_root / "table_03_cross_precision_change.csv")
    plot_cross_change(cross, report_root / "figure_03_cross_precision_change.png", cfg.evaluation.figure_dpi)

    summary_path = exp61_summary or (report_root / "exp61_matched_summary.json")
    if not summary_path.is_file():
        atomic_json_dump(
            {
                "status": "waiting_for_exp61",
                "required_file": str(summary_path),
                "template": {
                    domain: {
                        "lid_accuracy": None,
                        "lid_macro_f1": None,
                        "dialect_language_macro_accuracy": None,
                        "dialect_language_macro_f1": None,
                    }
                    for domain in DOMAINS
                },
                "note": "Fill only from completed matched EXP 6.1 test metrics; do not invent values.",
                "timestamp": time.time(),
            },
            report_root / "REPORT_WAITING_FOR_EXP61.json",
        )
        return report_root
    exp61 = _normalize_exp61(_read_json(summary_path))
    for domain in DOMAINS:
        for metric in ("accuracy", "macro_f1"):
            plot_task_group_comparison(
                rows,
                exp61,
                domain,
                metric,
                report_root / f"figure_task_group_{domain}_{metric}.png",
                cfg.evaluation.figure_dpi,
            )
    waiting = report_root / "REPORT_WAITING_FOR_EXP61.json"
    if waiting.exists():
        waiting.unlink()
    atomic_json_dump(
        {"status": "complete", "exp61_summary": str(summary_path), "timestamp": time.time()},
        report_root / "REPORT_COMPLETE.json",
    )
    return report_root

from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import CONDITION_NAME
from .utils import atomic_json_dump

DOMAINS = ("one_bit", "sixteen_bit")
TASK_METRICS = {
    "LID": "lid_macro_f1",
    "Dialect": "dialect_language_macro_f1",
    "Gender": "gender_macro_f1",
}


def _load_metrics(cfg, train_domain: str, test_domain: str) -> dict:
    path = (
        cfg.paths.results_root
        / train_domain
        / CONDITION_NAME
        / f"test_{test_domain}"
        / "metrics.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(cfg) -> Path:
    report_root = cfg.paths.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    metrics = {
        (train_domain, test_domain): _load_metrics(
            cfg, train_domain, test_domain
        )
        for train_domain in DOMAINS
        for test_domain in DOMAINS
    }

    summary_rows = []
    for (train_domain, test_domain), values in metrics.items():
        summary_rows.append(
            {
                "train_domain": train_domain,
                "test_domain": test_domain,
                "matched": train_domain == test_domain,
                "lid_accuracy": values["lid_accuracy"],
                "lid_macro_f1": values["lid_macro_f1"],
                "dialect_sample_weighted_accuracy": values[
                    "dialect_sample_weighted_accuracy"
                ],
                "dialect_language_macro_accuracy": values[
                    "dialect_language_macro_accuracy"
                ],
                "dialect_language_macro_f1": values[
                    "dialect_language_macro_f1"
                ],
                "gender_accuracy": values["gender_accuracy"],
                "gender_macro_f1": values["gender_macro_f1"],
                "three_task_mean_macro_f1": values[
                    "three_task_mean_macro_f1"
                ],
            }
        )
    _write_csv(
        report_root / "table_01_matched_and_cross_precision.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    dialect_rows = []
    for (train_domain, test_domain), values in metrics.items():
        for language, language_metrics in values["dialect_per_language"].items():
            dialect_rows.append(
                {
                    "train_domain": train_domain,
                    "test_domain": test_domain,
                    "language": language,
                    "accuracy": language_metrics["accuracy"],
                    "macro_f1": language_metrics["macro_f1"],
                    "support": sum(language_metrics["support"]),
                }
            )
    _write_csv(
        report_root / "table_02_per_language_dialect.csv",
        dialect_rows,
        list(dialect_rows[0]),
    )

    parameter_counts = metrics[("one_bit", "one_bit")]["parameter_counts"]
    _write_csv(
        report_root / "table_03_parameter_count.csv",
        [parameter_counts],
        list(parameter_counts),
    )

    # Matched 1-bit versus 16-bit task comparison.
    task_names = list(TASK_METRICS)
    one_values = [
        metrics[("one_bit", "one_bit")][TASK_METRICS[task]]
        for task in task_names
    ]
    sixteen_values = [
        metrics[("sixteen_bit", "sixteen_bit")][TASK_METRICS[task]]
        for task in task_names
    ]
    x = np.arange(len(task_names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(x - width / 2, one_values, width, label="1-bit matched")
    axis.bar(x + width / 2, sixteen_values, width, label="16-bit matched")
    axis.set_xticks(x, task_names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Macro-F1")
    axis.set_title("EXP 6.7 hierarchical matched performance")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_01_matched_1bit_vs_16bit.png", dpi=300)
    plt.close(fig)

    # Cross-minus-matched mismatch effect for both directions.
    one_to_sixteen = [
        metrics[("one_bit", "sixteen_bit")][TASK_METRICS[task]]
        - metrics[("one_bit", "one_bit")][TASK_METRICS[task]]
        for task in task_names
    ]
    sixteen_to_one = [
        metrics[("sixteen_bit", "one_bit")][TASK_METRICS[task]]
        - metrics[("sixteen_bit", "sixteen_bit")][TASK_METRICS[task]]
        for task in task_names
    ]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.bar(x - width / 2, one_to_sixteen, width, label="1→16")
    axis.bar(x + width / 2, sixteen_to_one, width, label="16→1")
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x, task_names)
    axis.set_ylabel("Cross minus matched macro-F1")
    axis.set_title("EXP 6.7 cross-precision mismatch")
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_root / "figure_02_cross_precision_delta.png", dpi=300)
    plt.close(fig)

    manifest = {
        "status": "complete",
        "condition": CONDITION_NAME,
        "tables": [
            "table_01_matched_and_cross_precision.csv",
            "table_02_per_language_dialect.csv",
            "table_03_parameter_count.csv",
        ],
        "figures": [
            "figure_01_matched_1bit_vs_16bit.png",
            "figure_02_cross_precision_delta.png",
        ],
        "comparison_with_exp61": "deferred_until_exp61_results_are_imported",
    }
    atomic_json_dump(manifest, report_root / "REPORT_COMPLETE.json")
    print("EXP67_REPORT_PASS", flush=True)
    return report_root

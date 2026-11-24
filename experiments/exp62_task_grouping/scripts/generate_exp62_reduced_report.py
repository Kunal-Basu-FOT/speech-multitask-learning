from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(
    "/data/Kunal/FORMAL_EXP6_2_TaskGrouping_IndicWav2Vec_FinalLayer"
)
RUN = ROOT / "runtime/runs/exp62_reduced"
RESULTS = ROOT / "results/exp62_reduced"
REPORT = ROOT / "reports/exp62_reduced"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
APPENDIX = REPORT / "appendix"

DOMAINS = ("one_bit", "sixteen_bit")
CONDITIONS = ("lid_only", "dialect_only", "lid_dialect")
TEST_DOMAINS = ("one_bit", "sixteen_bit")
LANGUAGES = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")

ACTIVE_TASKS = {
    "lid_only": ("language",),
    "dialect_only": ("dialect",),
    "lid_dialect": ("language", "dialect"),
}

CONDITION_LABELS = {
    "lid_only": "LID only",
    "dialect_only": "Dialect only",
    "lid_dialect": "LID + dialect",
}

DOMAIN_LABELS = {
    "one_bit": "1-bit",
    "sixteen_bit": "16-bit",
}

METRIC_ALIASES = {
    "language_accuracy": (
        "language_accuracy",
        "lid_accuracy",
        "language_metrics_accuracy",
        "lid_metrics_accuracy",
    ),
    "language_macro_f1": (
        "language_macro_f1",
        "lid_macro_f1",
        "language_f1_macro",
        "lid_f1_macro",
        "language_metrics_macro_f1",
        "lid_metrics_macro_f1",
    ),
    "dialect_language_macro_accuracy": (
        "dialect_language_macro_accuracy",
        "dialect_language_macro_acc",
        "language_macro_dialect_accuracy",
        "dialect_macro_accuracy",
    ),
    "dialect_language_macro_f1": (
        "dialect_language_macro_f1",
        "language_macro_dialect_f1",
        "dialect_macro_f1",
        "dialect_f1_language_macro",
    ),
    "dialect_sample_weighted_accuracy": (
        "dialect_sample_weighted_accuracy",
        "dialect_weighted_accuracy",
        "dialect_overall_accuracy",
    ),
}

FALLBACK_REQUIREMENTS = {
    "language_accuracy": (
        {"language", "accuracy"},
        {"lid", "accuracy"},
    ),
    "language_macro_f1": (
        {"language", "macro", "f1"},
        {"lid", "macro", "f1"},
    ),
    "dialect_language_macro_accuracy": (
        {"dialect", "language", "macro", "accuracy"},
    ),
    "dialect_language_macro_f1": (
        {"dialect", "language", "macro", "f1"},
    ),
    "dialect_sample_weighted_accuracy": (
        {"dialect", "sample", "weighted", "accuracy"},
        {"dialect", "weighted", "accuracy"},
    ),
}

FORBIDDEN = {
    "language_accuracy": {"dialect", "gender", "per_class"},
    "language_macro_f1": {"dialect", "gender", "per_class"},
    "dialect_language_macro_accuracy": {"per_class", "confusion"},
    "dialect_language_macro_f1": {"per_class", "confusion"},
    "dialect_sample_weighted_accuracy": {"per_class", "confusion"},
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def tokenise(value: str) -> set[str]:
    return set(normalise_key(value).split("_"))


def flatten_numbers(
    value: Any,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(flatten_numbers(child, prefix + (str(key),)))
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        original = ".".join(prefix)
        rows.append((normalise_key(original), original, float(value)))

    return rows


def score_to_fraction(value: float) -> float:
    number = float(value)

    if number < 0:
        return number

    if number <= 1.000001:
        return number

    if number <= 100.000001:
        return number / 100.0

    raise ValueError(f"Metric is outside a plausible score range: {number}")


def extract_metric(
    payload: Any,
    metric_name: str,
    required: bool,
) -> float | None:
    entries = flatten_numbers(payload)
    aliases = tuple(normalise_key(v) for v in METRIC_ALIASES[metric_name])

    # Exact and path-suffix matches are preferred.
    for alias in aliases:
        exact = [
            value
            for key, _, value in entries
            if key == alias
        ]
        if exact:
            return score_to_fraction(exact[0])

        suffix = [
            value
            for key, _, value in entries
            if key.endswith("_" + alias)
        ]
        if suffix:
            return score_to_fraction(suffix[0])

    # Carefully constrained fallback for small schema differences.
    candidates: list[tuple[int, int, float]] = []
    forbidden = FORBIDDEN[metric_name]

    for key, original, value in entries:
        tokens = tokenise(key)

        if tokens.intersection(forbidden):
            continue

        for requirement in FALLBACK_REQUIREMENTS[metric_name]:
            if requirement.issubset(tokens):
                score = 10 * len(requirement)

                if "metrics" in tokens:
                    score += 2
                if original.lower().endswith("accuracy"):
                    score += 1
                if original.lower().endswith("macro_f1"):
                    score += 1

                candidates.append((score, -len(tokens), value))

    if candidates:
        candidates.sort(reverse=True)
        return score_to_fraction(candidates[0][2])

    if required:
        raise KeyError(
            f"Could not locate required metric {metric_name!r}. "
            f"Available numeric paths include:\n"
            + "\n".join(f"  {path}" for _, path, _ in entries[:100])
        )

    return None


def select_model_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in (
            "model_state",
            "model_state_dict",
            "state_dict",
            "model",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate

        if payload and all(isinstance(key, str) for key in payload):
            tensor_values = sum(torch.is_tensor(v) for v in payload.values())
            if tensor_values:
                return payload

    raise TypeError("Could not identify a model state dictionary.")


def checkpoint_statistics(path: Path) -> tuple[int, int, int]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state = select_model_state(payload)

    tensors = 0
    parameters = 0
    nonfinite = 0

    for value in state.values():
        if not torch.is_tensor(value):
            continue

        tensors += 1
        parameters += int(value.numel())

        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                nonfinite += 1

    if tensors == 0:
        raise RuntimeError(f"No model tensors found in {path}")

    if nonfinite:
        raise RuntimeError(
            f"{nonfinite} model tensors contain non-finite values in {path}"
        )

    return parameters, tensors, path.stat().st_size


def history_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)

    epoch_column = None
    for candidate in ("global_epoch", "epoch"):
        if candidate in frame.columns:
            epoch_column = candidate
            break

    maximum_epoch = (
        int(pd.to_numeric(frame[epoch_column], errors="coerce").max())
        if epoch_column is not None
        else None
    )

    runtime_seconds = None
    for candidate in (
        "epoch_seconds",
        "runtime_seconds",
        "duration_seconds",
        "elapsed_seconds",
    ):
        if candidate in frame.columns:
            values = pd.to_numeric(frame[candidate], errors="coerce")
            runtime_seconds = float(values.sum())
            break

    amp_skips = None
    for candidate in (
        "amp_overflow_skips",
        "amp_skips",
        "overflow_skips",
    ):
        if candidate in frame.columns:
            values = pd.to_numeric(frame[candidate], errors="coerce")
            amp_skips = int(values.fillna(0).sum())
            break

    return {
        "history_rows": int(len(frame)),
        "maximum_epoch": maximum_epoch,
        "recorded_training_seconds": runtime_seconds,
        "recorded_amp_skips": amp_skips,
    }


def recursively_find_language_nodes(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[str, tuple[str, ...], dict[str, Any]]]:
    if not isinstance(value, dict):
        return

    for key, child in value.items():
        key_text = str(key).lower()

        if (
            key_text in LANGUAGES
            and isinstance(child, dict)
            and any("dialect" in segment.lower() for segment in path)
        ):
            yield key_text, path + (str(key),), child

        if isinstance(child, dict):
            yield from recursively_find_language_nodes(
                child,
                path + (str(key),),
            )


def extract_per_language_dialect(
    payload: Any,
) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    for language, path, node in recursively_find_language_nodes(payload):
        flat = flatten_numbers(node)

        accuracy = None
        macro_f1 = None

        for key, _, value in flat:
            tokens = tokenise(key)

            if (
                accuracy is None
                and "accuracy" in tokens
                and "per_class" not in tokens
            ):
                accuracy = score_to_fraction(value)

            if (
                macro_f1 is None
                and {"macro", "f1"}.issubset(tokens)
                and "per_class" not in tokens
            ):
                macro_f1 = score_to_fraction(value)

        if accuracy is not None or macro_f1 is not None:
            found[language] = {
                "language": language,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "json_path": ".".join(path),
            }

    return [found[language] for language in LANGUAGES if language in found]


def read_best_dev_metadata(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    entries = flatten_numbers(payload)

    best_epoch = None
    best_metric = None

    for key, _, value in entries:
        tokens = tokenise(key)

        if best_epoch is None and {"best", "epoch"}.issubset(tokens):
            best_epoch = int(value)

        if (
            best_metric is None
            and "best" in tokens
            and (
                "metric" in tokens
                or "score" in tokens
                or "selection" in tokens
            )
        ):
            try:
                best_metric = score_to_fraction(value)
            except ValueError:
                pass

    return {
        "best_epoch": best_epoch,
        "best_dev_selection_score": best_metric,
    }


def metric_row(
    train_domain: str,
    condition: str,
    test_domain: str,
) -> tuple[dict[str, Any], Any]:
    metrics_path = (
        RESULTS
        / train_domain
        / condition
        / f"test_{test_domain}"
        / "metrics.json"
    )
    payload = load_json(metrics_path)
    active = ACTIVE_TASKS[condition]

    language_required = "language" in active
    dialect_required = "dialect" in active

    language_accuracy = extract_metric(
        payload,
        "language_accuracy",
        required=language_required,
    )
    language_macro_f1 = extract_metric(
        payload,
        "language_macro_f1",
        required=language_required,
    )
    dialect_accuracy = extract_metric(
        payload,
        "dialect_language_macro_accuracy",
        required=dialect_required,
    )
    dialect_macro_f1 = extract_metric(
        payload,
        "dialect_language_macro_f1",
        required=dialect_required,
    )
    dialect_weighted_accuracy = extract_metric(
        payload,
        "dialect_sample_weighted_accuracy",
        required=False,
    )

    active_macro_f1 = [
        value
        for task, value in (
            ("language", language_macro_f1),
            ("dialect", dialect_macro_f1),
        )
        if task in active and value is not None
    ]

    active_accuracy = [
        value
        for task, value in (
            ("language", language_accuracy),
            ("dialect", dialect_accuracy),
        )
        if task in active and value is not None
    ]

    if len(active_macro_f1) != len(active):
        raise RuntimeError(
            f"Missing active macro-F1 metric for "
            f"{train_domain}/{condition}/test_{test_domain}"
        )

    if len(active_accuracy) != len(active):
        raise RuntimeError(
            f"Missing active accuracy metric for "
            f"{train_domain}/{condition}/test_{test_domain}"
        )

    row = {
        "train_domain": train_domain,
        "train_domain_label": DOMAIN_LABELS[train_domain],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "active_tasks": "+".join(active),
        "test_domain": test_domain,
        "test_domain_label": DOMAIN_LABELS[test_domain],
        "matched": train_domain == test_domain,
        "language_accuracy": language_accuracy,
        "language_macro_f1": language_macro_f1,
        "dialect_language_macro_accuracy": dialect_accuracy,
        "dialect_language_macro_f1": dialect_macro_f1,
        "dialect_sample_weighted_accuracy": dialect_weighted_accuracy,
        "condition_selection_macro_f1": float(np.mean(active_macro_f1)),
        "condition_mean_accuracy": float(np.mean(active_accuracy)),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
    }

    return row, payload


def percentage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    metric_columns = (
        "language_accuracy",
        "language_macro_f1",
        "dialect_language_macro_accuracy",
        "dialect_language_macro_f1",
        "dialect_sample_weighted_accuracy",
        "condition_selection_macro_f1",
        "condition_mean_accuracy",
    )

    for column in metric_columns:
        if column in result.columns:
            result[column + "_percent"] = (
                pd.to_numeric(result[column], errors="coerce") * 100.0
            )

    return result


def save_table(frame: pd.DataFrame, filename: str) -> Path:
    path = TABLES / filename
    frame.to_csv(path, index=False, float_format="%.8f")
    return path


def matched_lookup(
    matched: pd.DataFrame,
    domain: str,
    condition: str,
    metric: str,
) -> float:
    rows = matched[
        (matched["train_domain"] == domain)
        & (matched["condition"] == condition)
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one matched row for {domain}/{condition}; "
            f"found {len(rows)}"
        )

    value = rows.iloc[0][metric]

    if pd.isna(value):
        raise RuntimeError(
            f"Required matched metric {metric} is missing for "
            f"{domain}/{condition}"
        )

    return float(value)


def build_affinity_tables(
    matched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []

    definitions = (
        (
            "language",
            "dialect",
            "lid_only",
            "lid_dialect",
            "language_macro_f1",
            "language_accuracy",
        ),
        (
            "dialect",
            "language",
            "dialect_only",
            "lid_dialect",
            "dialect_language_macro_f1",
            "dialect_language_macro_accuracy",
        ),
    )

    for domain in DOMAINS:
        for (
            target,
            partner,
            baseline_condition,
            joint_condition,
            macro_metric,
            accuracy_metric,
        ) in definitions:
            baseline_macro = matched_lookup(
                matched,
                domain,
                baseline_condition,
                macro_metric,
            )
            joint_macro = matched_lookup(
                matched,
                domain,
                joint_condition,
                macro_metric,
            )
            baseline_accuracy = matched_lookup(
                matched,
                domain,
                baseline_condition,
                accuracy_metric,
            )
            joint_accuracy = matched_lookup(
                matched,
                domain,
                joint_condition,
                accuracy_metric,
            )

            macro_rows.append(
                {
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "target_task": target,
                    "partner_task": partner,
                    "baseline_condition": baseline_condition,
                    "joint_condition": joint_condition,
                    "single_task_macro_f1_percent": baseline_macro * 100.0,
                    "joint_macro_f1_percent": joint_macro * 100.0,
                    "directional_affinity_pp": (
                        joint_macro - baseline_macro
                    ) * 100.0,
                }
            )

            accuracy_rows.append(
                {
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "target_task": target,
                    "partner_task": partner,
                    "baseline_condition": baseline_condition,
                    "joint_condition": joint_condition,
                    "single_task_accuracy_percent": baseline_accuracy * 100.0,
                    "joint_accuracy_percent": joint_accuracy * 100.0,
                    "directional_affinity_pp": (
                        joint_accuracy - baseline_accuracy
                    ) * 100.0,
                }
            )

    return pd.DataFrame(macro_rows), pd.DataFrame(accuracy_rows)


def plot_matched_macro_f1(
    matched: pd.DataFrame,
    domain: str,
    output: Path,
) -> None:
    language_single = matched_lookup(
        matched,
        domain,
        "lid_only",
        "language_macro_f1",
    ) * 100.0
    language_joint = matched_lookup(
        matched,
        domain,
        "lid_dialect",
        "language_macro_f1",
    ) * 100.0
    dialect_single = matched_lookup(
        matched,
        domain,
        "dialect_only",
        "dialect_language_macro_f1",
    ) * 100.0
    dialect_joint = matched_lookup(
        matched,
        domain,
        "lid_dialect",
        "dialect_language_macro_f1",
    ) * 100.0

    tasks = ("Language ID", "Dialect ID")
    single = np.array((language_single, dialect_single))
    joint = np.array((language_joint, dialect_joint))
    x = np.arange(len(tasks))
    width = 0.36

    plt.figure(figsize=(8.5, 5.5))
    plt.bar(x - width / 2, single, width, label="Single-task")
    plt.bar(x + width / 2, joint, width, label="Joint LID + dialect")
    plt.xticks(x, tasks)
    plt.ylabel("Matched test macro-F1 (%)")
    plt.title(
        f"EXP 6.2 matched task-grouping performance — "
        f"{DOMAIN_LABELS[domain]}"
    )
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def affinity_matrix(
    affinity: pd.DataFrame,
    domain: str,
) -> np.ndarray:
    matrix = np.full((2, 2), np.nan, dtype=float)

    for _, row in affinity[affinity["domain"] == domain].iterrows():
        if (
            row["target_task"] == "language"
            and row["partner_task"] == "dialect"
        ):
            matrix[0, 1] = row["directional_affinity_pp"]
        elif (
            row["target_task"] == "dialect"
            and row["partner_task"] == "language"
        ):
            matrix[1, 0] = row["directional_affinity_pp"]

    return matrix


def plot_affinity_heatmap(
    affinity: pd.DataFrame,
    domain: str,
    output: Path,
) -> None:
    matrix = affinity_matrix(affinity, domain)
    masked = np.ma.masked_invalid(matrix)

    plt.figure(figsize=(6.5, 5.5))
    image = plt.imshow(masked)
    plt.colorbar(image, label="Macro-F1 affinity (percentage points)")
    plt.xticks((0, 1), ("Language", "Dialect"))
    plt.yticks((0, 1), ("Language", "Dialect"))
    plt.xlabel("Partner task")
    plt.ylabel("Target task")
    plt.title(
        f"Directional task affinity — {DOMAIN_LABELS[domain]}\n"
        "A(target | partner)"
    )

    for row in range(2):
        for column in range(2):
            if math.isfinite(matrix[row, column]):
                plt.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.2f}",
                    ha="center",
                    va="center",
                )
            else:
                plt.text(
                    column,
                    row,
                    "—",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cross_precision(
    all_results: pd.DataFrame,
    output: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for condition in CONDITIONS:
            matched_row = all_results[
                (all_results["train_domain"] == domain)
                & (all_results["condition"] == condition)
                & (all_results["test_domain"] == domain)
            ].iloc[0]

            opposite = (
                "sixteen_bit"
                if domain == "one_bit"
                else "one_bit"
            )
            cross_row = all_results[
                (all_results["train_domain"] == domain)
                & (all_results["condition"] == condition)
                & (all_results["test_domain"] == opposite)
            ].iloc[0]

            rows.append(
                {
                    "label": (
                        f"{DOMAIN_LABELS[domain]}\n"
                        f"{CONDITION_LABELS[condition]}"
                    ),
                    "matched": (
                        matched_row["condition_selection_macro_f1"] * 100.0
                    ),
                    "cross": (
                        cross_row["condition_selection_macro_f1"] * 100.0
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.36

    plt.figure(figsize=(12, 6.5))
    plt.bar(
        x - width / 2,
        frame["matched"],
        width,
        label="Matched test precision",
    )
    plt.bar(
        x + width / 2,
        frame["cross"],
        width,
        label="Opposite test precision",
    )
    plt.xticks(x, frame["label"], rotation=20, ha="right")
    plt.ylabel("Condition-specific mean macro-F1 (%)")
    plt.title("EXP 6.2 matched versus cross-precision transfer")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def write_results_readme(
    matched: pd.DataFrame,
    macro_affinity: pd.DataFrame,
    figure_paths: list[Path],
    table_paths: list[Path],
) -> Path:
    path = REPORT / "README_RESULTS.md"

    lines = [
        "# EXP 6.2 — Reduced LID–Dialect Task Grouping Report",
        "",
        "## Scope",
        "",
        "This is the reduced cache-compatible EXP 6.2 study.",
        "",
        "Conditions:",
        "",
        "- `lid_only`",
        "- `dialect_only`",
        "- `lid_dialect`",
        "",
        "Domains:",
        "",
        "- `one_bit`",
        "- `sixteen_bit`",
        "",
        "Gender, domain ID, and accent classification are not part of "
        "this experiment.",
        "",
        "## Directional macro-F1 affinity",
        "",
        "Affinity is defined as:",
        "",
        "`A(target | partner) = joint metric − single-task metric`",
        "",
        "| Domain | Target | Partner | Single-task macro-F1 | "
        "Joint macro-F1 | Affinity |",
        "|---|---|---|---:|---:|---:|",
    ]

    for _, row in macro_affinity.iterrows():
        lines.append(
            f"| {row['domain_label']} | {row['target_task']} | "
            f"{row['partner_task']} | "
            f"{row['single_task_macro_f1_percent']:.4f}% | "
            f"{row['joint_macro_f1_percent']:.4f}% | "
            f"{row['directional_affinity_pp']:+.4f} pp |"
        )

    lines.extend(
        [
            "",
            "## Matched test metrics",
            "",
            "| Domain | Condition | Language accuracy | "
            "Language macro-F1 | Dialect language-macro accuracy | "
            "Dialect language-macro F1 | Selection macro-F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in matched.iterrows():
        def percent(value: Any) -> str:
            if pd.isna(value):
                return "—"
            return f"{float(value) * 100.0:.4f}%"

        lines.append(
            f"| {row['train_domain_label']} | "
            f"{row['condition_label']} | "
            f"{percent(row['language_accuracy'])} | "
            f"{percent(row['language_macro_f1'])} | "
            f"{percent(row['dialect_language_macro_accuracy'])} | "
            f"{percent(row['dialect_language_macro_f1'])} | "
            f"{percent(row['condition_selection_macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Positive affinity indicates beneficial transfer.",
            "- Negative affinity indicates task interference.",
            "- Macro-F1 affinity is the primary transfer measure.",
            "- Accuracy affinity is retained as the secondary measure.",
            "- Matched and cross-precision scores are reported separately.",
            "- Test results were not used for checkpoint selection.",
            "",
            "## Generated figures",
            "",
        ]
    )

    lines.extend(
        f"- `{path.relative_to(REPORT)}`"
        for path in figure_paths
    )

    lines.extend(
        [
            "",
            "## Generated tables",
            "",
        ]
    )

    lines.extend(
        f"- `{path.relative_to(REPORT)}`"
        for path in table_paths
    )

    lines.extend(
        [
            "",
            "Confusion-matrix artifacts remain in the original evaluation "
            "directories under `results/exp62_reduced/`.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    for directory in (FIGURES, TABLES, APPENDIX):
        directory.mkdir(parents=True, exist_ok=True)

    completion = RUN / "EXPERIMENT_COMPLETE.json"
    if not completion.is_file():
        raise FileNotFoundError(completion)

    # ------------------------------------------------------------------
    # Condition inventory
    # ------------------------------------------------------------------
    condition_rows: list[dict[str, Any]] = []

    for domain in DOMAINS:
        for condition in CONDITIONS:
            base = RUN / domain / condition
            checkpoint = base / "checkpoints/best.pt"
            parameters, tensors, checkpoint_bytes = checkpoint_statistics(
                checkpoint
            )
            history = history_summary(base / "history.csv")
            best_dev = read_best_dev_metadata(
                base / "best_dev_metrics.json"
            )

            condition_rows.append(
                {
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "active_tasks": "+".join(ACTIVE_TASKS[condition]),
                    "parameter_count": parameters,
                    "model_tensor_count": tensors,
                    "best_checkpoint_bytes": checkpoint_bytes,
                    **history,
                    **best_dev,
                    "training_complete": True,
                    "matched_evaluation_complete": True,
                    "cross_precision_evaluation_complete": True,
                }
            )

    condition_frame = pd.DataFrame(condition_rows)

    # ------------------------------------------------------------------
    # All 12 evaluation cells
    # ------------------------------------------------------------------
    evaluation_rows: list[dict[str, Any]] = []
    per_language_rows: list[dict[str, Any]] = []

    for train_domain in DOMAINS:
        for condition in CONDITIONS:
            for test_domain in TEST_DOMAINS:
                row, payload = metric_row(
                    train_domain,
                    condition,
                    test_domain,
                )
                evaluation_rows.append(row)

                if "dialect" in ACTIVE_TASKS[condition]:
                    for language_row in extract_per_language_dialect(payload):
                        per_language_rows.append(
                            {
                                "train_domain": train_domain,
                                "condition": condition,
                                "test_domain": test_domain,
                                "matched": train_domain == test_domain,
                                "language": language_row["language"],
                                "dialect_accuracy": language_row["accuracy"],
                                "dialect_macro_f1": language_row["macro_f1"],
                                "json_path": language_row["json_path"],
                            }
                        )

    all_results = pd.DataFrame(evaluation_rows)
    matched = all_results[all_results["matched"]].copy()
    cross = all_results[~all_results["matched"]].copy()

    if len(all_results) != 12:
        raise RuntimeError(
            f"Expected 12 evaluation cells; found {len(all_results)}"
        )

    if len(matched) != 6 or len(cross) != 6:
        raise RuntimeError(
            f"Expected 6 matched and 6 cross cells; "
            f"found matched={len(matched)}, cross={len(cross)}"
        )

    macro_affinity, accuracy_affinity = build_affinity_tables(matched)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    table_paths: list[Path] = []

    table_paths.append(
        save_table(
            condition_frame,
            "table_01_condition_configuration.csv",
        )
    )
    table_paths.append(
        save_table(
            percentage_columns(matched),
            "table_02_matched_results.csv",
        )
    )
    table_paths.append(
        save_table(
            macro_affinity,
            "table_03_directional_macro_f1_affinity.csv",
        )
    )
    table_paths.append(
        save_table(
            accuracy_affinity,
            "table_04_directional_accuracy_affinity.csv",
        )
    )
    table_paths.append(
        save_table(
            percentage_columns(cross),
            "table_05_cross_precision_results.csv",
        )
    )

    if per_language_rows:
        per_language_frame = pd.DataFrame(per_language_rows)
        for column in ("dialect_accuracy", "dialect_macro_f1"):
            per_language_frame[column + "_percent"] = (
                pd.to_numeric(
                    per_language_frame[column],
                    errors="coerce",
                )
                * 100.0
            )

        table_paths.append(
            save_table(
                per_language_frame,
                "table_06_per_language_dialect.csv",
            )
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figure_paths = [
        FIGURES / "figure_01_one_bit_matched_macro_f1.png",
        FIGURES / "figure_02_sixteen_bit_matched_macro_f1.png",
        FIGURES / "figure_03_one_bit_affinity_heatmap.png",
        FIGURES / "figure_04_sixteen_bit_affinity_heatmap.png",
        FIGURES / "figure_05_cross_precision_transfer.png",
    ]

    plot_matched_macro_f1(
        matched,
        "one_bit",
        figure_paths[0],
    )
    plot_matched_macro_f1(
        matched,
        "sixteen_bit",
        figure_paths[1],
    )
    plot_affinity_heatmap(
        macro_affinity,
        "one_bit",
        figure_paths[2],
    )
    plot_affinity_heatmap(
        macro_affinity,
        "sixteen_bit",
        figure_paths[3],
    )
    plot_cross_precision(
        all_results,
        figure_paths[4],
    )

    readme = write_results_readme(
        matched,
        macro_affinity,
        figure_paths,
        table_paths,
    )

    # ------------------------------------------------------------------
    # Artifact verification and marker
    # ------------------------------------------------------------------
    artifacts = figure_paths + table_paths + [readme]

    for path in artifacts:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty report artifact: {path}")

    marker = {
        "status": "REPORT_COMPLETE",
        "experiment": "EXP 6.2 reduced LID-dialect task grouping",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "domains": list(DOMAINS),
            "conditions": list(CONDITIONS),
            "tasks": ["language", "dialect"],
            "training_conditions": 6,
            "evaluation_cells": 12,
            "matched_cells": 6,
            "cross_precision_cells": 6,
        },
        "source_completion_marker": str(
            (RUN / "EXPERIMENT_COMPLETE.json").relative_to(ROOT)
        ),
        "report_root": str(REPORT.relative_to(ROOT)),
        "figures": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in figure_paths
        ],
        "tables": [
            {
                "path": str(path.relative_to(ROOT)),
                "rows": int(len(pd.read_csv(path))),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in table_paths
        ],
        "summary": {
            "readme": str(readme.relative_to(ROOT)),
            "readme_sha256": sha256_file(readme),
        },
        "scientific_notes": [
            "This is the reduced six-condition LID-dialect study.",
            "Directional macro-F1 affinity is the primary transfer measure.",
            "Directional accuracy affinity is the secondary measure.",
            "Matched and cross-precision results are kept separate.",
            "No model was retrained during report generation.",
        ],
    }

    write_json(REPORT / "REPORT_COMPLETE.json", marker)

    print("============================================================")
    print(" EXP 6.2 REPORT GENERATED")
    print("============================================================")
    print(f"Matched cells:        {len(matched)}")
    print(f"Cross-precision cells:{len(cross)}")
    print(f"Figures:              {len(figure_paths)}")
    print(f"Tables:               {len(table_paths)}")
    print()
    print("Directional macro-F1 affinities:")

    for _, row in macro_affinity.iterrows():
        print(
            f"  {row['domain_label']:>6} "
            f"A({row['target_task']} | {row['partner_task']}) = "
            f"{row['directional_affinity_pp']:+.4f} pp"
        )

    print()
    print(f"Report: {REPORT}")
    print(f"Marker: {REPORT / 'REPORT_COMPLETE.json'}")
    print("EXP62_REPORT_GENERATION_PASS")


if __name__ == "__main__":
    main()

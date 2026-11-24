from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .config import AnalysisConfig, COMBINED_ROOT, EXP62_ROOT
from .metrics import classification_metrics
from .probe import RidgeLinearProbe, fit_ridge_probe, nested_stratified_indices
from .utils import atomic_json, atomic_text, ensure_finite, flatten_json, utc_now


@dataclass(frozen=True)
class SourceSpec:
    family: str
    model_domain: str
    condition: str
    source_key: str
    display_name: str


@dataclass
class CellData:
    embeddings: np.ndarray
    metadata: pd.DataFrame
    path: Path


LANGUAGES = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")


def source_specs(config: AnalysisConfig) -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    for domain in config.domains:
        for condition in config.conditions:
            specs.append(
                SourceSpec(
                    family="exp62",
                    model_domain=domain,
                    condition=condition,
                    source_key=f"{domain}_{condition}",
                    display_name=f"EXP6.2 {domain} {condition}",
                )
            )
    for domain in config.domains:
        specs.append(
            SourceSpec(
                family="exp61",
                model_domain=domain,
                condition="three_task_reference",
                source_key=domain,
                display_name=f"EXP6.1 {domain} supervised reference",
            )
        )
    for domain in config.domains:
        specs.append(
            SourceSpec(
                family="random",
                model_domain=domain,
                condition="random_encoder",
                source_key="random",
                display_name=f"Random encoder / {domain} input",
            )
        )
    return specs


def opposite(domain: str) -> str:
    return "sixteen_bit" if domain == "one_bit" else "one_bit"


def cell_path(spec: SourceSpec, input_domain: str, split: str) -> Path:
    if spec.family == "exp62":
        return (
            COMBINED_ROOT
            / "runtime/embeddings/exp62"
            / f"model_{spec.source_key}"
            / f"input_{input_domain}"
            / split
        )
    if spec.family == "exp61":
        return (
            COMBINED_ROOT
            / "runtime/reference_embeddings/exp61"
            / f"model_{spec.model_domain}"
            / f"input_{input_domain}"
            / split
        )
    return (
        COMBINED_ROOT
        / "runtime/embeddings/random/model_random"
        / f"input_{input_domain}"
        / split
    )


def load_cell(spec: SourceSpec, input_domain: str, split: str) -> CellData:
    path = cell_path(spec, input_domain, split)
    embeddings_path = path / "embeddings.npy"
    metadata_path = path / "metadata.csv"
    marker_path = path / "COMPLETE.json"
    for item in (embeddings_path, metadata_path, marker_path):
        if not item.is_file() or item.stat().st_size == 0:
            raise RuntimeError(f"missing completed embedding artifact: {item}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path, dtype={"utt_id": str, "language": str})
    if embeddings.ndim != 2 or embeddings.shape[0] != len(metadata):
        raise RuntimeError(f"embedding/metadata mismatch: {path}")
    if metadata["utt_id"].duplicated().any():
        raise RuntimeError(f"duplicate utt_id: {metadata_path}")
    return CellData(embeddings=embeddings, metadata=metadata, path=path)


def assert_alignment(*cells: CellData) -> None:
    reference = cells[0].metadata["utt_id"].astype(str).to_numpy()
    for cell in cells[1:]:
        candidate = cell.metadata["utt_id"].astype(str).to_numpy()
        if len(reference) != len(candidate) or not np.array_equal(reference, candidate):
            raise RuntimeError(
                f"utterance order mismatch between {cells[0].path} and {cell.path}"
            )


def _probe_row(
    spec: SourceSpec,
    task: str,
    evaluation: str,
    result,
    regularization: float,
    language: str | None = None,
) -> dict[str, Any]:
    return {
        "family": spec.family,
        "source": spec.display_name,
        "model_domain": spec.model_domain,
        "condition": spec.condition,
        "task": task,
        "language": language,
        "evaluation": evaluation,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "examples": result.examples,
        "classes": result.classes,
        "regularization": regularization,
    }


def _fit_global_task(
    spec: SourceSpec,
    task: str,
    train: CellData,
    dev: CellData,
    test: CellData,
    cross_test: CellData,
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], RidgeLinearProbe]:
    column = "lid" if task == "lid" else "gender"
    y_train = train.metadata[column].to_numpy(dtype=np.int64)
    y_dev = dev.metadata[column].to_numpy(dtype=np.int64)
    probe, dev_result = fit_ridge_probe(
        train.embeddings,
        y_train,
        dev.embeddings,
        y_dev,
        config.ridge_lambdas,
    )
    test_predictions = probe.predict(test.embeddings)
    cross_predictions = probe.predict(cross_test.embeddings)
    classes = probe.classes_
    test_result = classification_metrics(
        test.metadata[column].to_numpy(dtype=np.int64), test_predictions, classes
    )
    cross_result = classification_metrics(
        cross_test.metadata[column].to_numpy(dtype=np.int64), cross_predictions, classes
    )
    rows = [
        _probe_row(spec, task, "matched_dev", dev_result, probe.regularization),
        _probe_row(spec, task, "matched_test", test_result, probe.regularization),
        _probe_row(spec, task, "cross_test", cross_result, probe.regularization),
    ]
    return rows, probe


def _fit_dialect_task(
    spec: SourceSpec,
    train: CellData,
    dev: CellData,
    test: CellData,
    cross_test: CellData,
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregates: dict[str, list[dict[str, float]]] = {
        "matched_dev": [],
        "matched_test": [],
        "cross_test": [],
    }
    regularizations: list[float] = []
    total_examples = {key: 0 for key in aggregates}
    total_classes = 0
    for language in LANGUAGES:
        train_mask = train.metadata["language"].astype(str).to_numpy() == language
        dev_mask = dev.metadata["language"].astype(str).to_numpy() == language
        test_mask = test.metadata["language"].astype(str).to_numpy() == language
        cross_mask = cross_test.metadata["language"].astype(str).to_numpy() == language
        if not (train_mask.any() and dev_mask.any() and test_mask.any() and cross_mask.any()):
            raise RuntimeError(f"missing dialect examples for language={language}")
        y_train = train.metadata.loc[train_mask, "dialect"].to_numpy(dtype=np.int64)
        y_dev = dev.metadata.loc[dev_mask, "dialect"].to_numpy(dtype=np.int64)
        probe, dev_result = fit_ridge_probe(
            train.embeddings[train_mask],
            y_train,
            dev.embeddings[dev_mask],
            y_dev,
            config.ridge_lambdas,
        )
        matched_result = classification_metrics(
            test.metadata.loc[test_mask, "dialect"].to_numpy(dtype=np.int64),
            probe.predict(test.embeddings[test_mask]),
            probe.classes_,
        )
        cross_result = classification_metrics(
            cross_test.metadata.loc[cross_mask, "dialect"].to_numpy(dtype=np.int64),
            probe.predict(cross_test.embeddings[cross_mask]),
            probe.classes_,
        )
        for evaluation, result in (
            ("matched_dev", dev_result),
            ("matched_test", matched_result),
            ("cross_test", cross_result),
        ):
            rows.append(
                _probe_row(
                    spec,
                    "dialect_language",
                    evaluation,
                    result,
                    probe.regularization,
                    language=language,
                )
            )
            aggregates[evaluation].append(
                {"accuracy": result.accuracy, "macro_f1": result.macro_f1}
            )
            total_examples[evaluation] += result.examples
        regularizations.append(probe.regularization)
        total_classes += len(probe.classes_)
    for evaluation, values in aggregates.items():
        rows.append(
            {
                "family": spec.family,
                "source": spec.display_name,
                "model_domain": spec.model_domain,
                "condition": spec.condition,
                "task": "dialect",
                "language": None,
                "evaluation": evaluation,
                "accuracy": float(np.mean([value["accuracy"] for value in values])),
                "macro_f1": float(np.mean([value["macro_f1"] for value in values])),
                "examples": total_examples[evaluation],
                "classes": total_classes,
                "regularization": float(np.median(regularizations)),
            }
        )
    return rows


def run_full_probes(config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs(config):
        print(f"FULL_PROBES source={spec.display_name}", flush=True)
        matched = spec.model_domain
        train = load_cell(spec, matched, "train")
        dev = load_cell(spec, matched, "dev")
        test = load_cell(spec, matched, "test")
        cross_test = load_cell(spec, opposite(matched), "test")
        assert_alignment(test, cross_test)
        lid_rows, _ = _fit_global_task(
            spec, "lid", train, dev, test, cross_test, config
        )
        gender_rows, _ = _fit_global_task(
            spec, "gender", train, dev, test, cross_test, config
        )
        rows.extend(lid_rows)
        rows.extend(gender_rows)
        rows.extend(_fit_dialect_task(spec, train, dev, test, cross_test, config))
    result = pd.DataFrame(rows)
    output = COMBINED_ROOT / "results/exp65/probe_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def _stratum_labels(metadata: pd.DataFrame) -> np.ndarray:
    return (
        metadata["language"].astype(str)
        + "|g"
        + metadata["gender"].astype(str)
    ).to_numpy()


def run_fewshot_gender(config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs(config):
        print(f"FEWSHOT_GENDER source={spec.display_name}", flush=True)
        matched = spec.model_domain
        train = load_cell(spec, matched, "train")
        dev = load_cell(spec, matched, "dev")
        test = load_cell(spec, matched, "test")
        cross_test = load_cell(spec, opposite(matched), "test")
        assert_alignment(test, cross_test)
        subsets = nested_stratified_indices(
            _stratum_labels(train.metadata), config.fewshot_fractions, config.seed
        )
        for fraction in config.fewshot_fractions:
            indices = subsets[float(fraction)]
            probe, dev_result = fit_ridge_probe(
                train.embeddings[indices],
                train.metadata.iloc[indices]["gender"].to_numpy(dtype=np.int64),
                dev.embeddings,
                dev.metadata["gender"].to_numpy(dtype=np.int64),
                config.ridge_lambdas,
            )
            matched_result = classification_metrics(
                test.metadata["gender"].to_numpy(dtype=np.int64),
                probe.predict(test.embeddings),
                probe.classes_,
            )
            cross_result = classification_metrics(
                cross_test.metadata["gender"].to_numpy(dtype=np.int64),
                probe.predict(cross_test.embeddings),
                probe.classes_,
            )
            for evaluation, evaluation_result in (
                ("matched_dev", dev_result),
                ("matched_test", matched_result),
                ("cross_test", cross_result),
            ):
                rows.append(
                    {
                        "family": spec.family,
                        "source": spec.display_name,
                        "model_domain": spec.model_domain,
                        "condition": spec.condition,
                        "fraction": float(fraction),
                        "labelled_examples": len(indices),
                        "evaluation": evaluation,
                        "accuracy": evaluation_result.accuracy,
                        "macro_f1": evaluation_result.macro_f1,
                        "examples": evaluation_result.examples,
                        "regularization": probe.regularization,
                        "seed": config.seed,
                    }
                )
    result = pd.DataFrame(rows)
    output = COMBINED_ROOT / "results/exp68/fewshot_gender.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def _stratified_sample_indices(
    metadata: pd.DataFrame,
    count: int,
    seed: int,
    columns: tuple[str, ...] = ("language", "gender"),
) -> np.ndarray:
    if len(metadata) <= count:
        return np.arange(len(metadata), dtype=np.int64)
    labels = metadata[list(columns)].astype(str).agg("|".join, axis=1).to_numpy()
    rng = np.random.default_rng(seed)
    groups = {label: rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)}
    selected: list[int] = []
    while len(selected) < count:
        advanced = False
        for label in sorted(groups):
            position = sum(1 for value in selected if labels[value] == label)
            group = groups[label]
            if position < len(group):
                selected.append(int(group[position]))
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _centroid_geometry(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    centroids: list[np.ndarray] = []
    within_values: list[float] = []
    for label in np.unique(labels):
        block = z[labels == label]
        centroid = block.mean(axis=0)
        centroids.append(centroid)
        within_values.append(float(np.mean(np.sum(np.square(block - centroid), axis=1))))
    between_values: list[float] = []
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            between_values.append(float(np.sum(np.square(centroids[left] - centroids[right]))))
    within = float(np.mean(within_values))
    between = float(np.mean(between_values)) if between_values else float("nan")
    return {
        "within_scatter": within,
        "between_centroid_distance": between,
        "between_within_ratio": between / within if within > 0 else float("nan"),
    }


def _silhouette(
    x: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    seed: int,
) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    sample_size = min(sample_count, len(labels))
    return float(
        silhouette_score(
            np.asarray(x, dtype=np.float32),
            labels,
            metric="euclidean",
            sample_size=sample_size,
            random_state=seed,
        )
    )


def _task_geometry(
    x: np.ndarray,
    metadata: pd.DataFrame,
    task: str,
    config: AnalysisConfig,
) -> dict[str, float]:
    if task == "lid":
        labels = metadata["lid"].to_numpy(dtype=np.int64)
        result = _centroid_geometry(x, labels)
        result["silhouette"] = _silhouette(
            x, labels, config.silhouette_samples, config.seed
        )
        return result
    if task == "gender":
        labels = metadata["gender"].to_numpy(dtype=np.int64)
        result = _centroid_geometry(x, labels)
        result["silhouette"] = _silhouette(
            x, labels, config.silhouette_samples, config.seed
        )
        within_language: list[float] = []
        for language in LANGUAGES:
            mask = metadata["language"].astype(str).to_numpy() == language
            within_language.append(
                _silhouette(
                    x[mask],
                    metadata.loc[mask, "gender"].to_numpy(dtype=np.int64),
                    min(500, int(mask.sum())),
                    config.seed,
                )
            )
        result["within_language_silhouette_macro"] = float(
            np.nanmean(within_language)
        )
        return result
    per_language: list[dict[str, float]] = []
    for language in LANGUAGES:
        mask = metadata["language"].astype(str).to_numpy() == language
        block = x[mask]
        labels = metadata.loc[mask, "dialect"].to_numpy(dtype=np.int64)
        geometry = _centroid_geometry(block, labels)
        geometry["silhouette"] = _silhouette(
            block, labels, min(500, len(block)), config.seed
        )
        per_language.append(geometry)
    keys = per_language[0].keys()
    return {key: float(np.nanmean([row[key] for row in per_language])) for key in keys}


def run_geometry(config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    projections: dict[str, dict[str, np.ndarray]] = {}
    for spec in source_specs(config):
        print(f"GEOMETRY source={spec.display_name}", flush=True)
        cell = load_cell(spec, spec.model_domain, "test")
        sample_indices = _stratified_sample_indices(
            cell.metadata, config.umap_samples, config.seed
        )
        sample_x = np.asarray(cell.embeddings[sample_indices], dtype=np.float32)
        ensure_finite("geometry sample", sample_x)
        pca = PCA(n_components=min(20, sample_x.shape[1]), random_state=config.seed)
        pca_projection = pca.fit_transform(sample_x)
        projections[spec.display_name] = {
            "indices": sample_indices,
            "pca": pca_projection[:, :2],
            "explained": pca.explained_variance_ratio_,
        }
        for task in ("lid", "dialect", "gender"):
            geometry = _task_geometry(sample_x, cell.metadata.iloc[sample_indices], task, config)
            rows.append(
                {
                    "family": spec.family,
                    "source": spec.display_name,
                    "model_domain": spec.model_domain,
                    "condition": spec.condition,
                    "task": task,
                    "samples": len(sample_indices),
                    "pca_variance_first_2": float(pca.explained_variance_ratio_[:2].sum()),
                    "pca_variance_first_10": float(pca.explained_variance_ratio_[:10].sum()),
                    **geometry,
                }
            )
    result = pd.DataFrame(rows)
    output = COMBINED_ROOT / "results/exp65/geometry_metrics.csv"
    result.to_csv(output, index=False)
    return result, projections


def _find_metric(flat: dict[str, Any], candidate_fragments: Iterable[str]) -> float:
    candidates: list[tuple[int, str, float]] = []
    for key, value in flat.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        normalized = key.lower().replace("-", "_")
        for fragment in candidate_fragments:
            fragment = fragment.lower().replace("-", "_")
            if normalized == fragment or normalized.endswith("." + fragment) or fragment in normalized:
                candidates.append((len(normalized), key, float(value)))
                break
    if not candidates:
        return float("nan")
    candidates.sort()
    return candidates[0][2]


def collect_native_metrics(config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for domain in config.domains:
        for condition in config.conditions:
            for test_domain in config.domains:
                path = (
                    EXP62_ROOT
                    / "results/exp62_reduced"
                    / domain
                    / condition
                    / f"test_{test_domain}"
                    / "metrics.json"
                )
                if not path.is_file():
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                flat = flatten_json(value)
                raw_row: dict[str, Any] = {
                    "model_domain": domain,
                    "condition": condition,
                    "test_domain": test_domain,
                    "path": str(path),
                }
                raw_row.update({key: item for key, item in flat.items() if isinstance(item, (int, float, str, bool))})
                raw_rows.append(raw_row)
                rows.append(
                    {
                        "model_domain": domain,
                        "condition": condition,
                        "test_domain": test_domain,
                        "evaluation": "matched_test" if domain == test_domain else "cross_test",
                        "lid_accuracy": _find_metric(flat, ["language_accuracy", "lid_accuracy"]),
                        "lid_macro_f1": _find_metric(flat, ["language_macro_f1", "lid_macro_f1"]),
                        "dialect_accuracy": _find_metric(flat, ["dialect_language_macro_accuracy"]),
                        "dialect_macro_f1": _find_metric(flat, ["dialect_language_macro_f1"]),
                        "path": str(path),
                    }
                )
    metrics = pd.DataFrame(rows)
    raw = pd.DataFrame(raw_rows)
    reports = COMBINED_ROOT / "reports/exp65/tables"
    reports.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(reports / "table_06_native_head_metrics.csv", index=False)
    raw.to_csv(reports / "appendix_native_metrics_flattened.csv", index=False)
    return metrics, raw


def derive_probe_tables(probes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    main = probes[
        probes["task"].isin(["lid", "dialect", "gender"])
        & probes["evaluation"].isin(["matched_test", "cross_test"])
    ].copy()
    matched = main[main["evaluation"] == "matched_test"].copy()
    pivot = matched.pivot_table(
        index=["family", "model_domain", "condition", "source"],
        columns="task",
        values="macro_f1",
    ).reset_index()
    exp62 = matched[matched["family"] == "exp62"]
    delta_rows: list[dict[str, Any]] = []
    for domain in ("one_bit", "sixteen_bit"):
        group = exp62[exp62["model_domain"] == domain]
        def value(condition: str, task: str) -> float:
            selected = group[(group["condition"] == condition) & (group["task"] == task)]
            return float(selected.iloc[0]["macro_f1"])
        delta_rows.extend(
            [
                {
                    "model_domain": domain,
                    "task": "lid",
                    "joint_condition": "lid_dialect",
                    "single_condition": "lid_only",
                    "joint_macro_f1": value("lid_dialect", "lid"),
                    "single_macro_f1": value("lid_only", "lid"),
                    "delta_joint_minus_single": value("lid_dialect", "lid") - value("lid_only", "lid"),
                },
                {
                    "model_domain": domain,
                    "task": "dialect",
                    "joint_condition": "lid_dialect",
                    "single_condition": "dialect_only",
                    "joint_macro_f1": value("lid_dialect", "dialect"),
                    "single_macro_f1": value("dialect_only", "dialect"),
                    "delta_joint_minus_single": value("lid_dialect", "dialect") - value("dialect_only", "dialect"),
                },
                {
                    "model_domain": domain,
                    "task": "gender",
                    "joint_condition": "lid_dialect",
                    "single_condition": "mean(lid_only,dialect_only)",
                    "joint_macro_f1": value("lid_dialect", "gender"),
                    "single_macro_f1": (value("lid_only", "gender") + value("dialect_only", "gender")) / 2.0,
                    "delta_joint_minus_single": value("lid_dialect", "gender") - (value("lid_only", "gender") + value("dialect_only", "gender")) / 2.0,
                },
            ]
        )
    deltas = pd.DataFrame(delta_rows)
    precision_rows: list[dict[str, Any]] = []
    for condition in ("lid_only", "dialect_only", "lid_dialect"):
        for task in ("lid", "dialect", "gender"):
            one = exp62[(exp62["model_domain"] == "one_bit") & (exp62["condition"] == condition) & (exp62["task"] == task)].iloc[0]
            sixteen = exp62[(exp62["model_domain"] == "sixteen_bit") & (exp62["condition"] == condition) & (exp62["task"] == task)].iloc[0]
            precision_rows.append(
                {
                    "condition": condition,
                    "task": task,
                    "one_bit_accuracy": one["accuracy"],
                    "sixteen_bit_accuracy": sixteen["accuracy"],
                    "accuracy_delta_16_minus_1": sixteen["accuracy"] - one["accuracy"],
                    "one_bit_macro_f1": one["macro_f1"],
                    "sixteen_bit_macro_f1": sixteen["macro_f1"],
                    "macro_f1_delta_16_minus_1": sixteen["macro_f1"] - one["macro_f1"],
                }
            )
    precision = pd.DataFrame(precision_rows)
    return {"main": main, "pivot": pivot, "deltas": deltas, "precision": precision}


def derive_fewshot_tables(fewshot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    matched = fewshot[fewshot["evaluation"] == "matched_test"].copy()
    cross = fewshot[fewshot["evaluation"] == "cross_test"].copy()
    gain_rows: list[dict[str, Any]] = []
    for _, row in matched[matched["family"] == "exp62"].iterrows():
        random_row = matched[
            (matched["family"] == "random")
            & (matched["model_domain"] == row["model_domain"])
            & (matched["fraction"] == row["fraction"])
        ].iloc[0]
        reference_row = matched[
            (matched["family"] == "exp61")
            & (matched["model_domain"] == row["model_domain"])
            & (matched["fraction"] == row["fraction"])
        ].iloc[0]
        denominator = reference_row["macro_f1"] - random_row["macro_f1"]
        recovery = (
            (row["macro_f1"] - random_row["macro_f1"]) / denominator
            if abs(denominator) > 1e-12
            else float("nan")
        )
        gain_rows.append(
            {
                "model_domain": row["model_domain"],
                "condition": row["condition"],
                "fraction": row["fraction"],
                "transfer_accuracy": row["accuracy"],
                "transfer_macro_f1": row["macro_f1"],
                "random_accuracy": random_row["accuracy"],
                "random_macro_f1": random_row["macro_f1"],
                "supervised_reference_accuracy": reference_row["accuracy"],
                "supervised_reference_macro_f1": reference_row["macro_f1"],
                "accuracy_gain_over_random": row["accuracy"] - random_row["accuracy"],
                "macro_f1_gain_over_random": row["macro_f1"] - random_row["macro_f1"],
                "direct_fraction_of_reference_f1": row["macro_f1"] / reference_row["macro_f1"] if reference_row["macro_f1"] else float("nan"),
                "random_adjusted_fraction_recovered": recovery,
            }
        )
    gains = pd.DataFrame(gain_rows)
    matched_cross = matched.merge(
        cross,
        on=["family", "source", "model_domain", "condition", "fraction", "labelled_examples", "seed"],
        suffixes=("_matched", "_cross"),
    )
    matched_cross["accuracy_cross_drop"] = matched_cross["accuracy_matched"] - matched_cross["accuracy_cross"]
    matched_cross["macro_f1_cross_drop"] = matched_cross["macro_f1_matched"] - matched_cross["macro_f1_cross"]
    summary_100 = gains[gains["fraction"] == 1.0].copy()
    return {
        "matched": matched,
        "gains": gains,
        "matched_cross": matched_cross,
        "summary_100": summary_100,
    }


def _plot_heatmap(frame: pd.DataFrame, path: Path, dpi: int) -> None:
    exp62 = frame[(frame["family"] == "exp62") & (frame["evaluation"] == "matched_test") & frame["task"].isin(["lid", "dialect", "gender"])]
    pivot = exp62.pivot_table(index=["model_domain", "condition"], columns="task", values="macro_f1").reindex(columns=["lid", "dialect", "gender"])
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    image = axis.imshow(pivot.to_numpy() * 100.0, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), [value.upper() for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{domain} / {condition}" for domain, condition in pivot.index])
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, f"{pivot.iloc[row, column]*100:.1f}", ha="center", va="center", color="white" if pivot.iloc[row, column] < 0.7 else "black", fontsize=8)
    axis.set_title("Frozen-probe macro-F1 (%)")
    figure.colorbar(image, ax=axis, label="Macro-F1 (%)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_joint_deltas(deltas: pd.DataFrame, path: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    tasks = ["lid", "dialect", "gender"]
    x = np.arange(len(tasks))
    width = 0.36
    for index, domain in enumerate(("one_bit", "sixteen_bit")):
        values = [float(deltas[(deltas.model_domain == domain) & (deltas.task == task)].iloc[0].delta_joint_minus_single) * 100 for task in tasks]
        axis.bar(x + (index - 0.5) * width, values, width, label=domain)
    axis.axhline(0.0, linewidth=1, color="black")
    axis.set_xticks(x, ["LID", "Dialect", "Gender"])
    axis.set_ylabel("Joint minus single-task macro-F1 (percentage points)")
    axis.set_title("Representation benefit of LID+dialect supervision")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _umap_projection(x: np.ndarray, seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
    )
    return reducer.fit_transform(np.asarray(x, dtype=np.float32))


def _scatter_by_labels(axis, coordinates: np.ndarray, labels: np.ndarray, title: str) -> None:
    labels = np.asarray(labels)
    unique = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20", max(2, len(unique)))
    for index, label in enumerate(unique):
        mask = labels == label
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=5,
            alpha=0.55,
            label=str(label),
            color=cmap(index),
            linewidths=0,
        )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    if len(unique) <= 10:
        axis.legend(markerscale=2.5, fontsize=7, frameon=False)


def generate_umap_and_tsne(config: AnalysisConfig) -> None:
    figure_root = COMBINED_ROOT / "reports/exp65/figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    exp62_specs = [spec for spec in source_specs(config) if spec.family == "exp62"]
    canonical_spec = next(spec for spec in exp62_specs if spec.model_domain == "one_bit" and spec.condition == "lid_dialect")
    canonical = load_cell(canonical_spec, "one_bit", "test")
    indices = _stratified_sample_indices(canonical.metadata, config.umap_samples, config.seed)
    canonical_ids = canonical.metadata.iloc[indices]["utt_id"].astype(str).to_numpy()

    def aligned_sample(spec: SourceSpec) -> tuple[np.ndarray, pd.DataFrame]:
        cell = load_cell(spec, spec.model_domain, "test")
        lookup = pd.Series(np.arange(len(cell.metadata)), index=cell.metadata["utt_id"].astype(str))
        positions = lookup.loc[canonical_ids].to_numpy(dtype=np.int64)
        return np.asarray(cell.embeddings[positions], dtype=np.float32), cell.metadata.iloc[positions].reset_index(drop=True)

    for domain in ("one_bit", "sixteen_bit"):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, condition in zip(axes, ("lid_only", "dialect_only", "lid_dialect")):
            spec = next(spec for spec in exp62_specs if spec.model_domain == domain and spec.condition == condition)
            x, metadata = aligned_sample(spec)
            coordinates = _umap_projection(x, config.seed)
            _scatter_by_labels(axis, coordinates, metadata["language"].astype(str).to_numpy(), condition)
        figure.suptitle(f"UMAP by language — {domain}")
        figure.tight_layout()
        figure.savefig(figure_root / f"figure_03_umap_{domain}_task_groupings.png", dpi=config.output_dpi, bbox_inches="tight")
        plt.close(figure)

    primary_specs = [
        next(spec for spec in exp62_specs if spec.model_domain == domain and spec.condition == "lid_dialect")
        for domain in ("one_bit", "sixteen_bit")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for column, spec in enumerate(primary_specs):
        x, metadata = aligned_sample(spec)
        coordinates = _umap_projection(x, config.seed)
        _scatter_by_labels(axes[0, column], coordinates, metadata["language"].astype(str).to_numpy(), f"{spec.model_domain}: language")
        _scatter_by_labels(axes[1, column], coordinates, metadata["gender"].astype(str).to_numpy(), f"{spec.model_domain}: gender")
    figure.suptitle("LID+dialect representation geometry across precision")
    figure.tight_layout()
    figure.savefig(figure_root / "figure_04_umap_lid_dialect_precision.png", dpi=config.output_dpi, bbox_inches="tight")
    plt.close(figure)

    for spec in primary_specs:
        x, metadata = aligned_sample(spec)
        coordinates = _umap_projection(x, config.seed)
        figure, axes = plt.subplots(2, 4, figsize=(15, 7.5))
        for axis, language in zip(axes.ravel(), LANGUAGES):
            mask = metadata["language"].astype(str).to_numpy() == language
            _scatter_by_labels(
                axis,
                coordinates[mask],
                metadata.loc[mask, "dialect"].astype(str).to_numpy(),
                language,
            )
        figure.suptitle(f"Within-language dialect UMAP — {spec.model_domain} LID+dialect")
        figure.tight_layout()
        figure.savefig(figure_root / f"appendix_umap_dialect_{spec.model_domain}.png", dpi=config.output_dpi, bbox_inches="tight")
        plt.close(figure)

    tsne_count = min(config.tsne_samples, len(canonical_ids))
    tsne_ids = canonical_ids[:tsne_count]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, spec in zip(axes, primary_specs):
        cell = load_cell(spec, spec.model_domain, "test")
        lookup = pd.Series(np.arange(len(cell.metadata)), index=cell.metadata["utt_id"].astype(str))
        positions = lookup.loc[tsne_ids].to_numpy(dtype=np.int64)
        x = np.asarray(cell.embeddings[positions], dtype=np.float32)
        pca = PCA(n_components=min(50, x.shape[1]), random_state=config.seed)
        reduced = pca.fit_transform(x)
        coordinates = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=config.seed,
            method="barnes_hut",
        ).fit_transform(reduced)
        metadata = cell.metadata.iloc[positions]
        _scatter_by_labels(axis, coordinates, metadata["language"].astype(str).to_numpy(), spec.model_domain)
    figure.suptitle("t-SNE of LID+dialect embeddings by language")
    figure.tight_layout()
    figure.savefig(figure_root / "appendix_tsne_lid_dialect_precision.png", dpi=config.output_dpi, bbox_inches="tight")
    plt.close(figure)


def generate_geometry_figure(geometry: pd.DataFrame, config: AnalysisConfig) -> None:
    selected = geometry[(geometry["family"] == "exp62") & (geometry["task"].isin(["lid", "dialect", "gender"]))]
    labels = [f"{row.model_domain}\n{row.condition}" for row in selected.drop_duplicates(["model_domain", "condition"]).itertuples()]
    sources = selected.drop_duplicates(["model_domain", "condition"])[["model_domain", "condition"]]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for axis, task in zip(axes, ("lid", "dialect", "gender")):
        values = []
        for row in sources.itertuples(index=False):
            match = selected[(selected.model_domain == row.model_domain) & (selected.condition == row.condition) & (selected.task == task)].iloc[0]
            values.append(float(match.between_within_ratio))
        axis.bar(np.arange(len(values)), values)
        axis.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(task.upper())
        axis.set_ylabel("Between-centroid / within-class scatter")
    figure.suptitle("Frozen representation separation")
    figure.tight_layout()
    path = COMBINED_ROOT / "reports/exp65/figures/figure_05_geometry_separation.png"
    figure.savefig(path, dpi=config.output_dpi, bbox_inches="tight")
    plt.close(figure)


def generate_fewshot_figures(fewshot: pd.DataFrame, gains: pd.DataFrame, config: AnalysisConfig) -> None:
    root = COMBINED_ROOT / "reports/exp68/figures"
    root.mkdir(parents=True, exist_ok=True)
    matched = fewshot[fewshot.evaluation == "matched_test"]
    for metric, filename, ylabel in (
        ("accuracy", "figure_01_gender_accuracy_learning_curves.png", "Gender accuracy (%)"),
        ("macro_f1", "figure_02_gender_macro_f1_learning_curves.png", "Gender macro-F1 (%)"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for axis, domain in zip(axes, ("one_bit", "sixteen_bit")):
            subset = matched[matched.model_domain == domain]
            for (family, condition), group in subset.groupby(["family", "condition"], sort=False):
                if family == "exp62" or condition in {"three_task_reference", "random_encoder"}:
                    axis.plot(group.fraction * 100.0, group[metric] * 100.0, marker="o", label=f"{family}:{condition}")
            axis.set_title(domain)
            axis.set_xlabel("Labelled gender training fraction (%)")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.suptitle("Held-out gender transfer")
        figure.tight_layout()
        figure.savefig(root / filename, dpi=config.output_dpi, bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    final = gains[gains.fraction == 1.0].copy()
    labels = [f"{row.model_domain}\n{row.condition}" for row in final.itertuples()]
    axis.bar(np.arange(len(final)), final.macro_f1_gain_over_random * 100.0)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(final)), labels, rotation=45, ha="right")
    axis.set_ylabel("Macro-F1 gain over random (percentage points)")
    axis.set_title("Transfer value at 100% probe labels")
    figure.tight_layout()
    figure.savefig(root / "figure_03_transfer_gain_over_random.png", dpi=config.output_dpi, bbox_inches="tight")
    plt.close(figure)

    cross = fewshot[(fewshot.family == "exp62") & (fewshot.condition == "lid_dialect") & (fewshot.fraction == 1.0) & fewshot.evaluation.isin(["matched_test", "cross_test"])]
    pivot = cross.pivot_table(index="model_domain", columns="evaluation", values="macro_f1")
    figure, axis = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(pivot))
    width = 0.36
    axis.bar(x - width / 2, pivot["matched_test"] * 100.0, width, label="matched")
    axis.bar(x + width / 2, pivot["cross_test"] * 100.0, width, label="cross precision")
    axis.set_xticks(x, pivot.index)
    axis.set_ylabel("Gender macro-F1 (%)")
    axis.set_title("Direct cross-precision held-out gender transfer")
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "figure_04_matched_vs_cross_precision.png", dpi=config.output_dpi, bbox_inches="tight")
    plt.close(figure)


def native_probe_comparison(native: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    exp62 = probes[(probes.family == "exp62") & probes.task.isin(["lid", "dialect"]) & probes.evaluation.isin(["matched_test", "cross_test"])]
    for probe_row in exp62.itertuples(index=False):
        native_row = native[(native.model_domain == probe_row.model_domain) & (native.condition == probe_row.condition) & (native.evaluation == probe_row.evaluation)]
        if native_row.empty:
            continue
        native_row = native_row.iloc[0]
        native_accuracy = native_row["lid_accuracy"] if probe_row.task == "lid" else native_row["dialect_accuracy"]
        native_f1 = native_row["lid_macro_f1"] if probe_row.task == "lid" else native_row["dialect_macro_f1"]
        rows.append(
            {
                "model_domain": probe_row.model_domain,
                "condition": probe_row.condition,
                "task": probe_row.task,
                "evaluation": probe_row.evaluation,
                "native_accuracy": native_accuracy,
                "probe_accuracy": probe_row.accuracy,
                "probe_minus_native_accuracy": probe_row.accuracy - native_accuracy if np.isfinite(native_accuracy) else float("nan"),
                "native_macro_f1": native_f1,
                "probe_macro_f1": probe_row.macro_f1,
                "probe_minus_native_macro_f1": probe_row.macro_f1 - native_f1 if np.isfinite(native_f1) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value * 100:.2f}%"


def generate_summary(
    probes: pd.DataFrame,
    probe_tables: dict[str, pd.DataFrame],
    geometry: pd.DataFrame,
    fewshot: pd.DataFrame,
    fewshot_tables: dict[str, pd.DataFrame],
) -> str:
    deltas = probe_tables["deltas"]
    precision = probe_tables["precision"]
    gains = fewshot_tables["gains"]
    lines = [
        "# EXP 6.5 + EXP 6.8 Generated Results Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Scope",
        "",
        "This report analyzes frozen 192-dimensional ECAPA representations. No encoder was trained in EXP 6.5 or EXP 6.8.",
        "The available project labels are LID, language-local dialect, and gender; domain and accent are not tested.",
        "",
        "## EXP 6.5 — joint versus single-task representation",
        "",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            f"- {row.model_domain} {row.task}: joint minus comparison = {row.delta_joint_minus_single * 100:+.2f} percentage points macro-F1."
        )
    lines.extend(["", "## EXP 6.5 — precision effects", ""])
    largest = precision.iloc[np.argmax(np.abs(precision.macro_f1_delta_16_minus_1.to_numpy()))]
    lines.append(
        f"- Largest absolute 16-bit minus 1-bit probe gap: {largest['condition']} / {largest['task']} = {largest['macro_f1_delta_16_minus_1'] * 100:+.2f} percentage points macro-F1."
    )
    lines.extend(["", "## EXP 6.8 — held-out gender transfer", ""])
    primary = gains[(gains.condition == "lid_dialect") & (gains.fraction == 1.0)]
    for row in primary.itertuples(index=False):
        lines.append(
            f"- {row.model_domain} LID+dialect encoder: gender macro-F1 {_format_percent(row.transfer_macro_f1)}, gain over random {row.macro_f1_gain_over_random * 100:+.2f} points, random-adjusted supervised recovery {_format_percent(row.random_adjusted_fraction_recovered)}."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Frozen probes quantify linearly recoverable information; they do not prove causal disentanglement.",
            "- Dialect metrics are computed inside each language and macro-averaged, because dialect labels are language-local.",
            "- Cross-precision results use the same probe without target-domain refitting.",
            "- EXP 6.8 is held-out-task few-shot probing, not literal zero-label classification.",
            "",
            "## Output inventory",
            "",
            "See `reports/exp65/tables`, `reports/exp65/figures`, `reports/exp68/tables`, and `reports/exp68/figures`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tables(
    probes: pd.DataFrame,
    probe_tables: dict[str, pd.DataFrame],
    geometry: pd.DataFrame,
    native: pd.DataFrame,
    comparison: pd.DataFrame,
    fewshot: pd.DataFrame,
    fewshot_tables: dict[str, pd.DataFrame],
) -> None:
    exp65 = COMBINED_ROOT / "reports/exp65/tables"
    exp68 = COMBINED_ROOT / "reports/exp68/tables"
    exp65.mkdir(parents=True, exist_ok=True)
    exp68.mkdir(parents=True, exist_ok=True)
    probes.to_csv(exp65 / "table_01_all_probe_results.csv", index=False)
    probe_tables["pivot"].to_csv(exp65 / "table_02_matched_probe_macro_f1.csv", index=False)
    probe_tables["deltas"].to_csv(exp65 / "table_03_joint_vs_single_task_deltas.csv", index=False)
    probe_tables["precision"].to_csv(exp65 / "table_04_precision_deltas.csv", index=False)
    geometry.to_csv(exp65 / "table_05_geometry_metrics.csv", index=False)
    native.to_csv(exp65 / "table_06_native_head_metrics.csv", index=False)
    comparison.to_csv(exp65 / "table_07_native_head_vs_probe.csv", index=False)
    fewshot.to_csv(exp68 / "table_01_fewshot_gender_all_results.csv", index=False)
    fewshot_tables["summary_100"].to_csv(exp68 / "table_02_transfer_summary_100_percent.csv", index=False)
    fewshot_tables["gains"].to_csv(exp68 / "table_03_transfer_gain_and_recovery.csv", index=False)
    fewshot_tables["matched_cross"].to_csv(exp68 / "table_04_matched_vs_cross_precision.csv", index=False)


def run(config: AnalysisConfig) -> None:
    started = time.monotonic()
    probes = run_full_probes(config)
    probe_tables = derive_probe_tables(probes)
    fewshot = run_fewshot_gender(config)
    fewshot_tables = derive_fewshot_tables(fewshot)
    geometry, _ = run_geometry(config)
    native, _ = collect_native_metrics(config)
    comparison = native_probe_comparison(native, probes)
    write_tables(
        probes,
        probe_tables,
        geometry,
        native,
        comparison,
        fewshot,
        fewshot_tables,
    )
    figure_root = COMBINED_ROOT / "reports/exp65/figures"
    _plot_heatmap(
        probes,
        figure_root / "figure_01_probe_macro_f1_heatmap.png",
        config.output_dpi,
    )
    _plot_joint_deltas(
        probe_tables["deltas"],
        figure_root / "figure_02_joint_vs_single_task_deltas.png",
        config.output_dpi,
    )
    generate_geometry_figure(geometry, config)
    generate_umap_and_tsne(config)
    generate_fewshot_figures(fewshot, fewshot_tables["gains"], config)
    summary = generate_summary(probes, probe_tables, geometry, fewshot, fewshot_tables)
    atomic_text(COMBINED_ROOT / "reports/combined/README_RESULTS.md", summary)
    elapsed = time.monotonic() - started
    marker = {
        "status": "EXP65_EXP68_ANALYSIS_COMPLETE",
        "created_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "exp65_probe_rows": len(probes),
        "exp65_geometry_rows": len(geometry),
        "exp68_fewshot_rows": len(fewshot),
        "scientific_scope": "three-task cache-compatible LID, language-local dialect, and gender",
        "encoder_training_performed": False,
    }
    atomic_json(COMBINED_ROOT / "reports/combined/REPORT_COMPLETE.json", marker)
    print("EXP65_EXP68_ANALYSIS_COMPLETE")

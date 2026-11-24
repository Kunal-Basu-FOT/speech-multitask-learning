from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch


def confusion_matrix(true_values, predicted_values, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for true_value, predicted_value in zip(true_values, predicted_values, strict=True):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict:
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_support > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
    }


@dataclass
class PredictionAccumulator:
    labels: object
    lid_true: list[int] = field(default_factory=list)
    lid_pred: list[int] = field(default_factory=list)
    gender_true: list[int] = field(default_factory=list)
    gender_pred: list[int] = field(default_factory=list)
    dialect_true: dict[str, list[int]] = field(default_factory=dict)
    dialect_pred: dict[str, list[int]] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dialect_true = {language: [] for language in self.labels.languages}
        self.dialect_pred = {language: [] for language in self.labels.languages}

    def update(self, batch: dict, output: dict) -> None:
        lid_probabilities = output["lid_probabilities"].detach().cpu()
        lid_predictions = lid_probabilities.argmax(dim=-1).tolist()
        gender_predictions = output["gender_logits"].argmax(dim=-1).detach().cpu().tolist()
        lid_truth = batch["lid"].detach().cpu().tolist()
        gender_truth = batch["gender"].detach().cpu().tolist()

        self.lid_pred.extend(lid_predictions)
        self.lid_true.extend(lid_truth)
        self.gender_pred.extend(gender_predictions)
        self.gender_true.extend(gender_truth)

        dialect_predictions: list[int | None] = [None] * len(lid_predictions)
        for language, (indices, logits) in output["dialect_logits"].items():
            index_values = indices.detach().cpu().tolist()
            predicted_values = logits.argmax(dim=-1).detach().cpu().tolist()
            true_values = batch["dialect"][indices].detach().cpu().tolist()
            self.dialect_pred[language].extend(predicted_values)
            self.dialect_true[language].extend(true_values)
            for index, predicted_value in zip(
                index_values,
                predicted_values,
                strict=True,
            ):
                dialect_predictions[index] = int(predicted_value)

        for index in range(len(lid_predictions)):
            true_lid = int(lid_truth[index])
            predicted_lid = int(lid_predictions[index])
            true_dialect = int(batch["dialect"][index].item())
            predicted_dialect = dialect_predictions[index]
            probabilities = lid_probabilities[index]
            maximum_probability = float(probabilities.max().item())
            true_probability = float(probabilities[true_lid].item())
            entropy = float(
                -(
                    probabilities.clamp_min(1e-12)
                    * probabilities.clamp_min(1e-12).log()
                ).sum().item()
            )
            self.rows.append(
                {
                    "utt_id": batch["utt_id"][index],
                    "filepath": batch["filepath"][index],
                    "oracle_language": batch["language"][index],
                    "true_language_index": true_lid,
                    "predicted_language_index": predicted_lid,
                    "lid_correct": int(predicted_lid == true_lid),
                    "lid_max_probability": maximum_probability,
                    "lid_true_probability": true_probability,
                    "lid_entropy": entropy,
                    "true_dialect_index": true_dialect,
                    "predicted_dialect_index": int(predicted_dialect),
                    "dialect_correct": int(predicted_dialect == true_dialect),
                    "true_gender_index": int(gender_truth[index]),
                    "predicted_gender_index": int(gender_predictions[index]),
                }
            )

    def _confidence_analysis(self) -> dict:
        edges = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
        bins = []
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            selected = [
                row
                for row in self.rows
                if lower <= float(row["lid_max_probability"]) < upper
            ]
            count = len(selected)
            bins.append(
                {
                    "lower": lower,
                    "upper": 1.0 if upper > 1.0 else upper,
                    "label": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                    "count": count,
                    "mean_lid_confidence": (
                        float(np.mean([row["lid_max_probability"] for row in selected]))
                        if count
                        else None
                    ),
                    "lid_accuracy": (
                        float(np.mean([row["lid_correct"] for row in selected]))
                        if count
                        else None
                    ),
                    "dialect_accuracy": (
                        float(np.mean([row["dialect_correct"] for row in selected]))
                        if count
                        else None
                    ),
                }
            )
        return {
            "bins": bins,
            "overall_lid_dialect_correct_correlation": self._binary_correlation(),
        }

    def _binary_correlation(self) -> float | None:
        if len(self.rows) < 2:
            return None
        lid = np.asarray([row["lid_correct"] for row in self.rows], dtype=float)
        dialect = np.asarray([row["dialect_correct"] for row in self.rows], dtype=float)
        if lid.std() == 0 or dialect.std() == 0:
            return None
        return float(np.corrcoef(lid, dialect)[0, 1])

    def finalize(self) -> tuple[dict, dict[str, np.ndarray]]:
        metrics: dict = {}
        matrices: dict[str, np.ndarray] = {}

        for name, truth, predictions, classes in (
            ("lid", self.lid_true, self.lid_pred, self.labels.num_lid_classes),
            (
                "gender",
                self.gender_true,
                self.gender_pred,
                self.labels.num_gender_classes,
            ),
        ):
            matrix = confusion_matrix(truth, predictions, classes)
            matrices[name] = matrix
            metrics.update(
                {
                    f"{name}_{metric_name}": value
                    for metric_name, value in classification_metrics(matrix).items()
                }
            )

        language_accuracies = []
        language_macro_f1s = []
        total_correct = 0
        total_examples = 0
        per_language = {}
        for language in self.labels.languages:
            matrix = confusion_matrix(
                self.dialect_true[language],
                self.dialect_pred[language],
                self.labels.num_dialect_classes(language),
            )
            matrices[f"dialect_{language}"] = matrix
            language_metrics = classification_metrics(matrix)
            per_language[language] = language_metrics
            language_accuracies.append(language_metrics["accuracy"])
            language_macro_f1s.append(language_metrics["macro_f1"])
            total_correct += int(np.diag(matrix).sum())
            total_examples += int(matrix.sum())

        metrics["dialect_per_language"] = per_language
        metrics["dialect_sample_weighted_accuracy"] = (
            float(total_correct / total_examples) if total_examples else 0.0
        )
        metrics["dialect_language_macro_accuracy"] = float(
            np.mean(language_accuracies)
        )
        metrics["dialect_language_macro_f1"] = float(
            np.mean(language_macro_f1s)
        )
        metrics["three_task_mean_macro_f1"] = float(
            np.mean(
                [
                    metrics["lid_macro_f1"],
                    metrics["dialect_language_macro_f1"],
                    metrics["gender_macro_f1"],
                ]
            )
        )
        metrics["lid_confidence_dialect_analysis"] = self._confidence_analysis()
        return metrics, matrices

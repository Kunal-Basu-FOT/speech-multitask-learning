from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import ClassificationResult, classification_metrics
from .utils import ensure_finite


@dataclass
class RidgeLinearProbe:
    classes_: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    weights_: np.ndarray
    regularization: float

    def predict(self, x: np.ndarray, batch_size: int = 32768) -> np.ndarray:
        predictions: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            block = np.asarray(x[start : start + batch_size], dtype=np.float32)
            standardized = (block - self.mean_) / self.scale_
            augmented = np.concatenate(
                [standardized, np.ones((len(block), 1), dtype=np.float32)], axis=1
            )
            scores = augmented @ self.weights_
            predictions.append(self.classes_[np.argmax(scores, axis=1)])
        return np.concatenate(predictions)


def _moments(x: np.ndarray, batch_size: int = 32768) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros(x.shape[1], dtype=np.float64)
    total_sq = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, len(x), batch_size):
        block = np.asarray(x[start : start + batch_size], dtype=np.float64)
        count += len(block)
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _sufficient_statistics(
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = x.shape[1] + 1
    xtx = np.zeros((dimension, dimension), dtype=np.float64)
    xty = np.zeros((dimension, len(classes)), dtype=np.float64)
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    for start in range(0, len(x), batch_size):
        block = np.asarray(x[start : start + batch_size], dtype=np.float32)
        standardized = (block - mean) / scale
        augmented = np.concatenate(
            [standardized, np.ones((len(block), 1), dtype=np.float32)], axis=1
        ).astype(np.float64)
        targets = np.zeros((len(block), len(classes)), dtype=np.float64)
        indices = np.fromiter(
            (class_to_index[value] for value in y[start : start + len(block)].tolist()),
            dtype=np.int64,
            count=len(block),
        )
        targets[np.arange(len(block)), indices] = 1.0
        xtx += augmented.T @ augmented
        xty += augmented.T @ targets
    return xtx, xty


def fit_ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    lambdas: Iterable[float],
) -> tuple[RidgeLinearProbe, ClassificationResult]:
    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("a linear probe requires at least two training classes")
    mean, scale = _moments(x_train)
    xtx, xty = _sufficient_statistics(x_train, y_train, classes, mean, scale)
    identity = np.eye(xtx.shape[0], dtype=np.float64)
    identity[-1, -1] = 0.0
    best_probe: RidgeLinearProbe | None = None
    best_result: ClassificationResult | None = None
    for regularization in lambdas:
        weights = np.linalg.solve(xtx + float(regularization) * identity, xty)
        probe = RidgeLinearProbe(
            classes_=classes,
            mean_=mean,
            scale_=scale,
            weights_=weights.astype(np.float32),
            regularization=float(regularization),
        )
        predictions = probe.predict(x_dev)
        result = classification_metrics(y_dev, predictions, classes)
        if best_result is None or (result.macro_f1, result.accuracy) > (
            best_result.macro_f1,
            best_result.accuracy,
        ):
            best_probe = probe
            best_result = result
    assert best_probe is not None and best_result is not None
    ensure_finite("probe weights", best_probe.weights_)
    return best_probe, best_result


def nested_stratified_indices(
    labels: np.ndarray,
    fractions: Iterable[float],
    seed: int,
) -> dict[float, np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups: dict[object, np.ndarray] = {}
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        groups[value] = rng.permutation(indices)
    result: dict[float, np.ndarray] = {}
    previous: set[int] = set()
    for fraction in sorted(float(value) for value in fractions):
        selected: list[int] = []
        for indices in groups.values():
            count = len(indices) if fraction >= 1.0 else max(1, int(np.floor(len(indices) * fraction)))
            selected.extend(indices[:count].tolist())
        selected_set = set(selected)
        if not previous.issubset(selected_set):
            raise RuntimeError("few-shot subsets are not nested")
        previous = selected_set
        result[fraction] = np.asarray(sorted(selected), dtype=np.int64)
    return result

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .checkpoint import load_model_state_only
from .config import Config
from .evaluate import result_directory
from .model import LogMelProjectedMultitaskECAPA
from .utils import atomic_json, count_csv_data_rows, utc_now


def _assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}: {value}")


def verify_experiment(config: Config) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_test = int(config.expected_rows["test"])
    expected_dev = int(config.expected_rows["dev"])

    cache_marker = config.path("logmel_cache_root") / "CACHE_COMPLETE.json"
    if not cache_marker.is_file():
        raise FileNotFoundError(cache_marker)
    checks.append({"check": "cache_complete", "path": str(cache_marker)})

    for domain in config.domains:
        for duration in config.durations:
            run = config.path("run_root") / domain / duration
            completion = run / "TRAINING_COMPLETE.json"
            history = run / "history.csv"
            best = run / "checkpoints/best.pt"
            last = run / "checkpoints/last.pt"
            for path in (completion, history, best, last, run / "best_dev_metrics.json"):
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing training artifact: {path}")
            history_frame = pd.read_csv(history)
            if len(history_frame) != int(config.epochs):
                raise RuntimeError(f"Expected {config.epochs} history rows: {history}")
            if not np.isfinite(history_frame.select_dtypes(include=[np.number]).to_numpy()).all():
                raise RuntimeError(f"History contains non-finite values: {history}")

            # Evaluation checkpoint loads model weights only; RNG state is intentionally ignored.
            model = LogMelProjectedMultitaskECAPA(
                int(config.feature_dimension), int(config.projected_dimension)
            )
            load_model_state_only(best, model)
            for name, tensor in model.state_dict().items():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Non-finite checkpoint tensor {name}: {best}")
            del model

            evaluation_specs: list[tuple[str, str, str]] = []
            evaluation_specs.append((domain, "dev", duration))
            for test_duration in config.durations:
                evaluation_specs.append((domain, "test", test_duration))
            other = "sixteen_bit" if domain == "one_bit" else "one_bit"
            evaluation_specs.append((other, "test", duration))

            for input_domain, split, test_duration in evaluation_specs:
                root = result_directory(
                    config,
                    train_domain=domain,
                    train_duration=duration,
                    input_domain=input_domain,
                    split=split,
                    test_duration=test_duration,
                )
                marker = root / "EVALUATION_COMPLETE.json"
                metrics = root / "metrics.json"
                if not marker.is_file() or not metrics.is_file():
                    raise RuntimeError(f"Missing evaluation marker or metrics: {root}")
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                _assert_finite_tree(payload)
                expected = expected_test if split == "test" else expected_dev
                if int(payload["examples"]) != expected:
                    raise RuntimeError(
                        f"Evaluation example mismatch at {root}: expected {expected}, got {payload['examples']}"
                    )
                if split == "test":
                    predictions = root / "predictions.csv"
                    if not predictions.is_file():
                        raise RuntimeError(f"Missing predictions: {predictions}")
                    rows = count_csv_data_rows(predictions)
                    if rows != expected_test:
                        raise RuntimeError(
                            f"Prediction row mismatch at {predictions}: expected {expected_test}, got {rows}"
                        )
            checks.append(
                {
                    "check": "condition_complete",
                    "domain": domain,
                    "duration": duration,
                    "history_rows": len(history_frame),
                }
            )

    report_marker = config.path("report_root") / "REPORT_COMPLETE.json"
    if not report_marker.is_file():
        raise FileNotFoundError(report_marker)
    report = json.loads(report_marker.read_text(encoding="utf-8"))
    for category in ("tables", "figures"):
        for name, raw_path in report[category].items():
            path = Path(raw_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing report artifact {name}: {path}")
    analysis = Path(report["analysis"])
    if not analysis.is_file() or analysis.stat().st_size == 0:
        raise RuntimeError(f"Missing report analysis: {analysis}")

    marker = {
        "status": "EXPERIMENT_COMPLETE",
        "created_utc": utc_now(),
        "experiment": "EXP6.9",
        "conditions": 8,
        "training_epochs_per_condition": int(config.epochs),
        "checks": checks,
        "report": report,
        "rng_safety": {
            "training_loads_map_location": "cpu",
            "evaluation_restores_rng": False,
            "rng_tensor_conversion": "cpu uint8 before torch.set_rng_state",
        },
    }
    destination = config.path("run_root") / "EXPERIMENT_COMPLETE.json"
    atomic_json(destination, marker)
    atomic_json(config.path("project_root") / "runtime/EXPERIMENT_COMPLETE.json", marker)
    print("EXP69_EXPERIMENT_COMPLETE", flush=True)
    return marker

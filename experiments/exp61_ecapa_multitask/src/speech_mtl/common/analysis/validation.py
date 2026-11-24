from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .confusion_plots import save_composite_confusion_figure
from .cross_precision import save_cross_precision_summary
from .curves import save_training_curves
from .report_tables import (
    save_cross_precision_table,
    save_experimental_configuration_table,
    save_matched_results_table,
    save_per_language_dialect_table,
    save_runtime_table,
)


_DIALECT_COUNTS = {
    "bh": 3,
    "bn": 5,
    "ch": 4,
    "kn": 5,
    "mg": 4,
    "mr": 4,
    "mt": 4,
    "te": 4,
}


def _classification(
    classes: int,
    examples: int,
    accuracy: float,
    macro_f1: float,
) -> dict[str, Any]:
    return {
        "examples": examples,
        "accuracy": accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_f1": macro_f1,
        "per_class_precision": [macro_f1] * classes,
        "per_class_recall": [macro_f1] * classes,
        "per_class_f1": [macro_f1] * classes,
        "support": [examples // classes] * classes,
    }


def _confusion(classes: int) -> dict[str, Any]:
    counts = np.eye(classes, dtype=np.int64) * 10
    row_percent = np.eye(classes, dtype=np.float64) * 100.0

    return {
        "labels": [str(index) for index in range(classes)],
        "counts": counts.tolist(),
        "row_percent": row_percent.tolist(),
    }


def _evaluation(score: float) -> dict[str, Any]:
    dialect_per_language = {
        language: _classification(
            classes,
            100,
            score,
            score,
        )
        for language, classes in _DIALECT_COUNTS.items()
    }

    return {
        "examples": 800,
        "elapsed_seconds": 12.5,
        "summary": {
            "language": _classification(8, 800, score, score),
            "gender": _classification(2, 800, score, score),
            "dialect": {
                "examples": 800,
                "sample_weighted_accuracy": score,
                "language_macro_accuracy": score,
                "language_macro_f1": score,
                "per_language": dialect_per_language,
            },
            "selection_metric": {
                "name": "three_task_mean_macro_f1",
                "value": score,
                "components": {
                    "language_macro_f1": score,
                    "dialect_language_macro_f1": score,
                    "gender_macro_f1": score,
                },
            },
        },
        "confusion_matrices": {
            "language": _confusion(8),
            "gender": _confusion(2),
            "dialect": {
                language: _confusion(classes)
                for language, classes in _DIALECT_COUNTS.items()
            },
        },
    }


def validate_analysis_contract() -> dict[str, Any]:
    history = pd.DataFrame(
        {
            "epoch": list(range(1, 41)),
            "train_loss": np.linspace(3.0, 0.5, 40),
            "dev_loss": np.linspace(3.2, 0.7, 40),
            "train_three_task_mean_accuracy": np.linspace(
                0.2, 0.9, 40
            ),
            "dev_three_task_mean_accuracy": np.linspace(
                0.18, 0.85, 40
            ),
            "dev_language_accuracy": np.linspace(
                0.2, 0.88, 40
            ),
            "dev_dialect_language_macro_accuracy": np.linspace(
                0.15, 0.80, 40
            ),
            "dev_gender_accuracy": np.linspace(
                0.3, 0.92, 40
            ),
        }
    )

    one_bit = _evaluation(0.72)
    sixteen_bit = _evaluation(0.78)
    cross_1_to_16 = _evaluation(0.66)
    cross_16_to_1 = _evaluation(0.69)

    evaluations = {
        "1-bit → 1-bit": one_bit,
        "1-bit → 16-bit": cross_1_to_16,
        "16-bit → 16-bit": sixteen_bit,
        "16-bit → 1-bit": cross_16_to_1,
    }

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        generated = []

        generated.extend(
            save_training_curves(
                history,
                loss_output=root / "loss.png",
                accuracy_output=root / "accuracy.png",
                title_prefix="Synthetic 1-bit",
            ).values()
        )

        generated.append(
            save_composite_confusion_figure(
                one_bit["confusion_matrices"],
                output_path=root / "confusion.png",
                title="Synthetic confusion matrices",
            )
        )

        generated.append(
            save_cross_precision_summary(
                evaluations,
                output_path=root / "cross_precision.png",
            )
        )

        generated.append(
            save_experimental_configuration_table(
                {
                    "model": {
                        "encoder": "ECAPA-TDNN",
                        "embedding": 192,
                    },
                    "training": {
                        "epochs": 40,
                        "schedule": [5, 30, 5],
                    },
                },
                output_path=root / "configuration.csv",
            )
        )

        generated.append(
            save_matched_results_table(
                one_bit_evaluation=one_bit,
                sixteen_bit_evaluation=sixteen_bit,
                output_path=root / "matched.csv",
            )
        )

        generated.append(
            save_per_language_dialect_table(
                one_bit_evaluation=one_bit,
                sixteen_bit_evaluation=sixteen_bit,
                output_path=root / "dialect.csv",
            )
        )

        generated.append(
            save_cross_precision_table(
                evaluations,
                output_path=root / "cross_precision.csv",
            )
        )

        generated.append(
            save_runtime_table(
                [
                    {
                        "run": "synthetic",
                        "examples": 800,
                        "elapsed_seconds": 12.5,
                        "examples_per_second": 64.0,
                    }
                ],
                output_path=root / "runtime.csv",
            )
        )

        missing = [
            path
            for path in generated
            if not Path(path).is_file()
        ]

        if missing:
            raise RuntimeError(
                f"Expected analysis artifacts were not created: {missing}"
            )

        empty = [
            path
            for path in generated
            if Path(path).stat().st_size == 0
        ]

        if empty:
            raise RuntimeError(
                f"Generated analysis artifacts are empty: {empty}"
            )

        artifact_names = sorted(
            Path(path).name for path in generated
        )

    return {
        "status": "PASS",
        "generated_artifact_count": len(generated),
        "generated_artifacts": artifact_names,
        "figures": 4,
        "tables": 5,
        "curve_contract": "PASS",
        "confusion_contract": "PASS",
        "cross_precision_contract": "PASS",
    }

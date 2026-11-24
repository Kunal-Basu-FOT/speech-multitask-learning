from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from speech_mtl.common.data.collate import pad_cached_feature_batch
from speech_mtl.common.models.factory import build_exp61_model

from .evaluator import evaluate_model


class _SyntheticEvaluationDataset(Dataset):
    def __init__(self) -> None:
        self.languages = [
            "bh", "bn", "ch", "kn",
            "mg", "mr", "mt", "te",
        ]
        self.language_indices = {
            "bn": 0,
            "bh": 1,
            "mg": 2,
            "mt": 3,
            "mr": 4,
            "kn": 5,
            "te": 6,
            "ch": 7,
        }
        self.dialect_counts = {
            "bh": 3,
            "bn": 5,
            "ch": 4,
            "kn": 5,
            "mg": 4,
            "mr": 4,
            "mt": 4,
            "te": 4,
        }

    def __len__(self) -> int:
        return len(self.languages) * 2

    def __getitem__(self, index: int):
        language = self.languages[index % len(self.languages)]
        num_frames = 40 + index
        generator = torch.Generator().manual_seed(1000 + index)

        return {
            "features": torch.randn(
                num_frames,
                768,
                generator=generator,
            ),
            "utt_id": f"synthetic_{index}",
            "num_frames": num_frames,
            "lid": self.language_indices[language],
            "dialect": index % self.dialect_counts[language],
            "gender": index % 2,
            "language": language,
            "filepath": f"/tmp/synthetic_{index}.wav",
            "audio_samples": num_frames * 320,
            "dataset_index": index,
        }


def validate_evaluation_contract() -> dict[str, Any]:
    torch.manual_seed(1337)

    model = build_exp61_model()
    dataset = _SyntheticEvaluationDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=pad_cached_feature_batch,
        num_workers=0,
    )

    with tempfile.TemporaryDirectory() as directory:
        predictions_path = (
            Path(directory) / "predictions.csv"
        )

        result = evaluate_model(
            model=model,
            dataloader=dataloader,
            device="cpu",
            predictions_path=predictions_path,
            use_amp=False,
            show_progress_bar=False,
        )

        if result.examples != len(dataset):
            raise RuntimeError(
                f"Expected {len(dataset)} examples, "
                f"received {result.examples}"
            )

        if not predictions_path.is_file():
            raise RuntimeError(
                "Prediction CSV was not created"
            )

        prediction_lines = (
            predictions_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )

        if len(prediction_lines) != len(dataset) + 1:
            raise RuntimeError(
                "Prediction CSV row count is incorrect"
            )

    selection_metric = result.summary[
        "selection_metric"
    ]

    if selection_metric["name"] != (
        "three_task_mean_macro_f1"
    ):
        raise RuntimeError(
            "Unexpected selection metric"
        )

    if not 0.0 <= selection_metric["value"] <= 1.0:
        raise RuntimeError(
            "Selection metric is outside [0, 1]"
        )

    for task in ("language", "gender"):
        if not 0.0 <= result.summary[task]["accuracy"] <= 1.0:
            raise RuntimeError(
                f"{task} accuracy is outside [0, 1]"
            )

    dialect = result.summary["dialect"]

    if set(dialect["per_language"]) != {
        "bh", "bn", "ch", "kn",
        "mg", "mr", "mt", "te",
    }:
        raise RuntimeError(
            "Dialect per-language metrics are incomplete"
        )

    language_matrix = result.confusion_matrices[
        "language"
    ]["counts"]

    if sum(sum(row) for row in language_matrix) != len(dataset):
        raise RuntimeError(
            "Language confusion-matrix count mismatch"
        )

    dialect_total = sum(
        sum(sum(row) for row in item["counts"])
        for item in result.confusion_matrices[
            "dialect"
        ].values()
    )

    if dialect_total != len(dataset):
        raise RuntimeError(
            "Dialect confusion matrices do not cover "
            "the entire evaluation set"
        )

    return {
        "status": "PASS",
        "examples": result.examples,
        "selection_metric": selection_metric,
        "language_accuracy": result.summary[
            "language"
        ]["accuracy"],
        "gender_accuracy": result.summary[
            "gender"
        ]["accuracy"],
        "dialect_language_macro_accuracy": dialect[
            "language_macro_accuracy"
        ],
        "dialect_language_macro_f1": dialect[
            "language_macro_f1"
        ],
        "confusion_matrix_contract": "PASS",
        "prediction_csv_contract": "PASS",
    }

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .confusion import ConfusionMatrixBundle
from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_multitask_summary,
)
from .predictions import PredictionWriter


DIALECT_CLASS_COUNTS = {
    "bh": 3,
    "bn": 5,
    "ch": 4,
    "kn": 5,
    "mg": 4,
    "mr": 4,
    "mt": 4,
    "te": 4,
}


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, Any]
    confusion_matrices: dict[str, Any]
    examples: int
    elapsed_seconds: float
    predictions_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "confusion_matrices": self.confusion_matrices,
            "examples": self.examples,
            "elapsed_seconds": self.elapsed_seconds,
            "predictions_path": self.predictions_path,
        }


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, Any]:
    moved = dict(batch)

    for key in (
        "features",
        "relative_lengths",
        "valid_mask",
        "lid",
        "dialect",
        "gender",
    ):
        moved[key] = batch[key].to(
            device=device,
            non_blocking=non_blocking,
        )

    return moved


def _append_cpu_ints(
    destination: list[int],
    tensor: torch.Tensor,
) -> None:
    destination.extend(
        int(value)
        for value in tensor.detach().cpu().tolist()
    )


def evaluate_model(
    *,
    model,
    dataloader,
    device: torch.device | str,
    predictions_path: Path | str | None = None,
    use_amp: bool = True,
    non_blocking: bool = True,
    show_progress_bar: bool = True,
) -> EvaluationResult:
    """Evaluate language, oracle dialect, and gender tasks."""

    device = torch.device(device)
    model.eval()

    amp_enabled = bool(
        use_amp and device.type == "cuda"
    )

    true_language: list[int] = []
    predicted_language: list[int] = []
    true_gender: list[int] = []
    predicted_gender: list[int] = []

    true_dialect: dict[str, list[int]] = {
        language: [] for language in DIALECT_CLASS_COUNTS
    }
    predicted_dialect: dict[str, list[int]] = {
        language: [] for language in DIALECT_CLASS_COUNTS
    }

    total_examples = 0
    started = time.monotonic()

    writer_context = (
        PredictionWriter(predictions_path)
        if predictions_path is not None
        else None
    )

    if writer_context is None:
        context_manager = _NullPredictionWriter()
    else:
        context_manager = writer_context

    with context_manager as prediction_writer:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="evaluation",
            dynamic_ncols=True,
            disable=not show_progress_bar,
            leave=True,
        )

        with torch.inference_mode():
            for raw_batch in iterator:
                batch = _move_batch(
                    raw_batch,
                    device,
                    non_blocking=non_blocking,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(
                        features=batch["features"],
                        relative_lengths=batch["relative_lengths"],
                        oracle_languages=batch["language"],
                        valid_mask=batch["valid_mask"],
                    )

                language_predictions = torch.argmax(
                    output.language_logits,
                    dim=1,
                )
                gender_predictions = torch.argmax(
                    output.gender_logits,
                    dim=1,
                )

                _append_cpu_ints(true_language, batch["lid"])
                _append_cpu_ints(
                    predicted_language,
                    language_predictions,
                )
                _append_cpu_ints(true_gender, batch["gender"])
                _append_cpu_ints(
                    predicted_gender,
                    gender_predictions,
                )

                batch_size = int(batch["features"].shape[0])
                total_examples += batch_size

                dialect_predictions_for_batch = torch.empty(
                    batch_size,
                    dtype=torch.long,
                    device=device,
                )

                for language, logits in output.dialect_logits.items():
                    indices = output.dialect_batch_indices[language]
                    local_predictions = torch.argmax(
                        logits,
                        dim=1,
                    )
                    local_targets = batch["dialect"].index_select(
                        0,
                        indices,
                    )

                    dialect_predictions_for_batch.index_copy_(
                        0,
                        indices,
                        local_predictions,
                    )

                    _append_cpu_ints(
                        true_dialect[language],
                        local_targets,
                    )
                    _append_cpu_ints(
                        predicted_dialect[language],
                        local_predictions,
                    )

                if prediction_writer is not None:
                    language_true_cpu = batch["lid"].detach().cpu()
                    language_pred_cpu = (
                        language_predictions.detach().cpu()
                    )
                    dialect_true_cpu = batch["dialect"].detach().cpu()
                    dialect_pred_cpu = (
                        dialect_predictions_for_batch.detach().cpu()
                    )
                    gender_true_cpu = batch["gender"].detach().cpu()
                    gender_pred_cpu = (
                        gender_predictions.detach().cpu()
                    )

                    rows = []

                    for index in range(batch_size):
                        rows.append(
                            {
                                "utt_id": batch["utt_id"][index],
                                "filepath": batch["filepath"][index],
                                "oracle_language": (
                                    batch["language"][index]
                                ),
                                "true_language_index": int(
                                    language_true_cpu[index]
                                ),
                                "predicted_language_index": int(
                                    language_pred_cpu[index]
                                ),
                                "true_dialect_index": int(
                                    dialect_true_cpu[index]
                                ),
                                "predicted_dialect_index": int(
                                    dialect_pred_cpu[index]
                                ),
                                "true_gender_index": int(
                                    gender_true_cpu[index]
                                ),
                                "predicted_gender_index": int(
                                    gender_pred_cpu[index]
                                ),
                            }
                        )

                    prediction_writer.write_batch(rows)

                iterator.set_postfix(
                    examples=total_examples,
                )

    language_metrics = compute_classification_metrics(
        true_language,
        predicted_language,
        number_of_classes=8,
    )
    gender_metrics = compute_classification_metrics(
        true_gender,
        predicted_gender,
        number_of_classes=2,
    )

    dialect_metrics_by_language: dict[
        str, ClassificationMetrics
    ] = {}

    for language, class_count in DIALECT_CLASS_COUNTS.items():
        if len(true_dialect[language]) == 0:
            raise RuntimeError(
                f"No evaluation examples were routed to {language}"
            )

        dialect_metrics_by_language[language] = (
            compute_classification_metrics(
                true_dialect[language],
                predicted_dialect[language],
                number_of_classes=class_count,
            )
        )

    summary = compute_multitask_summary(
        language_metrics=language_metrics,
        gender_metrics=gender_metrics,
        dialect_metrics_by_language=(
            dialect_metrics_by_language
        ),
    )

    language_labels = [
        str(index) for index in range(8)
    ]
    gender_labels = [
        str(index) for index in range(2)
    ]

    confusion_matrices: dict[str, Any] = {
        "language": ConfusionMatrixBundle.from_predictions(
            true_language,
            predicted_language,
            language_labels,
        ).to_dict(),
        "gender": ConfusionMatrixBundle.from_predictions(
            true_gender,
            predicted_gender,
            gender_labels,
        ).to_dict(),
        "dialect": {},
    }

    for language, class_count in DIALECT_CLASS_COUNTS.items():
        labels = [
            f"D{index + 1}"
            for index in range(class_count)
        ]

        confusion_matrices["dialect"][language] = (
            ConfusionMatrixBundle.from_predictions(
                true_dialect[language],
                predicted_dialect[language],
                labels,
            ).to_dict()
        )

    elapsed_seconds = time.monotonic() - started

    return EvaluationResult(
        summary=summary,
        confusion_matrices=confusion_matrices,
        examples=total_examples,
        elapsed_seconds=elapsed_seconds,
        predictions_path=(
            str(Path(predictions_path))
            if predictions_path is not None
            else None
        ),
    )


class _NullPredictionWriter:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return None


def write_evaluation_result(
    result: EvaluationResult,
    path: Path | str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

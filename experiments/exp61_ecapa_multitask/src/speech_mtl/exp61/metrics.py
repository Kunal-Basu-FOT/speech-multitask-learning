from __future__ import annotations

from typing import Any

import torch

from speech_mtl.common.evaluation.confusion import (
    ConfusionMatrixBundle,
)
from speech_mtl.common.evaluation.metrics import (
    compute_classification_metrics,
    compute_multitask_summary,
)


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


class EpochMetricAccumulator:
    """Accumulate predictions without storing logits or embeddings."""

    def __init__(self) -> None:
        self.true_language: list[int] = []
        self.predicted_language: list[int] = []
        self.true_gender: list[int] = []
        self.predicted_gender: list[int] = []
        self.true_dialect = {
            language: [] for language in DIALECT_CLASS_COUNTS
        }
        self.predicted_dialect = {
            language: [] for language in DIALECT_CLASS_COUNTS
        }
        self.examples = 0

    @staticmethod
    def _extend(
        destination: list[int],
        tensor: torch.Tensor,
    ) -> None:
        destination.extend(
            int(value)
            for value in tensor.detach().cpu().tolist()
        )

    def update(
        self,
        *,
        output,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        batch_size = int(output.language_logits.shape[0])

        language_predictions = torch.argmax(
            output.language_logits.detach(),
            dim=1,
        )
        gender_predictions = torch.argmax(
            output.gender_logits.detach(),
            dim=1,
        )

        self._extend(self.true_language, batch["lid"])
        self._extend(
            self.predicted_language,
            language_predictions,
        )
        self._extend(self.true_gender, batch["gender"])
        self._extend(
            self.predicted_gender,
            gender_predictions,
        )

        dialect_predictions_for_batch = torch.empty(
            batch_size,
            dtype=torch.long,
            device=language_predictions.device,
        )

        routed = 0

        for language, logits in output.dialect_logits.items():
            indices = output.dialect_batch_indices[language]
            local_predictions = torch.argmax(
                logits.detach(),
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

            self._extend(
                self.true_dialect[language],
                local_targets,
            )
            self._extend(
                self.predicted_dialect[language],
                local_predictions,
            )
            routed += int(indices.numel())

        if routed != batch_size:
            raise RuntimeError(
                f"Dialect routing covered {routed} of "
                f"{batch_size} examples"
            )

        self.examples += batch_size
        return dialect_predictions_for_batch

    def finalize(self) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        language_metrics = compute_classification_metrics(
            self.true_language,
            self.predicted_language,
            number_of_classes=8,
        )
        gender_metrics = compute_classification_metrics(
            self.true_gender,
            self.predicted_gender,
            number_of_classes=2,
        )

        dialect_metrics = {}

        for language, class_count in DIALECT_CLASS_COUNTS.items():
            if not self.true_dialect[language]:
                raise RuntimeError(
                    f"No examples accumulated for language {language}"
                )

            dialect_metrics[language] = (
                compute_classification_metrics(
                    self.true_dialect[language],
                    self.predicted_dialect[language],
                    number_of_classes=class_count,
                )
            )

        summary = compute_multitask_summary(
            language_metrics=language_metrics,
            gender_metrics=gender_metrics,
            dialect_metrics_by_language=dialect_metrics,
        )

        confusion_matrices: dict[str, Any] = {
            "language": (
                ConfusionMatrixBundle.from_predictions(
                    self.true_language,
                    self.predicted_language,
                    [str(index) for index in range(8)],
                ).to_dict()
            ),
            "gender": (
                ConfusionMatrixBundle.from_predictions(
                    self.true_gender,
                    self.predicted_gender,
                    ["0", "1"],
                ).to_dict()
            ),
            "dialect": {},
        }

        for language, class_count in DIALECT_CLASS_COUNTS.items():
            confusion_matrices["dialect"][language] = (
                ConfusionMatrixBundle.from_predictions(
                    self.true_dialect[language],
                    self.predicted_dialect[language],
                    [
                        f"D{index + 1}"
                        for index in range(class_count)
                    ],
                ).to_dict()
            )

        if language_metrics.examples != self.examples:
            raise RuntimeError(
                "Language metrics do not cover all accumulated examples"
            )

        return summary, confusion_matrices

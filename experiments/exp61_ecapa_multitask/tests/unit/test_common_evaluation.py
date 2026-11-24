from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from speech_mtl.common.evaluation.confusion import (
    ConfusionMatrixBundle,
    confusion_matrix,
    row_normalized_percent,
)
from speech_mtl.common.evaluation.metrics import (
    compute_classification_metrics,
    compute_multitask_summary,
)
from speech_mtl.common.evaluation.predictions import (
    PredictionWriter,
)


class CommonEvaluationTests(unittest.TestCase):
    def test_confusion_matrix_orientation(self) -> None:
        true = [0, 0, 1, 1]
        predicted = [0, 1, 0, 1]

        matrix = confusion_matrix(
            true,
            predicted,
            number_of_classes=2,
        )

        np.testing.assert_array_equal(
            matrix,
            np.array([[1, 1], [1, 1]]),
        )

    def test_row_normalization(self) -> None:
        matrix = np.array(
            [[3, 1], [0, 0]],
            dtype=np.int64,
        )

        normalized = row_normalized_percent(matrix)

        np.testing.assert_allclose(
            normalized[0],
            np.array([75.0, 25.0]),
        )
        np.testing.assert_allclose(
            normalized[1],
            np.array([0.0, 0.0]),
        )

    def test_macro_f1_includes_absent_class(self) -> None:
        metrics = compute_classification_metrics(
            [0, 0],
            [0, 0],
            number_of_classes=2,
        )

        self.assertEqual(metrics.accuracy, 1.0)
        self.assertAlmostEqual(metrics.macro_f1, 0.5)

    def test_multitask_selection_metric(self) -> None:
        language = compute_classification_metrics(
            [0, 1],
            [0, 1],
            number_of_classes=2,
        )
        gender = compute_classification_metrics(
            [0, 1],
            [0, 1],
            number_of_classes=2,
        )

        dialect = {
            language_code: compute_classification_metrics(
                [0, 1],
                [0, 1],
                number_of_classes=2,
            )
            for language_code in (
                "bh", "bn", "ch", "kn",
                "mg", "mr", "mt", "te",
            )
        }

        summary = compute_multitask_summary(
            language_metrics=language,
            gender_metrics=gender,
            dialect_metrics_by_language=dialect,
        )

        self.assertEqual(
            summary["selection_metric"]["value"],
            1.0,
        )

    def test_prediction_writer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"

            row = {
                "utt_id": "1",
                "filepath": "/tmp/1.wav",
                "oracle_language": "bn",
                "true_language_index": 0,
                "predicted_language_index": 1,
                "true_dialect_index": 2,
                "predicted_dialect_index": 3,
                "true_gender_index": 0,
                "predicted_gender_index": 1,
            }

            with PredictionWriter(path) as writer:
                writer.write_batch([row])

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(len(lines), 2)

    def test_confusion_bundle(self) -> None:
        bundle = ConfusionMatrixBundle.from_predictions(
            [0, 1],
            [0, 1],
            ["A", "B"],
        )

        self.assertEqual(bundle.labels, ("A", "B"))
        self.assertEqual(bundle.counts.sum(), 2)
        self.assertEqual(bundle.row_percent.sum(), 200.0)


if __name__ == "__main__":
    unittest.main()

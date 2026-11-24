from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from speech_mtl.common.analysis.confusion_plots import (
    save_composite_confusion_figure,
)
from speech_mtl.common.analysis.cross_precision import (
    cross_precision_frame,
)
from speech_mtl.common.analysis.curves import (
    save_training_curves,
)


class CommonAnalysisTests(unittest.TestCase):
    def test_training_curve_files(self) -> None:
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [3.0, 2.0, 1.0],
                "dev_loss": [3.1, 2.1, 1.2],
                "train_three_task_mean_accuracy": [
                    0.2, 0.4, 0.6
                ],
                "dev_three_task_mean_accuracy": [
                    0.18, 0.35, 0.55
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            result = save_training_curves(
                history,
                loss_output=Path(directory) / "loss.png",
                accuracy_output=Path(directory) / "accuracy.png",
                title_prefix="Test",
                dpi=72,
            )

            for path in result.values():
                self.assertTrue(Path(path).is_file())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_cross_precision_order(self) -> None:
        def evaluation(score: float):
            return {
                "summary": {
                    "language": {"accuracy": score},
                    "gender": {"accuracy": score},
                    "dialect": {
                        "language_macro_accuracy": score
                    },
                    "selection_metric": {"value": score},
                }
            }

        frame = cross_precision_frame(
            {
                "1-bit → 1-bit": evaluation(0.7),
                "1-bit → 16-bit": evaluation(0.6),
                "16-bit → 16-bit": evaluation(0.8),
                "16-bit → 1-bit": evaluation(0.65),
            }
        )

        self.assertEqual(
            frame["condition"].tolist(),
            [
                "1-bit → 1-bit",
                "1-bit → 16-bit",
                "16-bit → 16-bit",
                "16-bit → 1-bit",
            ],
        )

    def test_composite_confusion_file(self) -> None:
        def item(classes: int):
            return {
                "labels": [str(index) for index in range(classes)],
                "counts": np.eye(classes, dtype=int).tolist(),
                "row_percent": (
                    np.eye(classes, dtype=float) * 100.0
                ).tolist(),
            }

        confusion = {
            "language": item(8),
            "gender": item(2),
            "dialect": {
                "bh": item(3),
                "bn": item(5),
                "ch": item(4),
                "kn": item(5),
                "mg": item(4),
                "mr": item(4),
                "mt": item(4),
                "te": item(4),
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = save_composite_confusion_figure(
                confusion,
                output_path=Path(directory) / "cm.png",
                title="Test",
                dpi=72,
            )

            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

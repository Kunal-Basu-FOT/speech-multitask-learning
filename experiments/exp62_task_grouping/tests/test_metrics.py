import unittest
import numpy as np
from speech_mtl.exp62.metrics import classification_metrics, confusion_matrix


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        matrix = confusion_matrix([0, 1, 1], [0, 1, 1], 2)
        metrics = classification_metrics(matrix)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_zero_support_is_finite(self):
        matrix = np.array([[2, 0], [0, 0]])
        metrics = classification_metrics(matrix)
        self.assertTrue(np.isfinite(metrics["macro_f1"]))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from exp65_68.encoder import map_encoder_state
from exp65_68.metrics import classification_metrics
from exp65_68.probe import fit_ridge_probe, nested_stratified_indices


class CoreTests(unittest.TestCase):
    def test_metrics(self):
        result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertAlmostEqual(result.accuracy, 0.75)
        self.assertGreater(result.macro_f1, 0.7)

    def test_ridge_probe(self):
        rng = np.random.default_rng(1337)
        x = rng.normal(size=(300, 8)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
        probe, dev = fit_ridge_probe(x[:200], y[:200], x[200:250], y[200:250], [0.01, 0.1, 1.0])
        predictions = probe.predict(x[250:])
        result = classification_metrics(y[250:], predictions)
        self.assertGreater(dev.accuracy, 0.8)
        self.assertGreater(result.accuracy, 0.8)

    def test_nested_subsets(self):
        labels = np.array([f"s{i % 4}" for i in range(100)])
        subsets = nested_stratified_indices(labels, [0.1, 0.25, 0.5, 1.0], 1337)
        self.assertTrue(set(subsets[0.1]).issubset(set(subsets[0.25])))
        self.assertTrue(set(subsets[0.25]).issubset(set(subsets[0.5])))
        self.assertEqual(len(subsets[1.0]), 100)

    def test_encoder_mapping(self):
        target = {"a.weight": torch.zeros(2, 3), "b.bias": torch.zeros(2)}
        source = {"encoder.a.weight": torch.ones(2, 3), "encoder.b.bias": torch.ones(2), "head.weight": torch.ones(1)}
        mapped = map_encoder_state(source, target)
        self.assertEqual(set(mapped), set(target))
        self.assertTrue(torch.equal(mapped["a.weight"], torch.ones(2, 3)))


if __name__ == "__main__":
    unittest.main()

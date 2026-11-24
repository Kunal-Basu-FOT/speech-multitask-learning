import unittest
import torch
from speech_mtl.exp62.conditions import CONDITION_ORDER, get_condition
from speech_mtl.exp62.config import ModelConfig
from speech_mtl.exp62.losses import compute_losses
from speech_mtl.exp62.model import ReducedMTLModel
from speech_mtl.exp62.validate import synthetic_batch, synthetic_label_space


class SyntheticModelTests(unittest.TestCase):
    def test_all_conditions_forward_backward(self):
        labels = synthetic_label_space()
        batch = synthetic_batch(labels, torch.device("cpu"))
        for name in CONDITION_ORDER:
            model = ReducedMTLModel(ModelConfig(), labels, get_condition(name))
            self.assertEqual(sum(p.numel() for p in model.ecapa.parameters()), 7_212_660)
            output = model(batch)
            losses = compute_losses(model, batch, output)
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()


if __name__ == "__main__":
    unittest.main()

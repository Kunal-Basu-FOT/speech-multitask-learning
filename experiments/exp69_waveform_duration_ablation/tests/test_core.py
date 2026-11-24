from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exp69.cache import sign_quantize
from exp69.checkpoint import _cpu_byte_tensor, capture_rng_state, restore_rng_state, load_model_state_only
from exp69.metrics import classification_metrics, multitask_metrics
from exp69.schedule import Exp69LRScheduler, phase_for_epoch
from exp69.config import Config
from exp69.utils import frame_cap


class CoreTests(unittest.TestCase):
    def test_duration_frame_caps(self):
        self.assertEqual(frame_cap(3.0, 16000, 400, 160), 298)
        self.assertEqual(frame_cap(7.0, 16000, 400, 160), 698)
        self.assertEqual(frame_cap(9.0, 16000, 400, 160), 898)
        self.assertIsNone(frame_cap(None, 16000, 400, 160))

    def test_sign_quantizer_exact_boundary(self):
        waveform = torch.tensor([-0.2, -0.0, 0.0, 0.1])
        result = sign_quantize(waveform)
        self.assertTrue(torch.equal(result, torch.tensor([-1.0, -1.0, -1.0, 1.0])))

    def test_rng_tensor_is_cpu_uint8(self):
        value = torch.arange(16, dtype=torch.int64)
        result = _cpu_byte_tensor(value)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.uint8)
        self.assertTrue(result.is_contiguous())

    def test_evaluation_loader_ignores_rng(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "rng_state": {"torch_cpu": torch.arange(12, dtype=torch.int64)},
                    "training_state": {"completed_epoch": 40},
                },
                path,
            )
            target = torch.nn.Linear(3, 2)
            metadata = load_model_state_only(path, target)
            self.assertEqual(metadata["training_state"]["completed_epoch"], 40)
            for left, right in zip(model.parameters(), target.parameters()):
                self.assertTrue(torch.equal(left, right))


    def test_rng_capture_restore_roundtrip_cpu(self):
        torch.manual_seed(123)
        state = capture_rng_state()
        expected = torch.rand(4)
        restore_rng_state(state)
        actual = torch.rand(4)
        self.assertTrue(torch.equal(expected, actual))

    def test_phase_schedule_contract(self):
        config = Config({
            "phase1_epochs": 5,
            "phase2_end_epoch": 35,
            "warmup_start_lr": 1e-5,
            "peak_lr": 1e-4,
            "cosine_end_lr": 5e-6,
            "phase3_task_lr": 2e-5,
        })
        self.assertEqual(phase_for_epoch(1, config).name, "phase1_warmup_joint")
        self.assertEqual(phase_for_epoch(35, config).name, "phase2_full_joint")
        self.assertEqual(phase_for_epoch(36, config).name, "phase3_heads_only")
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        other = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.AdamW([{"params": [parameter]}, {"params": [other]}])
        scheduler = Exp69LRScheduler(optimizer, steps_per_epoch=10, config=config)
        encoder_lr, task_lr = scheduler.learning_rates(optimizer_step=0, epoch=1)
        self.assertAlmostEqual(encoder_lr, 1e-5)
        self.assertAlmostEqual(task_lr, 1e-5)
        encoder_lr, task_lr = scheduler.learning_rates(optimizer_step=999999, epoch=36)
        self.assertEqual(encoder_lr, 0.0)
        self.assertAlmostEqual(task_lr, 2e-5)

    def test_classification_metrics(self):
        result = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            2,
        )
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(np.asarray(result["confusion_counts"]).shape, (2, 2))

    def test_multitask_metric_contract(self):
        languages = np.array(["bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"])
        labels = np.arange(8)
        result = multitask_metrics(
            true_lid=labels,
            pred_lid=labels,
            true_gender=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            pred_gender=np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            true_dialect=np.zeros(8, dtype=np.int64),
            pred_dialect=np.zeros(8, dtype=np.int64),
            languages=languages,
            dialect_class_counts={language: 1 for language in languages},
        )
        self.assertAlmostEqual(result["three_task_mean_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

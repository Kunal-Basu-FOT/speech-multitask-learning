from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from speech_mtl.exp61.batching import (
    StableFrameBudgetBatchSampler,
)
from speech_mtl.exp61.config import load_exp61_config


class Exp61IntegrationTests(unittest.TestCase):
    def test_stable_sampler_count_and_coverage(self) -> None:
        lengths = [10, 11, 12, 50, 51, 52, 100]
        sampler = StableFrameBudgetBatchSampler(
            lengths,
            max_batch_frames=120,
            max_batch_size=4,
            number_of_buckets=3,
            shuffle=True,
            seed=1337,
        )

        counts = []
        orders = []

        for epoch in (1, 2, 3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            counts.append(len(batches))
            orders.append(batches)

            flattened = sorted(
                index
                for batch in batches
                for index in batch
            )
            self.assertEqual(
                flattened,
                list(range(len(lengths))),
            )

        self.assertEqual(len(set(counts)), 1)
        self.assertNotEqual(orders[0], orders[1])

    def test_config_contract(self) -> None:
        text = """
paths:
  project_root: /tmp/project
  cache_root: /tmp/cache
  run_root: /tmp/run
  results_root: /tmp/results
  report_root: /tmp/report
runtime:
  physical_gpu: 1
  seed: 1337
model:
  feature_dimension: 768
data:
  expected_split_counts:
    train: 146359
    dev: 11507
    test: 17550
  max_batch_frames: 12000
  max_batch_size: 32
  number_of_buckets: 40
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
training:
  total_epochs: 40
  gradient_accumulation_steps: 1
  maximum_gradient_norm: 1.0
  amp_enabled: true
  progress_interval_batches: 50
optimizer:
  peak_learning_rate: 0.0001
  weight_decay: 0.0001
scheduler:
  warmup_start_learning_rate: 0.00001
  cosine_minimum_learning_rate: 0.000005
  heads_only_learning_rate: 0.00002
""".lstrip()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            config = load_exp61_config(path)

        self.assertEqual(config.physical_gpu, 1)
        self.assertEqual(config.total_epochs, 40)
        self.assertEqual(config.feature_dimension, 768)


if __name__ == "__main__":
    unittest.main()

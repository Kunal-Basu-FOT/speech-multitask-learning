from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from speech_mtl.common.data import (
    CachedFeatureDataset,
    FrameBudgetBatchSampler,
    pad_cached_feature_batch,
)


class CommonDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.split_root = self.root / "train"
        self.split_root.mkdir(parents=True)

        self.feature_dimension = 3
        lengths = [2, 3, 1, 4]
        matrix = np.arange(
            sum(lengths) * self.feature_dimension,
            dtype=np.float16,
        ).reshape(sum(lengths), self.feature_dimension)
        matrix.tofile(self.split_root / "features.f16")

        offset = 0
        rows = []
        for index, length in enumerate(lengths):
            rows.append(
                {
                    "utt_id": str(index),
                    "start_frame": offset,
                    "num_frames": length,
                    "lid": index % 2,
                    "dialect": index % 2,
                    "gender": index % 2,
                    "language": "bn",
                    "filepath": f"/tmp/example_{index}.wav",
                    "audio_samples": length * 320,
                }
            )
            offset += length

        with (self.split_root / "index.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_dataset_reads_exact_sequence(self) -> None:
        dataset = CachedFeatureDataset(
            self.split_root,
            feature_dimension=self.feature_dimension,
        )
        item = dataset[1]

        self.assertEqual(
            tuple(item["features"].shape),
            (3, self.feature_dimension),
        )
        self.assertEqual(item["features"].dtype, torch.float32)
        self.assertEqual(item["utt_id"], "1")

    def test_collator_preserves_full_lengths(self) -> None:
        dataset = CachedFeatureDataset(
            self.split_root,
            feature_dimension=self.feature_dimension,
        )
        batch = pad_cached_feature_batch([dataset[0], dataset[3]])

        self.assertEqual(
            tuple(batch["features"].shape),
            (2, 4, self.feature_dimension),
        )
        self.assertEqual(batch["frame_lengths"].tolist(), [2, 4])
        self.assertAlmostEqual(
            float(batch["relative_lengths"][0]),
            0.5,
        )

    def test_sampler_respects_budget(self) -> None:
        lengths = [2, 3, 1, 4, 8]
        sampler = FrameBudgetBatchSampler(
            lengths,
            max_batch_frames=6,
            max_batch_size=2,
            number_of_buckets=2,
            shuffle=False,
        )

        batches = list(sampler)
        flattened = [item for batch in batches for item in batch]

        self.assertEqual(sorted(flattened), list(range(len(lengths))))

        for batch in batches:
            padded_cost = max(lengths[index] for index in batch) * len(batch)
            if len(batch) > 1:
                self.assertLessEqual(padded_cost, 6)


if __name__ == "__main__":
    unittest.main()

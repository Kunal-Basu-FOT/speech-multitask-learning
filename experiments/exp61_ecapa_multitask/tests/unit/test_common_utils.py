from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from speech_mtl.common.utils.atomic_io import (
    atomic_write_json,
    sha256_file,
)
from speech_mtl.common.utils.config import (
    deep_merge,
    get_required,
)
from speech_mtl.common.utils.device import (
    parse_compute_process_csv,
    parse_gpu_inventory_csv,
)
from speech_mtl.common.utils.reproducibility import (
    seed_everything,
)


class CommonUtilsTests(unittest.TestCase):
    def test_atomic_json_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"

            atomic_write_json(
                path,
                {"b": 2, "a": 1},
            )

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"a": 1, "b": 2},
            )
            self.assertEqual(len(sha256_file(path)), 64)

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {
            "training": {
                "epochs": 40,
                "optimizer": {"name": "AdamW"},
            }
        }
        override = {
            "training": {
                "optimizer": {"lr": 1.0e-4},
            }
        }

        merged = deep_merge(base, override)

        self.assertEqual(
            get_required(
                merged,
                "training.optimizer.name",
            ),
            "AdamW",
        )
        self.assertEqual(
            get_required(
                merged,
                "training.optimizer.lr",
            ),
            1.0e-4,
        )
        self.assertNotIn(
            "lr",
            base["training"]["optimizer"],
        )

    def test_seed_reproducibility(self) -> None:
        seed_everything(1337)
        tensor_a = torch.rand(3)
        array_a = np.random.rand(3)

        seed_everything(1337)
        tensor_b = torch.rand(3)
        array_b = np.random.rand(3)

        self.assertTrue(torch.equal(tensor_a, tensor_b))
        self.assertTrue(np.array_equal(array_a, array_b))

    def test_gpu_csv_parsers(self) -> None:
        inventory = parse_gpu_inventory_csv(
            "0, GPU-a, NVIDIA GeForce RTX 3090, 24576, 100, 5\n"
            "1, GPU-b, NVIDIA GeForce RTX 3090, 24576, 200, 7\n"
        )
        processes = parse_compute_process_csv(
            "GPU-b, 111, python, 4096\n"
        )

        self.assertEqual(len(inventory), 2)
        self.assertEqual(inventory[1].index, 1)
        self.assertEqual(
            inventory[1].memory_total_mib,
            24576,
        )
        self.assertEqual(len(processes), 1)
        self.assertEqual(processes[0].pid, 111)

    def test_missing_required_key_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_required(
                {"a": {}},
                "a.b",
            )


if __name__ == "__main__":
    unittest.main()

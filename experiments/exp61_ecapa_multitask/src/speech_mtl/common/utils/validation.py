from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .atomic_io import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from .config import deep_merge, get_required, load_yaml
from .device import (
    assess_physical_gpu,
    parse_compute_process_csv,
    parse_gpu_inventory_csv,
)
from .logging_utils import configure_logging
from .reproducibility import seed_everything
from .runtime import collect_runtime_metadata


def validate_utils_contract() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        yaml_path = root / "config.yaml"
        atomic_write_text(
            yaml_path,
            """
project:
  name: exp61
training:
  epochs: 40
  optimizer:
    name: AdamW
""".lstrip(),
        )

        loaded = load_yaml(yaml_path)

        merged = deep_merge(
            loaded,
            {
                "training": {
                    "optimizer": {
                        "learning_rate": 0.0001,
                    }
                }
            },
        )

        if get_required(merged, "project.name") != "exp61":
            raise RuntimeError("Required-key lookup failed")

        if get_required(
            merged,
            "training.optimizer.learning_rate",
        ) != 0.0001:
            raise RuntimeError("Deep merge failed")

        json_path = root / "state.json"
        atomic_write_json(
            json_path,
            {"status": "PASS", "epoch": 1},
        )

        loaded_json = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        if loaded_json != {"epoch": 1, "status": "PASS"}:
            raise RuntimeError("Atomic JSON write failed")

        digest = sha256_file(json_path)

        if len(digest) != 64:
            raise RuntimeError("SHA-256 contract failed")

        logger = configure_logging(
            root / "validation.log",
            level=logging.INFO,
            logger_name="speech_mtl.validation",
        )
        logger.info("COMMON_UTILS_LOGGING_TEST")

        if "COMMON_UTILS_LOGGING_TEST" not in (
            root / "validation.log"
        ).read_text(encoding="utf-8"):
            raise RuntimeError("Logging contract failed")

    first_seed_report = seed_everything(
        1337,
        deterministic_algorithms=False,
    )
    python_tensor_a = torch.rand(4)
    numpy_array_a = np.random.rand(4)

    second_seed_report = seed_everything(
        1337,
        deterministic_algorithms=False,
    )
    python_tensor_b = torch.rand(4)
    numpy_array_b = np.random.rand(4)

    if not torch.equal(python_tensor_a, python_tensor_b):
        raise RuntimeError("Torch reproducibility contract failed")

    if not np.array_equal(numpy_array_a, numpy_array_b):
        raise RuntimeError("NumPy reproducibility contract failed")

    inventory = parse_gpu_inventory_csv(
        "0, GPU-a, RTX 3090, 24576, 12, 0\n"
        "1, GPU-b, RTX 3090, 24576, 24, 1\n"
    )

    processes = parse_compute_process_csv(
        "GPU-b, 1234, python, 4096\n"
    )

    if [item.index for item in inventory] != [0, 1]:
        raise RuntimeError("GPU inventory parser failed")

    if len(processes) != 1 or processes[0].pid != 1234:
        raise RuntimeError("GPU process parser failed")

    live_gpu_assessment: dict[str, Any]

    try:
        live_gpu_assessment = assess_physical_gpu(
            1,
            maximum_used_memory_mib=1024,
            maximum_utilization_percent=10,
            reject_compute_processes=True,
        )
    except Exception as exc:
        live_gpu_assessment = {
            "queried": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        live_gpu_assessment["queried"] = True

    runtime_metadata = collect_runtime_metadata()

    return {
        "status": "PASS",
        "configuration_contract": "PASS",
        "atomic_io_contract": "PASS",
        "sha256_contract": "PASS",
        "logging_contract": "PASS",
        "reproducibility_contract": "PASS",
        "gpu_parser_contract": "PASS",
        "seed_reports_equal": (
            first_seed_report == second_seed_report
        ),
        "live_physical_gpu1_assessment": live_gpu_assessment,
        "runtime_metadata": runtime_metadata,
    }

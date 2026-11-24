from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic_algorithms: bool = False,
    warn_only: bool = True,
) -> dict[str, Any]:
    """Seed Python, NumPy, CPU Torch, and all visible CUDA devices."""

    seed = int(seed)

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=warn_only,
    )

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic_algorithms
        torch.backends.cudnn.deterministic = deterministic_algorithms

    return {
        "seed": seed,
        "deterministic_algorithms": (
            deterministic_algorithms
        ),
        "warn_only": warn_only,
        "cuda_seeded": torch.cuda.is_available(),
    }

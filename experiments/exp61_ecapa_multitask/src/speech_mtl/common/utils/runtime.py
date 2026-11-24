from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata(
    *,
    include_environment: tuple[str, ...] = (
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
    ),
) -> dict[str, Any]:
    """Collect stable runtime metadata without initializing CUDA."""

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "PyYAML": _package_version("PyYAML"),
            "torch": _package_version("torch"),
            "torchaudio": _package_version("torchaudio"),
            "speechbrain": _package_version("speechbrain"),
            "tqdm": _package_version("tqdm"),
            "matplotlib": _package_version("matplotlib"),
            "scikit-learn": _package_version("scikit-learn"),
        },
        "environment": {
            key: os.environ.get(key)
            for key in include_environment
        },
    }

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GPUInventoryItem:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUProcess:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_nvidia_smi(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi is not available on this host"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "nvidia-smi failed: "
            + (exc.stderr.strip() or exc.stdout.strip())
        ) from exc

    return completed.stdout.strip()


def _parse_integer(value: str) -> int:
    cleaned = value.strip().replace(" MiB", "").replace("%", "")

    if cleaned in {"", "N/A", "[N/A]"}:
        return 0

    return int(cleaned)


def parse_gpu_inventory_csv(text: str) -> list[GPUInventoryItem]:
    items: list[GPUInventoryItem] = []

    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue

        if len(row) != 6:
            raise ValueError(
                f"Unexpected GPU inventory row: {row}"
            )

        items.append(
            GPUInventoryItem(
                index=int(row[0].strip()),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_total_mib=_parse_integer(row[3]),
                memory_used_mib=_parse_integer(row[4]),
                utilization_percent=_parse_integer(row[5]),
            )
        )

    return sorted(items, key=lambda item: item.index)


def parse_compute_process_csv(text: str) -> list[GPUProcess]:
    items: list[GPUProcess] = []

    if not text.strip():
        return items

    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue

        if len(row) != 4:
            raise ValueError(
                f"Unexpected compute-process row: {row}"
            )

        items.append(
            GPUProcess(
                gpu_uuid=row[0].strip(),
                pid=int(row[1].strip()),
                process_name=row[2].strip(),
                used_memory_mib=_parse_integer(row[3]),
            )
        )

    return items


def query_gpu_inventory() -> list[GPUInventoryItem]:
    output = _run_nvidia_smi(
        [
            "--query-gpu="
            "index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_inventory_csv(output)


def query_compute_processes() -> list[GPUProcess]:
    try:
        output = _run_nvidia_smi(
            [
                "--query-compute-apps="
                "gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except RuntimeError as exc:
        # Some driver versions return nonzero when there are no processes.
        message = str(exc).lower()

        if "no running processes found" in message:
            return []

        raise

    return parse_compute_process_csv(output)


def assess_physical_gpu(
    physical_index: int,
    *,
    maximum_used_memory_mib: int = 1024,
    maximum_utilization_percent: int = 10,
    reject_compute_processes: bool = True,
) -> dict[str, Any]:
    """Inspect a physical GPU without initializing a CUDA context."""

    inventory = {
        item.index: item
        for item in query_gpu_inventory()
    }

    if physical_index not in inventory:
        raise RuntimeError(
            f"Physical GPU {physical_index} does not exist; "
            f"available={sorted(inventory)}"
        )

    gpu = inventory[physical_index]
    processes = [
        process
        for process in query_compute_processes()
        if process.gpu_uuid == gpu.uuid
    ]

    reasons: list[str] = []

    if gpu.memory_used_mib > maximum_used_memory_mib:
        reasons.append(
            "used memory "
            f"{gpu.memory_used_mib} MiB exceeds "
            f"{maximum_used_memory_mib} MiB"
        )

    if gpu.utilization_percent > maximum_utilization_percent:
        reasons.append(
            "utilization "
            f"{gpu.utilization_percent}% exceeds "
            f"{maximum_utilization_percent}%"
        )

    if reject_compute_processes and processes:
        reasons.append(
            f"{len(processes)} compute process(es) already present"
        )

    return {
        "available": not reasons,
        "physical_gpu": gpu.to_dict(),
        "compute_processes": [
            process.to_dict() for process in processes
        ],
        "reasons": reasons,
        "thresholds": {
            "maximum_used_memory_mib": maximum_used_memory_mib,
            "maximum_utilization_percent": (
                maximum_utilization_percent
            ),
            "reject_compute_processes": reject_compute_processes,
        },
    }


def configure_physical_gpu(physical_index: int) -> None:
    """Map one physical GPU to process-local cuda:0.

    This must run before CUDA is initialized.
    """

    if physical_index < 0:
        raise ValueError("physical_index must be non-negative")

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA is already initialized; physical GPU mapping "
                "cannot be changed safely"
            )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_index)


def verify_single_visible_cuda_device() -> dict[str, Any]:
    """Require exactly one visible CUDA device after GPU mapping."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable; CPU fallback is forbidden"
        )

    count = torch.cuda.device_count()

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA device, received {count}"
        )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    return {
        "process_device": "cuda:0",
        "visible_device_count": count,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
    }

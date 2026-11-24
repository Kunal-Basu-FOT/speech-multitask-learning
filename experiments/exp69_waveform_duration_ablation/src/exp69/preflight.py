from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .checkpoint import (
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from .config import Config
from .model import LogMelProjectedMultitaskECAPA, parameter_report
from .utils import atomic_json, frame_cap, seed_everything, sha256_file, utc_now


def _check_source_indexes(config: Config) -> dict[str, Any]:
    root = config.path("source_cache_root")
    details: dict[str, Any] = {}
    for split, expected in config.expected_rows.items():
        one = root / "one_bit" / split / "index.csv"
        sixteen = root / "sixteen_bit" / split / "index.csv"
        if not one.is_file() or not sixteen.is_file():
            raise FileNotFoundError(f"Missing source indexes for {split}")
        one_hash = sha256_file(one)
        sixteen_hash = sha256_file(sixteen)
        if one_hash != sixteen_hash:
            raise RuntimeError(f"Source indexes are not byte-identical for {split}")
        frame = pd.read_csv(one, dtype={"utt_id": str, "filepath": str, "language": str})
        if len(frame) != int(expected):
            raise RuntimeError(f"Expected {expected} rows for {split}, found {len(frame)}")
        if frame["utt_id"].duplicated().any():
            raise RuntimeError(f"Duplicate utterance IDs in {split}")
        sample_indices = sorted(set([0, len(frame) // 4, len(frame) // 2, 3 * len(frame) // 4, len(frame) - 1]))
        missing = [str(frame.iloc[index].filepath) for index in sample_indices if not Path(str(frame.iloc[index].filepath)).is_file()]
        if missing:
            raise FileNotFoundError(f"Sample source audio missing: {missing[:5]}")
        details[split] = {
            "rows": len(frame),
            "sha256": one_hash,
            "sample_audio_checked": len(sample_indices),
        }
    return details


def _check_exp61_reference(config: Config) -> dict[str, Any]:
    root = config.path("exp61_root")
    marker = root / "runtime/runs/ecapa_7p2m/exp61/EXPERIMENT_COMPLETE.json"
    table = root / "reports/ecapa_7p2m/exp61/tables/table_02_matched_results.csv"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    if not table.is_file():
        raise FileNotFoundError(table)
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "EXPERIMENT_COMPLETE":
        raise RuntimeError("EXP 6.1 is not complete")
    return {"completion_marker": str(marker), "reference_table": str(table)}


def _model_smoke(config: Config, device: torch.device) -> dict[str, Any]:
    model = LogMelProjectedMultitaskECAPA(
        int(config.feature_dimension), int(config.projected_dimension)
    ).to(device)
    model.eval()
    batch = 3
    frames = 80
    features = torch.randn(batch, frames, int(config.feature_dimension), device=device)
    valid_mask = torch.ones(batch, frames, dtype=torch.bool, device=device)
    relative_lengths = torch.ones(batch, device=device)
    languages = ["bn", "bh", "te"]
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(
                features=features,
                relative_lengths=relative_lengths,
                oracle_languages=languages,
                valid_mask=valid_mask,
            )
    if tuple(output.embeddings.shape) != (batch, int(config.embedding_dimension)):
        raise RuntimeError(f"Unexpected embedding shape: {tuple(output.embeddings.shape)}")
    if not torch.isfinite(output.embeddings).all():
        raise RuntimeError("Synthetic model forward produced non-finite embeddings")
    return {"embedding_shape": list(output.embeddings.shape), "parameters": parameter_report(model)}


def _rng_roundtrip(config: Config, device: torch.device) -> dict[str, Any]:
    """Real training-checkpoint roundtrip on the selected CUDA device."""
    seed_everything(int(config.seed))
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    contract = {"test": "rng_roundtrip", "visible_cuda_devices": torch.cuda.device_count()}
    with tempfile.TemporaryDirectory(prefix="exp69_rng_") as temporary:
        path = Path(temporary) / "checkpoint.pt"
        save_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            training_state={"completed_epoch": 0},
            contract=contract,
        )
        # Perturb RNG before restore.
        _ = torch.rand(8)
        if device.type == "cuda":
            _ = torch.rand(8, device=device)
        state = load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_contract=contract,
        )
        cpu_rng = torch.get_rng_state()
        if cpu_rng.device.type != "cpu" or cpu_rng.dtype != torch.uint8:
            raise RuntimeError("Restored CPU RNG state is not a CPU ByteTensor")
        return {
            "status": "RNG_CHECKPOINT_ROUNDTRIP_PASS",
            "checkpoint_bytes": path.stat().st_size,
            "cpu_rng_device": cpu_rng.device.type,
            "cpu_rng_dtype": str(cpu_rng.dtype),
            "training_state": state,
        }


def run_preflight(config: Config, *, require_gpu: bool) -> dict[str, Any]:
    if socket.gethostname() != "wtc9-server":
        raise RuntimeError(f"EXP 6.9 is restricted to wtc9-server, got {socket.gethostname()}")
    if os.environ.get("USER") != "wtc9":
        raise RuntimeError(f"EXP 6.9 is restricted to user wtc9, got {os.environ.get('USER')}")

    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "The launcher must expose exactly physical GPU 1 as one local cuda:0 device; "
                f"visible={torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    source_indexes = _check_source_indexes(config)
    reference = _check_exp61_reference(config)
    model = _model_smoke(config, device)
    rng = _rng_roundtrip(config, device)

    caps = {
        name: frame_cap(
            config.duration_seconds[name],
            int(config.sample_rate),
            int(config.win_length),
            int(config.hop_length),
        )
        for name in config.durations
    }
    expected_caps = {"3s": 298, "7s": 698, "9s": 898, "full": None}
    if caps != expected_caps:
        raise RuntimeError(f"Duration-cap contract changed: {caps}")

    checkpoint_source = (config.path("project_root") / "src/exp69/checkpoint.py").read_text(encoding="utf-8")
    if 'map_location="cpu"' not in checkpoint_source:
        raise RuntimeError("Checkpoint loader does not explicitly map to CPU")
    if '.to(device="cpu", dtype=torch.uint8)' not in checkpoint_source:
        raise RuntimeError("Checkpoint RNG restoration lacks explicit CPU uint8 conversion")
    if "map_location=device" in checkpoint_source:
        raise RuntimeError("Unsafe map_location=device found in checkpoint source")

    payload = {
        "status": "EXP69_PREFLIGHT_PASS",
        "created_utc": utc_now(),
        "require_gpu": require_gpu,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "source_indexes": source_indexes,
        "exp61_reference": reference,
        "model_smoke": model,
        "rng_roundtrip": rng,
        "duration_frame_caps": caps,
    }
    output = config.path("project_root") / "runtime" / (
        "GPU1_PREFLIGHT.json" if require_gpu else "CPU_PREFLIGHT.json"
    )
    atomic_json(output, payload)
    print("EXP69_PREFLIGHT_PASS", flush=True)
    return payload

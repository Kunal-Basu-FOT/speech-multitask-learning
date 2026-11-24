from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm

from .cache import build_loader
from .config import AnalysisConfig, CACHE_ROOT, COMBINED_ROOT, EXP62_ROOT
from .encoder import build_random_encoder, forward_encoder, load_frozen_encoder
from .utils import atomic_json, ensure_finite, sha256_file, utc_now


def exp62_checkpoint(domain: str, condition: str) -> Path:
    return (
        EXP62_ROOT
        / "runtime/runs/exp62_reduced"
        / domain
        / condition
        / "checkpoints/best.pt"
    )


def output_cell(source: str, input_domain: str, split: str) -> Path:
    if source == "random":
        return COMBINED_ROOT / "runtime/embeddings/random/model_random" / f"input_{input_domain}" / split
    return COMBINED_ROOT / "runtime/embeddings/exp62" / f"model_{source}" / f"input_{input_domain}" / split


def _load_index(domain: str, split: str, expected_rows: int) -> tuple[pd.DataFrame, Path, Path]:
    source = CACHE_ROOT / domain / split
    index_path = source / "index.csv"
    features_path = source / "features.f16"
    if not index_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(f"cache split is incomplete: {source}")
    frame = pd.read_csv(index_path)
    required = {
        "utt_id",
        "start_frame",
        "num_frames",
        "lid",
        "dialect",
        "gender",
        "language",
        "filepath",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{index_path} missing columns: {missing}")
    if len(frame) != expected_rows:
        raise RuntimeError(f"{index_path} rows={len(frame)} expected={expected_rows}")
    frame = frame.sort_values(["num_frames", "utt_id"], kind="mergesort").reset_index(drop=True)
    if frame["utt_id"].astype(str).duplicated().any():
        raise RuntimeError(f"duplicate utt_id in {index_path}")
    return frame, index_path, features_path


def _existing_complete(final: Path, rows: int, dimension: int) -> bool:
    marker = final / "COMPLETE.json"
    embeddings = final / "embeddings.npy"
    metadata = final / "metadata.csv"
    if not (marker.is_file() and embeddings.is_file() and metadata.is_file()):
        return False
    array = np.load(embeddings, mmap_mode="r")
    if array.shape != (rows, dimension) or array.dtype != np.float32:
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return int(record.get("rows", -1)) == rows and int(record.get("embedding_dimension", -1)) == dimension


def extract_cell(
    *,
    encoder,
    source: str,
    checkpoint_path: Path | None,
    input_domain: str,
    split: str,
    device: torch.device,
    config: AnalysisConfig,
    workers: int,
) -> dict[str, Any]:
    frame, index_path, features_path = _load_index(
        input_domain, split, config.expected_rows[split]
    )
    final = output_cell(source, input_domain, split)
    if _existing_complete(final, len(frame), config.embedding_dimension):
        print(f"SKIP_COMPLETE source={source} input={input_domain} split={split}", flush=True)
        return json.loads((final / "COMPLETE.json").read_text(encoding="utf-8"))
    if final.exists():
        raise RuntimeError(f"incomplete output exists and was not overwritten: {final}")
    partial = final.with_name(f".{final.name}.partial.{os.getpid()}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    frame.to_csv(partial / "metadata.csv", index=False)
    destination = open_memmap(
        partial / "embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(frame), config.embedding_dimension),
    )
    loader = build_loader(
        frame=frame,
        features_path=features_path,
        feature_dimension=config.feature_dimension,
        frame_budget=config.frame_budget,
        maximum_batch_size=config.maximum_batch_size,
        workers=workers,
    )
    cursor = 0
    started = time.monotonic()
    iterator = tqdm(
        loader,
        total=len(loader),
        desc=f"{source}->{input_domain}/{split}",
        dynamic_ncols=True,
        leave=True,
    )
    with torch.inference_mode():
        for features, relative_lengths, valid_mask, _metadata in iterator:
            features = features.to(device=device, non_blocking=True)
            relative_lengths = relative_lengths.to(device=device, non_blocking=True)
            valid_mask = valid_mask.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                embeddings = forward_encoder(
                    encoder, features, relative_lengths, valid_mask
                )
            block = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
            if block.shape[1] != config.embedding_dimension:
                raise RuntimeError(f"unexpected embedding shape: {block.shape}")
            ensure_finite("embedding block", block)
            destination[cursor : cursor + len(block)] = block
            cursor += len(block)
            iterator.set_postfix(examples=cursor)
    destination.flush()
    del destination
    if cursor != len(frame):
        raise RuntimeError(f"extracted rows={cursor} expected={len(frame)}")
    verification = np.load(partial / "embeddings.npy", mmap_mode="r")
    ensure_finite("completed embedding array", np.asarray(verification))
    elapsed = time.monotonic() - started
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path else None
    marker = {
        "status": "EMBEDDING_CELL_COMPLETE",
        "created_utc": utc_now(),
        "source": source,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_hash,
        "input_domain": input_domain,
        "split": split,
        "rows": len(frame),
        "embedding_dimension": config.embedding_dimension,
        "embedding_dtype": "float32",
        "source_index": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "ordering": "stable sort by num_frames then utt_id",
        "elapsed_seconds": elapsed,
        "examples_per_second": len(frame) / elapsed,
    }
    atomic_json(partial / "COMPLETE.json", marker)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, final)
    print(f"CELL_COMPLETE source={source} input={input_domain} split={split}", flush=True)
    return marker


def smoke(config: AnalysisConfig, device: torch.device) -> None:
    frame, _, features_path = _load_index("one_bit", "test", config.expected_rows["test"])
    frame = frame.head(16).copy()
    loader = build_loader(
        frame=frame,
        features_path=features_path,
        feature_dimension=config.feature_dimension,
        frame_budget=config.frame_budget,
        maximum_batch_size=16,
        workers=0,
    )
    features, relative_lengths, valid_mask, _ = next(iter(loader))
    features = features.to(device)
    relative_lengths = relative_lengths.to(device)
    valid_mask = valid_mask.to(device)
    for domain in config.domains:
        for condition in config.conditions:
            path = exp62_checkpoint(domain, condition)
            encoder = load_frozen_encoder(path, device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                embeddings = forward_encoder(encoder, features, relative_lengths, valid_mask)
            if embeddings.shape != (16, config.embedding_dimension):
                raise RuntimeError(f"smoke shape failure for {domain}/{condition}: {embeddings.shape}")
            if not torch.isfinite(embeddings).all():
                raise RuntimeError(f"smoke non-finite values for {domain}/{condition}")
            print(f"SMOKE_PASS {domain}/{condition} shape={tuple(embeddings.shape)}")
            del encoder
            torch.cuda.empty_cache()
    random_encoder = build_random_encoder(config.seed, device)
    with torch.inference_mode():
        embeddings = forward_encoder(random_encoder, features, relative_lengths, valid_mask)
    if embeddings.shape != (16, config.embedding_dimension):
        raise RuntimeError("random encoder smoke shape failure")
    print("EXP65_68_EXTRACTION_SMOKE_PASS")


def run(config: AnalysisConfig, device: torch.device, workers: int) -> None:
    all_markers: list[dict[str, Any]] = []
    for domain in config.domains:
        for condition in config.conditions:
            source = f"{domain}_{condition}"
            checkpoint_path = exp62_checkpoint(domain, condition)
            encoder = load_frozen_encoder(checkpoint_path, device)
            for input_domain in config.domains:
                for split in config.splits:
                    all_markers.append(
                        extract_cell(
                            encoder=encoder,
                            source=source,
                            checkpoint_path=checkpoint_path,
                            input_domain=input_domain,
                            split=split,
                            device=device,
                            config=config,
                            workers=workers,
                        )
                    )
            del encoder
            torch.cuda.empty_cache()
    random_encoder = build_random_encoder(config.seed, device)
    for input_domain in config.domains:
        for split in config.splits:
            all_markers.append(
                extract_cell(
                    encoder=random_encoder,
                    source="random",
                    checkpoint_path=None,
                    input_domain=input_domain,
                    split=split,
                    device=device,
                    config=config,
                    workers=workers,
                )
            )
    expected = len(config.domains) * len(config.conditions) * len(config.domains) * len(config.splits) + len(config.domains) * len(config.splits)
    if len(all_markers) != expected:
        raise RuntimeError(f"completed cell count={len(all_markers)} expected={expected}")
    atomic_json(
        COMBINED_ROOT / "runtime/embeddings/EXP62_RANDOM_EXTRACTION_COMPLETE.json",
        {
            "status": "EXP62_RANDOM_EXTRACTION_COMPLETE",
            "created_utc": utc_now(),
            "cells": len(all_markers),
            "expected_cells": expected,
            "outputs": all_markers,
        },
    )
    print(f"EXP62_RANDOM_EXTRACTION_COMPLETE cells={len(all_markers)}")

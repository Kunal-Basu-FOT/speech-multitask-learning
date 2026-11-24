#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from speech_mtl.common.models.factory import build_exp61_model


EXP61 = Path("/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK")
CACHE = Path(
    "/data/Kunal/shared_cache/"
    "exp61_indicwav2vec_final_layer_fp16"
)
OUT = Path(
    "/data/Kunal/"
    "FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_"
    "IndicWav2Vec_FinalLayer/runtime/reference_embeddings/exp61"
)

DOMAINS = ("one_bit", "sixteen_bit")
SPLITS = ("train", "dev", "test")

FEATURE_DIM = 768
EMBED_DIM = 192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


class CacheRows(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features_path: Path,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.features_path = str(features_path)

        frame_bytes = (
            FEATURE_DIM
            * np.dtype(np.float16).itemsize
        )

        file_bytes = features_path.stat().st_size

        if file_bytes % frame_bytes:
            raise RuntimeError(
                f"Invalid feature file size: {features_path}"
            )

        self.total_frames = file_bytes // frame_bytes
        self.memmap = None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, str]:
        if self.memmap is None:
            self.memmap = np.memmap(
                self.features_path,
                mode="r",
                dtype=np.float16,
                shape=(
                    self.total_frames,
                    FEATURE_DIM,
                ),
            )

        row = self.frame.iloc[index]

        start = int(row.start_frame)
        length = int(row.num_frames)
        stop = start + length

        if stop > self.total_frames:
            raise RuntimeError(
                f"Cache index overflow at row {index}"
            )

        sequence = torch.from_numpy(
            np.array(
                self.memmap[start:stop],
                copy=True,
            )
        )

        return sequence, str(row.language)


class FrameBatches(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        frame_budget: int = 12_000,
        maximum_batch_size: int = 32,
    ) -> None:
        self.batches: list[list[int]] = []

        batch: list[int] = []
        maximum = 0

        for index, length_value in enumerate(lengths):
            length = int(length_value)

            proposed_maximum = max(
                maximum,
                length,
            )

            proposed_size = len(batch) + 1

            exceeds_budget = (
                proposed_size > maximum_batch_size
                or proposed_size * proposed_maximum
                > frame_budget
            )

            if batch and exceeds_budget:
                self.batches.append(batch)
                batch = []
                maximum = 0

            batch.append(index)
            maximum = max(maximum, length)

        if batch:
            self.batches.append(batch)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate(
    items: list[tuple[torch.Tensor, str]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
]:
    lengths = torch.tensor(
        [
            sequence.shape[0]
            for sequence, _ in items
        ],
        dtype=torch.long,
    )

    maximum = int(lengths.max().item())

    features = torch.zeros(
        len(items),
        maximum,
        FEATURE_DIM,
        dtype=torch.float16,
    )

    valid_mask = torch.zeros(
        len(items),
        maximum,
        dtype=torch.bool,
    )

    languages: list[str] = []

    for index, (sequence, language) in enumerate(items):
        length = int(sequence.shape[0])

        features[index, :length] = sequence
        valid_mask[index, :length] = True
        languages.append(language)

    relative_lengths = (
        lengths.to(torch.float32)
        / float(maximum)
    )

    return (
        features,
        relative_lengths,
        valid_mask,
        languages,
    )


def checkpoint(
    model_domain: str,
) -> Path:
    return (
        EXP61
        / "runtime/runs/ecapa_7p2m/exp61"
        / model_domain
        / "checkpoints/best.pt"
    )


def load_model(
    model_domain: str,
    device: torch.device,
):
    path = checkpoint(model_domain)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    if "model" not in payload:
        raise RuntimeError(
            f"Checkpoint has no model state: {path}"
        )

    model = build_exp61_model()

    model.load_state_dict(
        payload["model"],
        strict=True,
    )

    model.eval()
    model.requires_grad_(False)
    model.to(device)

    del payload

    return model, sha256(path)


def cell_directory(
    model_domain: str,
    input_domain: str,
    split: str,
) -> Path:
    return (
        OUT
        / f"model_{model_domain}"
        / f"input_{input_domain}"
        / split
    )


def extract_cell(
    *,
    model,
    checkpoint_hash: str,
    model_domain: str,
    input_domain: str,
    split: str,
    device: torch.device,
    workers: int,
) -> dict[str, Any]:
    source = CACHE / input_domain / split

    index_path = source / "index.csv"
    features_path = source / "features.f16"

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

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise RuntimeError(
            f"{index_path} missing columns: {missing}"
        )

    frame = (
        frame.sort_values(
            ["num_frames", "utt_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    final = cell_directory(
        model_domain,
        input_domain,
        split,
    )

    marker_path = final / "COMPLETE.json"
    embeddings_path = final / "embeddings.npy"

    if (
        marker_path.is_file()
        and embeddings_path.is_file()
    ):
        existing = np.load(
            embeddings_path,
            mmap_mode="r",
        )

        if existing.shape == (
            len(frame),
            EMBED_DIM,
        ):
            print(
                "SKIP_COMPLETE "
                f"{model_domain}->{input_domain}/{split}",
                flush=True,
            )

            return json.loads(
                marker_path.read_text(
                    encoding="utf-8"
                )
            )

    if final.exists():
        raise RuntimeError(
            f"Incomplete output already exists: {final}"
        )

    partial = final.with_name(
        f".{final.name}.partial.{os.getpid()}"
    )

    partial.mkdir(
        parents=True,
        exist_ok=False,
    )

    dataset = CacheRows(
        frame,
        features_path,
    )

    batch_sampler = FrameBatches(
        frame["num_frames"]
        .astype(int)
        .tolist()
    )

    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate,
    }

    if workers > 0:
        loader_arguments.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    loader = DataLoader(
        **loader_arguments
    )

    chunks: list[np.ndarray] = []
    started = time.monotonic()
    examples = 0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc=(
            f"{model_domain}"
            f"->{input_domain}/{split}"
        ),
        dynamic_ncols=True,
        leave=True,
    )

    with torch.inference_mode():
        for (
            features,
            relative_lengths,
            valid_mask,
            languages,
        ) in iterator:
            features = features.to(
                device=device,
                non_blocking=True,
            )

            relative_lengths = (
                relative_lengths.to(
                    device=device,
                    non_blocking=True,
                )
            )

            valid_mask = valid_mask.to(
                device=device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(
                    features=features,
                    relative_lengths=relative_lengths,
                    oracle_languages=languages,
                    valid_mask=valid_mask,
                )

            embedding_batch = (
                output.embeddings
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .numpy()
            )

            if embedding_batch.ndim != 2:
                raise RuntimeError(
                    "Embeddings are not two-dimensional"
                )

            if embedding_batch.shape[1] != EMBED_DIM:
                raise RuntimeError(
                    "Unexpected embedding dimension: "
                    f"{embedding_batch.shape}"
                )

            if not np.isfinite(
                embedding_batch
            ).all():
                raise RuntimeError(
                    "Embedding batch contains "
                    "non-finite values"
                )

            chunks.append(
                embedding_batch
            )

            examples += int(
                embedding_batch.shape[0]
            )

            iterator.set_postfix(
                examples=examples
            )

    embeddings = np.concatenate(
        chunks,
        axis=0,
    )

    expected_shape = (
        len(frame),
        EMBED_DIM,
    )

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Expected {expected_shape}; "
            f"received {embeddings.shape}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise RuntimeError(
            "Final embeddings contain "
            "non-finite values"
        )

    np.save(
        partial / "embeddings.npy",
        embeddings.astype(
            np.float32,
            copy=False,
        ),
    )

    frame.to_csv(
        partial / "metadata.csv",
        index=False,
    )

    elapsed = time.monotonic() - started

    marker = {
        "status":
            "REFERENCE_EMBEDDINGS_COMPLETE",
        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "source_experiment":
            "EXP6.1",
        "source_model_role":
            "gender-supervised three-task "
            "MTL reference",
        "model_domain":
            model_domain,
        "input_domain":
            input_domain,
        "split":
            split,
        "rows":
            len(frame),
        "embedding_dimension":
            EMBED_DIM,
        "embedding_dtype":
            "float32",
        "amp_inference":
            True,
        "checkpoint":
            str(
                checkpoint(model_domain)
            ),
        "checkpoint_sha256":
            checkpoint_hash,
        "source_index":
            str(index_path),
        "source_index_sha256":
            sha256(index_path),
        "elapsed_seconds":
            elapsed,
        "examples_per_second":
            len(frame) / elapsed,
        "ordering":
            "stable sort by num_frames "
            "then utt_id; metadata.csv "
            "aligns row-for-row with "
            "embeddings.npy",
    }

    atomic_json(
        partial / "COMPLETE.json",
        marker,
    )

    final.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(
        partial,
        final,
    )

    print(
        "CELL_COMPLETE "
        f"{model_domain}"
        f"->{input_domain}/{split}",
        flush=True,
    )

    return marker


def smoke_test(
    device: torch.device,
) -> None:
    source = CACHE / "one_bit/test"

    frame = pd.read_csv(
        source / "index.csv"
    ).head(16)

    dataset = CacheRows(
        frame,
        source / "features.f16",
    )

    (
        features,
        relative_lengths,
        valid_mask,
        languages,
    ) = collate(
        [
            dataset[index]
            for index in range(16)
        ]
    )

    for model_domain in DOMAINS:
        model, _ = load_model(
            model_domain,
            device,
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                output = model(
                    features=features.to(device),
                    relative_lengths=(
                        relative_lengths.to(device)
                    ),
                    oracle_languages=languages,
                    valid_mask=valid_mask.to(device),
                )

        embeddings = (
            output.embeddings
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        if embeddings.shape != (
            16,
            EMBED_DIM,
        ):
            raise RuntimeError(
                f"Smoke shape failure: "
                f"{embeddings.shape}"
            )

        if not torch.isfinite(
            embeddings
        ).all():
            raise RuntimeError(
                "Smoke embeddings contain "
                "non-finite values"
            )

        print(
            f"SMOKE_PASS model={model_domain} "
            f"shape={tuple(embeddings.shape)}",
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

    print(
        "EXP61_REFERENCE_SMOKE_PASS",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU; "
            f"found {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")

    print(
        "VISIBLE_GPU",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    if arguments.smoke:
        smoke_test(device)
        return

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cells: list[dict[str, Any]] = []

    for model_domain in DOMAINS:
        model, checkpoint_hash = load_model(
            model_domain,
            device,
        )

        for input_domain in DOMAINS:
            for split in SPLITS:
                cells.append(
                    extract_cell(
                        model=model,
                        checkpoint_hash=(
                            checkpoint_hash
                        ),
                        model_domain=(
                            model_domain
                        ),
                        input_domain=(
                            input_domain
                        ),
                        split=split,
                        device=device,
                        workers=arguments.workers,
                    )
                )

        del model
        torch.cuda.empty_cache()

    if len(cells) != 12:
        raise RuntimeError(
            f"Expected 12 completed cells; "
            f"found {len(cells)}"
        )

    atomic_json(
        OUT
        / "REFERENCE_EXTRACTION_COMPLETE.json",
        {
            "status":
                "EXP61_REFERENCE_EXTRACTION_COMPLETE",
            "created_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "cells":
                12,
            "outputs":
                cells,
        },
    )

    print(
        "EXP61_REFERENCE_EXTRACTION_COMPLETE "
        "cells=12",
        flush=True,
    )


if __name__ == "__main__":
    main()

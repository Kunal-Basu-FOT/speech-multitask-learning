from __future__ import annotations

import math
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import Config
from .utils import frame_cap, stable_hash_int


class LogMelCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        config: Config,
        split: str,
        domain: str,
        duration_name: str,
        training: bool,
    ) -> None:
        if domain not in config.domains:
            raise ValueError(domain)
        if duration_name not in config.durations:
            raise ValueError(duration_name)
        self.config = config
        self.split = split
        self.domain = domain
        self.duration_name = duration_name
        self.training = bool(training)
        self.root = config.path("logmel_cache_root") / split
        self.index_path = self.root / "index.csv"
        self.features_path = self.root / f"features.{domain}.f16"
        self.frame = pd.read_csv(
            self.index_path,
            dtype={"utt_id": str, "language": str, "filepath": str, "speaker_id": str},
        )
        expected = int(config.expected_rows[split])
        if len(self.frame) != expected:
            raise RuntimeError(f"Expected {expected} rows for {split}, found {len(self.frame)}")
        if not self.features_path.is_file():
            raise FileNotFoundError(self.features_path)
        feature_dim = int(config.n_mels)
        frame_bytes = feature_dim * np.dtype(np.float16).itemsize
        file_bytes = self.features_path.stat().st_size
        if file_bytes % frame_bytes:
            raise RuntimeError(f"Invalid binary size: {self.features_path}")
        self.total_frames = file_bytes // frame_bytes
        indexed_total = int((self.frame["start_frame"] + self.frame["num_frames"]).max())
        if indexed_total != self.total_frames:
            raise RuntimeError(
                f"Index/binary total mismatch: index={indexed_total} binary={self.total_frames}"
            )
        seconds = config.duration_seconds[duration_name]
        self.cap_frames = frame_cap(
            seconds,
            int(config.sample_rate),
            int(config.win_length),
            int(config.hop_length),
        )
        self.seed = int(config.seed)
        self._epoch = mp.Value("i", 0, lock=True)
        self._memmap = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_memmap"] = None
        return state

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def effective_lengths(self) -> list[int]:
        lengths = self.frame["num_frames"].astype(int).to_numpy()
        if self.cap_frames is not None:
            lengths = np.minimum(lengths, int(self.cap_frames))
        return lengths.tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def _open(self):
        if self._memmap is None:
            self._memmap = np.memmap(
                self.features_path,
                mode="r",
                dtype=np.float16,
                shape=(self.total_frames, int(self.config.n_mels)),
            )
        return self._memmap

    def _crop_start(self, *, length: int, cap: int, utt_id: str) -> int:
        if length <= cap:
            return 0
        if not self.training:
            return (length - cap) // 2
        epoch = int(self._epoch.value)
        return stable_hash_int(self.seed, epoch, utt_id, self.duration_name) % (length - cap + 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        start = int(row.start_frame)
        length = int(row.num_frames)
        stop = start + length
        if start < 0 or stop > self.total_frames:
            raise RuntimeError(f"Cache range overflow for row {index}")
        cap = self.cap_frames
        local_start = 0 if cap is None else self._crop_start(
            length=length,
            cap=int(cap),
            utt_id=str(row.utt_id),
        )
        selected_length = length if cap is None else min(length, int(cap))
        selected = self._open()[
            start + local_start : start + local_start + selected_length
        ]
        features = torch.from_numpy(np.array(selected, dtype=np.float32, copy=True, order="C"))
        if features.shape != (selected_length, int(self.config.n_mels)):
            raise RuntimeError(f"Unexpected feature shape: {tuple(features.shape)}")
        return {
            "features": features,
            "utt_id": str(row.utt_id),
            "num_frames": selected_length,
            "lid": int(row.lid),
            "dialect": int(row.dialect),
            "gender": int(row.gender),
            "language": str(row.language),
            "filepath": str(row.filepath),
            "speaker_id": str(row.speaker_id),
            "dataset_index": int(index),
            "crop_start_frame": int(local_start),
            "source_num_frames": int(length),
        }


class StableFrameBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        *,
        max_batch_frames: int,
        max_batch_size: int,
        number_of_buckets: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if not lengths:
            raise ValueError("lengths may not be empty")
        self.lengths = [int(value) for value in lengths]
        self.max_batch_frames = int(max_batch_frames)
        self.max_batch_size = int(max_batch_size)
        self.number_of_buckets = int(number_of_buckets)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i))
        buckets = np.array_split(np.asarray(sorted_indices, dtype=np.int64), self.number_of_buckets)
        base_batches: list[list[int]] = []
        for bucket in buckets:
            batch: list[int] = []
            maximum = 0
            for raw_index in bucket.tolist():
                index = int(raw_index)
                length = self.lengths[index]
                proposed_maximum = max(maximum, length)
                proposed_size = len(batch) + 1
                exceeds = (
                    proposed_size > self.max_batch_size
                    or proposed_size * proposed_maximum > self.max_batch_frames
                )
                if batch and exceeds:
                    base_batches.append(batch)
                    batch = []
                    maximum = 0
                batch.append(index)
                maximum = max(maximum, length)
            if batch:
                base_batches.append(batch)
        if not base_batches:
            raise RuntimeError("No batches were constructed")
        covered = sorted(index for batch in base_batches for index in batch)
        if covered != list(range(len(self.lengths))):
            raise RuntimeError("Batch sampler does not cover every example exactly once")
        self.base_batches = base_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = [list(batch) for batch in self.base_batches]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self.base_batches)


def collate_logmel(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([int(item["num_frames"]) for item in items], dtype=torch.long)
    maximum = int(lengths.max().item())
    feature_dim = int(items[0]["features"].shape[1])
    features = torch.zeros(len(items), maximum, feature_dim, dtype=torch.float32)
    valid_mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for index, item in enumerate(items):
        sequence = item["features"]
        length = int(sequence.shape[0])
        features[index, :length] = sequence
        valid_mask[index, :length] = True
    padded_frames = int(len(items) * maximum)
    real_frames = int(lengths.sum().item())
    return {
        "features": features,
        "frame_lengths": lengths,
        "relative_lengths": lengths.to(torch.float32) / float(maximum),
        "valid_mask": valid_mask,
        "lid": torch.tensor([int(item["lid"]) for item in items], dtype=torch.long),
        "dialect": torch.tensor([int(item["dialect"]) for item in items], dtype=torch.long),
        "gender": torch.tensor([int(item["gender"]) for item in items], dtype=torch.long),
        "language": [str(item["language"]) for item in items],
        "utt_id": [str(item["utt_id"]) for item in items],
        "filepath": [str(item["filepath"]) for item in items],
        "speaker_id": [str(item["speaker_id"]) for item in items],
        "dataset_index": torch.tensor([int(item["dataset_index"]) for item in items], dtype=torch.long),
        "crop_start_frame": torch.tensor([int(item["crop_start_frame"]) for item in items], dtype=torch.long),
        "source_num_frames": torch.tensor([int(item["source_num_frames"]) for item in items], dtype=torch.long),
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_ratio": 1.0 - real_frames / padded_frames if padded_frames else 0.0,
    }


def build_dataloader(
    config: Config,
    *,
    split: str,
    domain: str,
    duration_name: str,
    training: bool,
) -> tuple[DataLoader, StableFrameBudgetBatchSampler, LogMelCacheDataset]:
    dataset = LogMelCacheDataset(
        config=config,
        split=split,
        domain=domain,
        duration_name=duration_name,
        training=training,
    )
    sampler = StableFrameBudgetBatchSampler(
        dataset.effective_lengths(),
        max_batch_frames=int(config.max_batch_frames),
        max_batch_size=int(config.max_batch_size),
        number_of_buckets=int(config.number_of_buckets),
        seed=int(config.seed),
        shuffle=training,
    )
    workers = int(config.num_workers)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(config.pin_memory),
        "collate_fn": collate_logmel,
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=bool(config.persistent_workers),
            prefetch_factor=int(config.prefetch_factor),
        )
    return DataLoader(**kwargs), sampler, dataset


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "features", "frame_lengths", "relative_lengths", "valid_mask",
        "lid", "dialect", "gender", "dataset_index", "crop_start_frame",
        "source_num_frames",
    ):
        moved[key] = batch[key].to(device=device, non_blocking=True)
    return moved

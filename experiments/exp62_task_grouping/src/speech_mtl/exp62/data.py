"""Read-only full-sequence cache dataset and stable frame-budget batching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import csv
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .config import ExperimentConfig
from .labels import LabelSpace


@dataclass(frozen=True)
class CacheRow:
    utt_id: str
    start_frame: int
    num_frames: int
    language: str
    dialect_raw: str
    gender_raw: str
    filepath: str
    audio_samples: int


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        split_root: Path,
        label_space: LabelSpace,
        feature_dim: int = 768,
    ) -> None:
        self.split_root = Path(split_root)
        self.index_path = self.split_root / "index.csv"
        self.features_path = self.split_root / "features.f16"
        self.label_space = label_space
        self.feature_dim = int(feature_dim)
        self.rows = self._read_rows()
        self.lengths = [row.num_frames for row in self.rows]
        self._memmap: np.memmap | None = None
        self._validate_binary()

    def _read_rows(self) -> list[CacheRow]:
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        rows: list[CacheRow] = []
        seen: set[str] = set()
        expected_start = 0
        with self.index_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "utt_id", "start_frame", "num_frames", "dialect", "gender", "language"
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns in {self.index_path}: {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                utt_id = str(row["utt_id"])
                if utt_id in seen:
                    raise ValueError(f"Duplicate utt_id {utt_id!r} at line {line_number}")
                seen.add(utt_id)
                start = int(row["start_frame"])
                length = int(row["num_frames"])
                if start != expected_start:
                    raise ValueError(
                        f"Non-contiguous cache index at line {line_number}: "
                        f"expected start {expected_start}, got {start}"
                    )
                if length <= 0:
                    raise ValueError(f"Non-positive length for {utt_id}: {length}")
                expected_start = start + length
                lang = str(row["language"])
                if lang not in self.label_space.language_to_index:
                    raise ValueError(f"Unknown language {lang!r} for {utt_id}")
                dialect_raw = str(row["dialect"])
                if dialect_raw not in self.label_space.dialect_to_index_by_language[lang]:
                    raise ValueError(f"Unknown dialect {dialect_raw!r} for language {lang}")
                gender_raw = str(row["gender"])
                if gender_raw not in self.label_space.gender_to_index:
                    raise ValueError(f"Unknown gender {gender_raw!r}")
                rows.append(
                    CacheRow(
                        utt_id=utt_id,
                        start_frame=start,
                        num_frames=length,
                        language=lang,
                        dialect_raw=dialect_raw,
                        gender_raw=gender_raw,
                        filepath=str(row.get("filepath", "")),
                        audio_samples=int(float(row.get("audio_samples", 0) or 0)),
                    )
                )
        if not rows:
            raise ValueError(f"Empty index: {self.index_path}")
        return rows

    def _validate_binary(self) -> None:
        if not self.features_path.is_file():
            raise FileNotFoundError(self.features_path)
        expected_frames = self.rows[-1].start_frame + self.rows[-1].num_frames
        expected_bytes = expected_frames * self.feature_dim * np.dtype(np.float16).itemsize
        actual_bytes = self.features_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Binary size mismatch for {self.features_path}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        self.total_frames = expected_frames

    def _get_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.features_path,
                mode="r",
                dtype=np.float16,
                shape=(self.total_frames, self.feature_dim),
                order="C",
            )
        return self._memmap

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_memmap"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        source = self._get_memmap()[row.start_frame : row.start_frame + row.num_frames]
        features = np.array(source, dtype=np.float32, order="C", copy=True)
        return {
            "features": torch.from_numpy(features),
            "utt_id": row.utt_id,
            "num_frames": row.num_frames,
            "lid": self.label_space.language_to_index[row.language],
            "dialect": self.label_space.dialect_to_index_by_language[row.language][row.dialect_raw],
            "gender": self.label_space.gender_to_index[row.gender_raw],
            "language": row.language,
            "filepath": row.filepath,
            "audio_samples": row.audio_samples,
            "dataset_index": index,
        }


class StableFrameBudgetBatchSampler(Sampler[list[int]]):
    """Build batches once, then reshuffle their order deterministically per epoch."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_batch_frames: int,
        max_batch_size: int,
        number_of_buckets: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.lengths = [int(x) for x in lengths]
        self.max_batch_frames = int(max_batch_frames)
        self.max_batch_size = int(max_batch_size)
        self.number_of_buckets = int(number_of_buckets)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._base_batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        ordered = sorted(range(len(self.lengths)), key=lambda i: (self.lengths[i], i))
        bucket_size = max(1, math.ceil(len(ordered) / self.number_of_buckets))
        batches: list[list[int]] = []
        for offset in range(0, len(ordered), bucket_size):
            bucket = ordered[offset : offset + bucket_size]
            current: list[int] = []
            current_max = 0
            for index in bucket:
                length = self.lengths[index]
                proposed_max = max(current_max, length)
                proposed_size = len(current) + 1
                exceeds = (
                    proposed_size > self.max_batch_size
                    or proposed_max * proposed_size > self.max_batch_frames
                )
                if current and exceeds:
                    batches.append(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, length)
                if len(current) == 1 and length > self.max_batch_frames:
                    batches.append(current)
                    current = []
                    current_max = 0
            if current:
                batches.append(current)
        if not batches:
            raise ValueError("No batches were constructed")
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches = [list(batch) for batch in self._base_batches]
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._base_batches)


def collate_full_sequences(examples: list[dict]) -> dict:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([x["num_frames"] for x in examples], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = int(examples[0]["features"].shape[1])
    features = torch.zeros((len(examples), max_len, feature_dim), dtype=torch.float32)
    valid_mask = torch.zeros((len(examples), max_len), dtype=torch.bool)
    for i, example in enumerate(examples):
        length = int(example["num_frames"])
        features[i, :length] = example["features"]
        valid_mask[i, :length] = True
    padded_frames = int(max_len * len(examples))
    real_frames = int(lengths.sum().item())
    return {
        "features": features,
        "frame_lengths": lengths,
        "relative_lengths": lengths.to(torch.float32) / float(max_len),
        "valid_mask": valid_mask,
        "lid": torch.tensor([x["lid"] for x in examples], dtype=torch.long),
        "dialect": torch.tensor([x["dialect"] for x in examples], dtype=torch.long),
        "gender": torch.tensor([x["gender"] for x in examples], dtype=torch.long),
        "language": [x["language"] for x in examples],
        "utt_id": [x["utt_id"] for x in examples],
        "filepath": [x["filepath"] for x in examples],
        "dataset_index": torch.tensor([x["dataset_index"] for x in examples], dtype=torch.long),
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_ratio": 1.0 - (real_frames / padded_frames),
    }


def build_dataset(cfg: ExperimentConfig, domain: str, split: str, labels: LabelSpace) -> CachedFeatureDataset:
    return CachedFeatureDataset(
        cfg.paths.cache_root / domain / split,
        label_space=labels,
        feature_dim=cfg.data.feature_dim,
    )


def build_loader(
    cfg: ExperimentConfig,
    dataset: CachedFeatureDataset,
    *,
    train: bool,
) -> tuple[DataLoader, StableFrameBudgetBatchSampler]:
    sampler = StableFrameBudgetBatchSampler(
        dataset.lengths,
        max_batch_frames=cfg.data.max_batch_frames,
        max_batch_size=cfg.data.max_batch_size,
        number_of_buckets=cfg.data.number_of_buckets,
        seed=cfg.training.seed,
        shuffle=train,
    )
    kwargs = dict(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_full_sequences,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    if cfg.data.num_workers > 0:
        kwargs.update(
            persistent_workers=cfg.data.persistent_workers,
            prefetch_factor=cfg.data.prefetch_factor,
        )
    return DataLoader(**kwargs), sampler

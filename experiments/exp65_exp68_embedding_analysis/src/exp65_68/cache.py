from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class CachedSequenceDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    def __init__(self, frame: pd.DataFrame, features_path: Path, feature_dimension: int):
        self.frame = frame.reset_index(drop=True)
        self.features_path = str(features_path)
        self.feature_dimension = int(feature_dimension)
        bytes_per_frame = self.feature_dimension * np.dtype(np.float16).itemsize
        file_bytes = features_path.stat().st_size
        if file_bytes % bytes_per_frame:
            raise RuntimeError(f"invalid feature binary size: {features_path}")
        self.total_frames = file_bytes // bytes_per_frame
        self._memmap: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.features_path,
                mode="r",
                dtype=np.float16,
                shape=(self.total_frames, self.feature_dimension),
            )
        row = self.frame.iloc[index]
        start = int(row.start_frame)
        length = int(row.num_frames)
        stop = start + length
        if start < 0 or length <= 0 or stop > self.total_frames:
            raise RuntimeError(f"cache bounds failure at row {index}: {start}+{length}")
        sequence = torch.from_numpy(np.array(self._memmap[start:stop], copy=True))
        return sequence, row.to_dict()


class FrameBudgetSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], frame_budget: int, maximum_batch_size: int):
        self.batches: list[list[int]] = []
        current: list[int] = []
        current_maximum = 0
        for index, raw_length in enumerate(lengths):
            length = int(raw_length)
            proposed_maximum = max(current_maximum, length)
            proposed_size = len(current) + 1
            over = proposed_size > maximum_batch_size or proposed_size * proposed_maximum > frame_budget
            if current and over:
                self.batches.append(current)
                current = []
                current_maximum = 0
            current.append(index)
            current_maximum = max(current_maximum, length)
        if current:
            self.batches.append(current)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def collate_cached(items: list[tuple[torch.Tensor, dict[str, Any]]]):
    lengths = torch.tensor([sequence.shape[0] for sequence, _ in items], dtype=torch.long)
    maximum = int(lengths.max().item())
    dimension = int(items[0][0].shape[1])
    features = torch.zeros((len(items), maximum, dimension), dtype=torch.float16)
    valid_mask = torch.zeros((len(items), maximum), dtype=torch.bool)
    metadata: list[dict[str, Any]] = []
    for index, (sequence, row) in enumerate(items):
        length = int(sequence.shape[0])
        features[index, :length] = sequence
        valid_mask[index, :length] = True
        metadata.append(row)
    relative_lengths = lengths.to(torch.float32) / float(maximum)
    return features, relative_lengths, valid_mask, metadata


def build_loader(
    frame: pd.DataFrame,
    features_path: Path,
    feature_dimension: int,
    frame_budget: int,
    maximum_batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = CachedSequenceDataset(frame, features_path, feature_dimension)
    sampler = FrameBudgetSampler(
        frame["num_frames"].astype(int).tolist(),
        frame_budget=frame_budget,
        maximum_batch_size=maximum_batch_size,
    )
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": True,
        "collate_fn": collate_cached,
    }
    if workers > 0:
        arguments.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**arguments)

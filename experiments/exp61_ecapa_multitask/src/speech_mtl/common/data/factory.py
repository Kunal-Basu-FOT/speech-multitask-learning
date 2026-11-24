from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .cache_dataset import CachedFeatureDataset
from .collate import pad_cached_feature_batch
from .dynamic_batching import FrameBudgetBatchSampler


VALID_DOMAINS = {"one_bit", "sixteen_bit"}
VALID_SPLITS = {"train", "dev", "test"}


def build_dataset(
    *,
    cache_root: Path | str,
    domain: str,
    split: str,
    feature_dimension: int = 768,
) -> CachedFeatureDataset:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unsupported cache domain: {domain}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    split_root = Path(cache_root) / domain / split

    return CachedFeatureDataset(
        split_root=split_root,
        feature_dimension=feature_dimension,
    )


def build_dataloader(
    dataset: CachedFeatureDataset,
    *,
    training: bool,
    max_batch_frames: int = 12000,
    max_batch_size: int = 32,
    number_of_buckets: int = 40,
    seed: int = 1337,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> tuple[DataLoader, FrameBudgetBatchSampler]:
    sampler = FrameBudgetBatchSampler(
        lengths=dataset.lengths,
        max_batch_frames=max_batch_frames,
        max_batch_size=max_batch_size,
        number_of_buckets=number_of_buckets,
        shuffle=training,
        seed=seed,
        drop_last=False,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": pad_cached_feature_batch,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(**loader_kwargs)
    return loader, sampler

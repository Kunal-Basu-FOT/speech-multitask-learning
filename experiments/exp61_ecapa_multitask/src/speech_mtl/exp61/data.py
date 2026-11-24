from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from speech_mtl.common.data.cache_dataset import CachedFeatureDataset
from speech_mtl.common.data.collate import pad_cached_feature_batch

from .batching import StableFrameBudgetBatchSampler
from .config import Exp61Config, VALID_DOMAINS, VALID_SPLITS


def build_exp61_dataset(
    config: Exp61Config,
    *,
    domain: str,
    split: str,
) -> CachedFeatureDataset:
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")

    return CachedFeatureDataset(
        split_root=config.cache_root / domain / split,
        feature_dimension=config.feature_dimension,
    )


def build_exp61_dataloader(
    config: Exp61Config,
    *,
    domain: str,
    split: str,
    training: bool,
) -> tuple[
    DataLoader,
    StableFrameBudgetBatchSampler,
    CachedFeatureDataset,
]:
    dataset = build_exp61_dataset(
        config,
        domain=domain,
        split=split,
    )

    sampler = StableFrameBudgetBatchSampler(
        lengths=dataset.lengths,
        max_batch_frames=config.max_batch_frames,
        max_batch_size=config.max_batch_size,
        number_of_buckets=config.number_of_buckets,
        shuffle=training,
        seed=config.seed,
        drop_last=False,
    )

    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": pad_cached_feature_batch,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }

    if config.num_workers > 0:
        loader_arguments["persistent_workers"] = (
            config.persistent_workers
        )
        loader_arguments["prefetch_factor"] = (
            config.prefetch_factor
        )

    loader = DataLoader(**loader_arguments)
    return loader, sampler, dataset

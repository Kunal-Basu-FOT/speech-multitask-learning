from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from speech_mtl.common.data.dynamic_batching import (
    FrameBudgetBatchSampler,
)


class StableFrameBudgetBatchSampler(FrameBudgetBatchSampler):
    """Frame-budget sampler with a constant optimizer-step count.

    The sorted, length-bucketed batch composition is packed once. Each epoch
    shuffles batch order and the example order inside each batch. This keeps
    the padded-frame budget and number of batches invariant across all 40
    epochs while still changing presentation order deterministically.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_batch_frames: int,
        max_batch_size: int,
        number_of_buckets: int = 40,
        shuffle: bool = True,
        seed: int = 1337,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            lengths=lengths,
            max_batch_frames=max_batch_frames,
            max_batch_size=max_batch_size,
            number_of_buckets=number_of_buckets,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        base_batches: list[list[int]] = []

        for bucket in self._create_buckets():
            base_batches.extend(self._pack_bucket(bucket))

        if not base_batches:
            raise RuntimeError("Stable batch sampler produced no batches")

        self._base_batches = tuple(
            tuple(batch) for batch in base_batches
        )

    def _build_batches(self) -> list[list[int]]:
        batches = [
            list(batch) for batch in self._base_batches
        ]

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)

            for batch in batches:
                rng.shuffle(batch)

            rng.shuffle(batches)

        return batches

    def __len__(self) -> int:
        return len(self._base_batches)

    @property
    def optimizer_batch_count(self) -> int:
        return len(self._base_batches)

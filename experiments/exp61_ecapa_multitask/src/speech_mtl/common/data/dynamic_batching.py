from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Length-bucketed batches constrained by padded frame cost.

    The budget is computed as:
        longest_sequence_in_batch * batch_size

    An utterance longer than ``max_batch_frames`` is retained as a singleton.
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
        if not lengths:
            raise ValueError("lengths cannot be empty")
        if max_batch_frames <= 0:
            raise ValueError("max_batch_frames must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if number_of_buckets <= 0:
            raise ValueError("number_of_buckets must be positive")

        self.lengths = [int(value) for value in lengths]

        if any(value <= 0 for value in self.lengths):
            raise ValueError("All sequence lengths must be positive")

        self.max_batch_frames = int(max_batch_frames)
        self.max_batch_size = int(max_batch_size)
        self.number_of_buckets = int(number_of_buckets)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _create_buckets(self) -> list[list[int]]:
        sorted_indices = sorted(
            range(len(self.lengths)),
            key=self.lengths.__getitem__,
        )
        bucket_size = max(
            1,
            math.ceil(len(sorted_indices) / self.number_of_buckets),
        )

        return [
            sorted_indices[start : start + bucket_size]
            for start in range(0, len(sorted_indices), bucket_size)
        ]

    def _pack_bucket(self, indices: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_maximum = 0

        for index in indices:
            length = self.lengths[index]

            if length > self.max_batch_frames:
                if current:
                    batches.append(current)
                    current = []
                    current_maximum = 0
                batches.append([index])
                continue

            candidate_maximum = max(current_maximum, length)
            candidate_size = len(current) + 1
            candidate_cost = candidate_maximum * candidate_size

            exceeds_budget = candidate_cost > self.max_batch_frames
            exceeds_size = candidate_size > self.max_batch_size

            if current and (exceeds_budget or exceeds_size):
                batches.append(current)
                current = [index]
                current_maximum = length
            else:
                current.append(index)
                current_maximum = candidate_maximum

        if current:
            if not self.drop_last or len(current) == self.max_batch_size:
                batches.append(current)

        return batches

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets = self._create_buckets()

        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)

        batches: list[list[int]] = []
        for bucket in buckets:
            batches.extend(self._pack_bucket(bucket))

        if self.shuffle:
            rng.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())

    def diagnostics(self) -> dict[str, float | int]:
        batches = self._build_batches()
        real_frames = 0
        padded_frames = 0
        largest_batch = 0
        over_budget_singletons = 0

        for batch in batches:
            batch_lengths = [self.lengths[index] for index in batch]
            real_frames += sum(batch_lengths)
            padded_frames += max(batch_lengths) * len(batch_lengths)
            largest_batch = max(largest_batch, len(batch))

            if (
                len(batch) == 1
                and batch_lengths[0] > self.max_batch_frames
            ):
                over_budget_singletons += 1

        padding_ratio = (
            1.0 - (real_frames / padded_frames)
            if padded_frames
            else 0.0
        )

        return {
            "examples": len(self.lengths),
            "batches": len(batches),
            "real_frames": real_frames,
            "padded_frames": padded_frames,
            "padding_ratio": float(padding_ratio),
            "largest_batch": largest_batch,
            "over_budget_singletons": over_budget_singletons,
        }

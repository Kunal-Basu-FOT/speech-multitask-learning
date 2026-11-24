from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .factory import build_dataset


def validate_real_cache(
    cache_root: Path | str,
    *,
    expected_counts: dict[str, int] | None = None,
    feature_dimension: int = 768,
) -> dict[str, Any]:
    """Run lightweight real-cache checks without scanning all feature values."""

    expected_counts = expected_counts or {
        "train": 146_359,
        "dev": 11_507,
        "test": 17_550,
    }

    report: dict[str, Any] = {"status": "PASS", "domains": {}}

    for domain in ("one_bit", "sixteen_bit"):
        report["domains"][domain] = {}

        for split in ("train", "dev", "test"):
            dataset = build_dataset(
                cache_root=cache_root,
                domain=domain,
                split=split,
                feature_dimension=feature_dimension,
            )

            expected = expected_counts[split]
            if len(dataset) != expected:
                raise RuntimeError(
                    f"{domain}/{split}: expected {expected:,}, "
                    f"found {len(dataset):,}"
                )

            sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
            samples = []

            for index in sample_indices:
                item = dataset[index]

                if tuple(item["features"].shape) != (
                    item["num_frames"],
                    feature_dimension,
                ):
                    raise RuntimeError(
                        f"Invalid feature shape for {domain}/{split}/"
                        f"{item['utt_id']}"
                    )

                if not torch.isfinite(item["features"]).all():
                    raise RuntimeError(
                        f"Non-finite feature values for "
                        f"{domain}/{split}/{item['utt_id']}"
                    )

                samples.append(
                    {
                        "dataset_index": index,
                        "utt_id": item["utt_id"],
                        "num_frames": item["num_frames"],
                    }
                )

            report["domains"][domain][split] = {
                "rows": len(dataset),
                "total_frames": dataset.total_frames,
                "minimum_frames": min(dataset.lengths),
                "maximum_frames": max(dataset.lengths),
                "samples": samples,
            }

            dataset.close()

    return report

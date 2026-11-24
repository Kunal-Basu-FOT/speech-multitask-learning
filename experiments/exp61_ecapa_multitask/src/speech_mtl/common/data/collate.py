from __future__ import annotations

from typing import Any, Sequence

import torch


def pad_cached_feature_batch(
    examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Pad full feature sequences without cropping or truncation."""

    if not examples:
        raise ValueError("Cannot collate an empty batch")

    feature_dimension = int(examples[0]["features"].shape[1])
    feature_dtype = examples[0]["features"].dtype

    frame_lengths = torch.tensor(
        [int(item["num_frames"]) for item in examples],
        dtype=torch.long,
    )

    maximum_frames = int(frame_lengths.max().item())
    batch_size = len(examples)

    features = torch.zeros(
        (batch_size, maximum_frames, feature_dimension),
        dtype=feature_dtype,
    )
    valid_mask = torch.zeros(
        (batch_size, maximum_frames),
        dtype=torch.bool,
    )

    for row_index, item in enumerate(examples):
        sequence = item["features"]

        if sequence.ndim != 2:
            raise ValueError(
                f"Expected [time, feature], received {tuple(sequence.shape)}"
            )

        if int(sequence.shape[1]) != feature_dimension:
            raise ValueError("Feature dimension differs within the batch")

        length = int(sequence.shape[0])

        if length != int(item["num_frames"]):
            raise ValueError(
                f"Index/feature length mismatch for utterance {item['utt_id']}"
            )

        features[row_index, :length].copy_(sequence)
        valid_mask[row_index, :length] = True

    relative_lengths = frame_lengths.to(torch.float32) / float(maximum_frames)
    real_frames = int(frame_lengths.sum().item())
    padded_frames = int(batch_size * maximum_frames)
    padding_ratio = (
        1.0 - (real_frames / padded_frames)
        if padded_frames > 0
        else 0.0
    )

    return {
        "features": features,
        "frame_lengths": frame_lengths,
        "relative_lengths": relative_lengths,
        "valid_mask": valid_mask,
        "lid": torch.tensor(
            [int(item["lid"]) for item in examples],
            dtype=torch.long,
        ),
        "dialect": torch.tensor(
            [int(item["dialect"]) for item in examples],
            dtype=torch.long,
        ),
        "gender": torch.tensor(
            [int(item["gender"]) for item in examples],
            dtype=torch.long,
        ),
        "language": [str(item["language"]) for item in examples],
        "utt_id": [str(item["utt_id"]) for item in examples],
        "filepath": [str(item["filepath"]) for item in examples],
        "dataset_index": torch.tensor(
            [int(item["dataset_index"]) for item in examples],
            dtype=torch.long,
        ),
        "real_frames": real_frames,
        "padded_frames": padded_frames,
        "padding_ratio": float(padding_ratio),
    }

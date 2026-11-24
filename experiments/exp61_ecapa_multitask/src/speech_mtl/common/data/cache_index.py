from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_COLUMNS = {
    "utt_id",
    "start_frame",
    "num_frames",
    "lid",
    "dialect",
    "gender",
    "language",
    "filepath",
    "audio_samples",
}


@dataclass(frozen=True, slots=True)
class CacheRecord:
    """One utterance entry in a cache-native index."""

    position: int
    utt_id: str
    start_frame: int
    num_frames: int
    lid: int
    dialect: int
    gender: int
    language: str
    filepath: str
    audio_samples: int


def load_cache_index(path: Path | str) -> list[CacheRecord]:
    """Read and strictly validate a cache-native index.csv file."""

    index_path = Path(path)

    if not index_path.is_file():
        raise FileNotFoundError(f"Cache index does not exist: {index_path}")

    records: list[CacheRecord] = []
    seen_utterance_ids: set[str] = set()
    expected_start = 0

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)

        if missing:
            raise ValueError(f"{index_path} is missing required columns: {missing}")

        for position, row in enumerate(reader):
            utt_id = str(row["utt_id"]).strip()
            language = str(row["language"]).strip()
            filepath = str(row["filepath"]).strip()

            if not utt_id:
                raise ValueError(f"Empty utt_id at index position {position}")

            if utt_id in seen_utterance_ids:
                raise ValueError(f"Duplicate utt_id in {index_path}: {utt_id}")

            seen_utterance_ids.add(utt_id)

            try:
                start_frame = int(row["start_frame"])
                num_frames = int(row["num_frames"])
                lid = int(row["lid"])
                dialect = int(row["dialect"])
                gender = int(row["gender"])
                audio_samples = int(row["audio_samples"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed numeric field at index position {position}"
                ) from exc

            if start_frame != expected_start:
                raise ValueError(
                    f"Non-contiguous frame offset at position {position}: "
                    f"expected {expected_start}, found {start_frame}"
                )

            if num_frames <= 0:
                raise ValueError(
                    f"Invalid num_frames at position {position}: {num_frames}"
                )

            if audio_samples <= 0:
                raise ValueError(
                    f"Invalid audio_samples at position {position}: {audio_samples}"
                )

            if not language:
                raise ValueError(f"Empty language at index position {position}")

            if not filepath:
                raise ValueError(f"Empty filepath at index position {position}")

            records.append(
                CacheRecord(
                    position=position,
                    utt_id=utt_id,
                    start_frame=start_frame,
                    num_frames=num_frames,
                    lid=lid,
                    dialect=dialect,
                    gender=gender,
                    language=language,
                    filepath=filepath,
                    audio_samples=audio_samples,
                )
            )

            expected_start = start_frame + num_frames

    if not records:
        raise ValueError(f"Cache index is empty: {index_path}")

    return records


def total_frames(records: Sequence[CacheRecord]) -> int:
    """Return the total number of contiguous frames represented by records."""

    if not records:
        return 0

    last = records[-1]
    return last.start_frame + last.num_frames


def frame_lengths(records: Iterable[CacheRecord]) -> list[int]:
    return [record.num_frames for record in records]


def validate_binary_size(
    features_path: Path | str,
    records: Sequence[CacheRecord],
    feature_dimension: int,
    bytes_per_value: int = 2,
) -> int:
    """Validate raw float binary size and return its frame count."""

    binary_path = Path(features_path)

    if not binary_path.is_file():
        raise FileNotFoundError(f"Feature binary does not exist: {binary_path}")

    frames = total_frames(records)
    expected_bytes = frames * int(feature_dimension) * int(bytes_per_value)
    actual_bytes = binary_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            "Feature-binary size mismatch: "
            f"path={binary_path}, expected={expected_bytes}, actual={actual_bytes}"
        )

    return frames

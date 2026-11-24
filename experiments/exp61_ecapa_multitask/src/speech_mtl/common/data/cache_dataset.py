from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache_index import (
    CacheRecord,
    frame_lengths,
    load_cache_index,
    validate_binary_size,
)


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Memory-mapped full-utterance IndicWav2Vec feature dataset.

    Expected split layout:
        <split_root>/index.csv
        <split_root>/features.f16

    The historical ``features_path`` stored in metadata.json is intentionally
    ignored. The local binary beside the split index is always used.
    """

    def __init__(
        self,
        split_root: Path | str,
        *,
        index_path: Path | str | None = None,
        feature_dimension: int = 768,
        output_dtype: torch.dtype = torch.float32,
        verify_binary_size: bool = True,
    ) -> None:
        super().__init__()

        self.split_root = Path(split_root)
        self.index_path = (
            Path(index_path)
            if index_path is not None
            else self.split_root / "index.csv"
        )
        self.features_path = self.split_root / "features.f16"
        self.feature_dimension = int(feature_dimension)
        self.output_dtype = output_dtype

        if self.feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")

        self.records: list[CacheRecord] = load_cache_index(self.index_path)

        if verify_binary_size:
            self.total_frames = validate_binary_size(
                features_path=self.features_path,
                records=self.records,
                feature_dimension=self.feature_dimension,
                bytes_per_value=2,
            )
        else:
            final = self.records[-1]
            self.total_frames = final.start_frame + final.num_frames

        self.lengths = frame_lengths(self.records)
        self._memmap: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _get_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.features_path,
                mode="r",
                dtype=np.float16,
                shape=(self.total_frames, self.feature_dimension),
                order="C",
            )

        return self._memmap

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        start = record.start_frame
        stop = start + record.num_frames

        view = self._get_memmap()[start:stop]

        # Make a writable, C-contiguous float32 copy before Torch conversion.
        array = np.array(view, dtype=np.float32, copy=True, order="C")
        features = torch.from_numpy(array)

        if self.output_dtype != torch.float32:
            features = features.to(dtype=self.output_dtype)

        return {
            "features": features,
            "utt_id": record.utt_id,
            "num_frames": record.num_frames,
            "lid": record.lid,
            "dialect": record.dialect,
            "gender": record.gender,
            "language": record.language,
            "filepath": record.filepath,
            "audio_samples": record.audio_samples,
            "dataset_index": index,
        }

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers must open independent read-only memmaps.
        state = self.__dict__.copy()
        state["_memmap"] = None
        return state

    def close(self) -> None:
        if self._memmap is not None:
            mmap_object = getattr(self._memmap, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
            self._memmap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

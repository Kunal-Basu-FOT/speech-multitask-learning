from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from .config import Config
from .utils import atomic_json, sha256_file, utc_now


CACHE_COLUMNS = [
    "utt_id",
    "start_frame",
    "num_frames",
    "lid",
    "dialect",
    "gender",
    "language",
    "filepath",
    "audio_samples",
    "speaker_id",
]


def sign_quantize(waveform: torch.Tensor) -> torch.Tensor:
    # Canonical Scheme 1: +1 when x > 0, otherwise -1.
    return torch.where(waveform > 0, torch.ones_like(waveform), -torch.ones_like(waveform))


class _OrderedRangeSampler(Sampler[int]):
    def __init__(self, start: int, stop: int) -> None:
        self.start = int(start)
        self.stop = int(stop)

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return max(self.stop - self.start, 0)


class AudioToDualLogMelDataset(Dataset):
    """Decode one waveform once and derive both 1-bit and 16-bit log-Mel features."""

    def __init__(self, frame: pd.DataFrame, config: Config) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sample_rate = int(config.sample_rate)
        self.n_fft = int(config.n_fft)
        self.win_length = int(config.win_length)
        self.hop_length = int(config.hop_length)
        self.n_mels = int(config.n_mels)
        self._mel = None

    def __len__(self) -> int:
        return len(self.frame)

    def _transform(self):
        if self._mel is None:
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                pad=0,
                n_mels=self.n_mels,
                power=2.0,
                normalized=False,
                center=False,
                norm=None,
                mel_scale="htk",
            )
        return self._mel

    def _logmel(self, waveform: torch.Tensor) -> np.ndarray:
        mel = self._transform()(waveform)
        if mel.ndim != 3 or mel.shape[0] != 1:
            raise RuntimeError(f"Unexpected MelSpectrogram shape: {tuple(mel.shape)}")
        features = torch.log(mel.clamp_min(1.0e-10)).squeeze(0).transpose(0, 1).contiguous()
        if features.ndim != 2 or features.shape[1] != self.n_mels:
            raise RuntimeError(f"Unexpected log-Mel shape: {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise RuntimeError("Non-finite log-Mel values")
        return features.to(torch.float16).cpu().numpy()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row.filepath))
        if not path.is_file():
            raise FileNotFoundError(path)

        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Sample-rate mismatch for {path}: expected {self.sample_rate}, got {sample_rate}"
            )
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise RuntimeError(f"Expected mono waveform for {path}, got {tuple(waveform.shape)}")
        waveform = waveform.to(torch.float32).contiguous()
        expected_samples = int(row.audio_samples)
        if int(waveform.shape[1]) != expected_samples:
            raise RuntimeError(
                f"Audio-sample mismatch for {path}: index={expected_samples} decoded={waveform.shape[1]}"
            )
        if not torch.isfinite(waveform).all():
            raise RuntimeError(f"Non-finite waveform samples: {path}")
        if waveform.numel() < self.win_length:
            raise RuntimeError(f"Waveform shorter than one analysis frame: {path}")
        if float(waveform.abs().max().item()) > 1.0001:
            raise RuntimeError(f"Waveform is not normalized to [-1,1]: {path}")

        sixteen = self._logmel(waveform)
        one = self._logmel(sign_quantize(waveform))
        if one.shape != sixteen.shape:
            raise RuntimeError(f"1-bit/16-bit feature shape mismatch for {path}")

        return {
            "row_index": int(index),
            "one_bit": one,
            "sixteen_bit": sixteen,
            "audio_samples_actual": int(waveform.shape[1]),
        }


def _identity(value):
    return value


def _validate_source_index(path: Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"utt_id": str, "language": str, "filepath": str})
    required = {
        "utt_id", "lid", "dialect", "gender", "language", "filepath", "audio_samples"
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Source index missing columns {missing}: {path}")
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows in {path}, found {len(frame)}")
    if frame["utt_id"].duplicated().any():
        raise RuntimeError(f"Duplicate utt_id in {path}")
    return frame


def _partial_state(split_root: Path) -> dict[str, Path]:
    return {
        "one": split_root / "features.one_bit.f16.partial",
        "sixteen": split_root / "features.sixteen_bit.f16.partial",
        "index": split_root / "index.csv.partial",
        "progress": split_root / "progress.json",
    }


def _read_progress(paths: dict[str, Path]) -> dict[str, int]:
    if not paths["progress"].is_file():
        return {"rows_written": 0, "frames_written": 0}
    value = json.loads(paths["progress"].read_text(encoding="utf-8"))
    return {
        "rows_written": int(value["rows_written"]),
        "frames_written": int(value["frames_written"]),
    }




def _rollback_partial(paths: dict[str, Path], rows: int, frames: int, feature_dim: int) -> None:
    """Roll partial files back to the last fsynced progress marker.

    A process can die after appending rows but before the next progress update.
    Those trailing bytes/CSV rows are intentionally uncommitted and are removed
    before resume. Missing committed bytes remain a hard failure.
    """
    expected_bytes = frames * feature_dim * np.dtype(np.float16).itemsize
    for key in ("one", "sixteen"):
        path = paths[key]
        if not path.exists():
            if expected_bytes:
                raise RuntimeError(f"Missing committed partial binary: {path}")
            continue
        size = path.stat().st_size
        if size < expected_bytes:
            raise RuntimeError(
                f"Partial binary is smaller than committed progress for {path}: "
                f"size={size} expected_at_least={expected_bytes}"
            )
        if size > expected_bytes:
            with path.open("r+b") as handle:
                handle.truncate(expected_bytes)

    index_path = paths["index"]
    if not index_path.exists():
        if rows:
            raise RuntimeError("Missing committed partial index")
        return
    lines = index_path.read_text(encoding="utf-8").splitlines()
    required_lines = rows + 1 if lines else 0
    if rows and len(lines) < required_lines:
        raise RuntimeError(
            f"Partial index is smaller than committed progress: "
            f"lines={len(lines)} required={required_lines}"
        )
    if not rows:
        # Preserve the header if it exists; otherwise the writer recreates it.
        retained = lines[:1]
    else:
        retained = lines[:required_lines]
    if len(retained) != len(lines):
        temporary = index_path.with_name(f".{index_path.name}.rollback")
        temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
        os.replace(temporary, index_path)


def _validate_partial(paths: dict[str, Path], rows: int, frames: int, feature_dim: int) -> None:
    expected_bytes = frames * feature_dim * np.dtype(np.float16).itemsize
    for key in ("one", "sixteen"):
        if rows == 0 and not paths[key].exists():
            continue
        if not paths[key].is_file() or paths[key].stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Partial binary mismatch for {paths[key]}: expected {expected_bytes} bytes"
            )
    if rows:
        if not paths["index"].is_file():
            raise RuntimeError("Partial index is missing")
        with paths["index"].open("rb") as handle:
            line_count = sum(1 for _ in handle)
        if line_count != rows + 1:
            raise RuntimeError(
                f"Partial index row mismatch: expected {rows}, found {line_count - 1}"
            )


def build_logmel_cache(config: Config) -> dict[str, Any]:
    source_root = config.path("source_cache_root")
    cache_root = config.path("logmel_cache_root")
    cache_root.mkdir(parents=True, exist_ok=True)

    # The two EXP 6.1 indexes must remain byte-identical.
    for split, expected in config.expected_rows.items():
        one = source_root / "one_bit" / split / "index.csv"
        sixteen = source_root / "sixteen_bit" / split / "index.csv"
        if sha256_file(one) != sha256_file(sixteen):
            raise RuntimeError(f"Cross-domain source index mismatch for {split}")
        _build_split(config, split, int(expected), one)

    marker = {
        "status": "EXP69_LOGMEL_CACHE_COMPLETE",
        "created_utc": utc_now(),
        "root": str(cache_root),
        "splits": list(config.expected_rows),
        "feature_dimension": int(config.n_mels),
        "dtype": "float16",
        "front_end": {
            "sample_rate": int(config.sample_rate),
            "win_length": int(config.win_length),
            "hop_length": int(config.hop_length),
            "n_fft": int(config.n_fft),
            "n_mels": int(config.n_mels),
            "center": False,
            "mel_scale": "htk",
            "log_floor": 1.0e-10,
            "one_bit_quantizer": "+1 if x>0 else -1",
        },
    }
    atomic_json(cache_root / "CACHE_COMPLETE.json", marker)
    return marker


def _build_split(config: Config, split: str, expected_rows: int, source_index: Path) -> None:
    split_root = config.path("logmel_cache_root") / split
    complete_path = split_root / "COMPLETE.json"
    final_one = split_root / "features.one_bit.f16"
    final_sixteen = split_root / "features.sixteen_bit.f16"
    final_index = split_root / "index.csv"

    if complete_path.is_file():
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(marker["rows"]) != expected_rows:
            raise RuntimeError(f"Completed cache row mismatch for {split}")
        for path in (final_one, final_sixteen, final_index):
            if not path.is_file():
                raise RuntimeError(f"Completed cache missing file: {path}")
        print(f"CACHE_SPLIT_SKIP_COMPLETE split={split}", flush=True)
        return

    split_root.mkdir(parents=True, exist_ok=True)
    frame = _validate_source_index(source_index, expected_rows)
    paths = _partial_state(split_root)

    # Recover a crash between individual final-file renames and the final
    # COMPLETE.json commit by moving any uncommitted finals back to partials.
    for final_path, partial_key in (
        (final_one, "one"),
        (final_sixteen, "sixteen"),
        (final_index, "index"),
    ):
        partial_path = paths[partial_key]
        if final_path.exists():
            if partial_path.exists():
                raise RuntimeError(
                    f"Both final and partial cache files exist without completion marker: "
                    f"{final_path} and {partial_path}"
                )
            os.replace(final_path, partial_path)

    progress = _read_progress(paths)
    start_row = progress["rows_written"]
    start_frame = progress["frames_written"]
    _rollback_partial(paths, start_row, start_frame, int(config.n_mels))
    _validate_partial(paths, start_row, start_frame, int(config.n_mels))

    if start_row > expected_rows:
        raise RuntimeError(f"Partial cache has too many rows for {split}")

    dataset = AudioToDualLogMelDataset(frame, config)
    sampler = _OrderedRangeSampler(start_row, len(frame))
    workers = int(config.cache_workers)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": _identity,
        "pin_memory": False,
    }
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=int(config.prefetch_factor),
        )
    loader = DataLoader(**loader_kwargs)

    one_mode = "ab" if start_row else "wb"
    index_mode = "a" if start_row else "w"
    with paths["one"].open(one_mode) as one_handle, paths["sixteen"].open(one_mode) as sixteen_handle, paths["index"].open(
        index_mode, encoding="utf-8", newline=""
    ) as index_handle:
        writer = csv.DictWriter(index_handle, fieldnames=CACHE_COLUMNS)
        if not start_row:
            writer.writeheader()

        frames_written = start_frame
        rows_written = start_row
        iterator = tqdm(
            loader,
            total=len(frame) - start_row,
            desc=f"logmel-cache/{split}",
            dynamic_ncols=True,
            leave=True,
        )
        for item in iterator:
            row_index = int(item["row_index"])
            if row_index != rows_written:
                raise RuntimeError(
                    f"DataLoader ordering failure: expected row {rows_written}, got {row_index}"
                )
            one_features = np.asarray(item["one_bit"], dtype=np.float16, order="C")
            sixteen_features = np.asarray(item["sixteen_bit"], dtype=np.float16, order="C")
            if one_features.shape != sixteen_features.shape:
                raise RuntimeError("Feature shape mismatch")
            if one_features.ndim != 2 or one_features.shape[1] != int(config.n_mels):
                raise RuntimeError(f"Invalid cached feature shape: {one_features.shape}")

            one_handle.write(one_features.tobytes(order="C"))
            sixteen_handle.write(sixteen_features.tobytes(order="C"))

            source = frame.iloc[row_index]
            num_frames = int(one_features.shape[0])
            writer.writerow(
                {
                    "utt_id": str(source.utt_id),
                    "start_frame": frames_written,
                    "num_frames": num_frames,
                    "lid": int(source.lid),
                    "dialect": int(source.dialect),
                    "gender": int(source.gender),
                    "language": str(source.language),
                    "filepath": str(source.filepath),
                    "audio_samples": int(item["audio_samples_actual"]),
                    "speaker_id": Path(str(source.filepath)).parent.name,
                }
            )
            frames_written += num_frames
            rows_written += 1

            if rows_written % 250 == 0:
                one_handle.flush()
                sixteen_handle.flush()
                index_handle.flush()
                os.fsync(one_handle.fileno())
                os.fsync(sixteen_handle.fileno())
                os.fsync(index_handle.fileno())
                atomic_json(
                    paths["progress"],
                    {
                        "status": "running",
                        "split": split,
                        "rows_written": rows_written,
                        "frames_written": frames_written,
                        "updated_utc": utc_now(),
                    },
                )
            iterator.set_postfix(rows=rows_written, frames=frames_written)

        one_handle.flush()
        sixteen_handle.flush()
        index_handle.flush()
        os.fsync(one_handle.fileno())
        os.fsync(sixteen_handle.fileno())
        os.fsync(index_handle.fileno())

    if rows_written != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, wrote {rows_written}")
    _validate_partial(paths, rows_written, frames_written, int(config.n_mels))

    os.replace(paths["one"], final_one)
    os.replace(paths["sixteen"], final_sixteen)
    os.replace(paths["index"], final_index)
    paths["progress"].unlink(missing_ok=True)

    marker = {
        "status": "LOGMEL_SPLIT_COMPLETE",
        "created_utc": utc_now(),
        "split": split,
        "rows": rows_written,
        "total_frames": frames_written,
        "feature_dimension": int(config.n_mels),
        "dtype": "float16",
        "index": str(final_index),
        "index_sha256": sha256_file(final_index),
        "one_bit_features": str(final_one),
        "one_bit_sha256": sha256_file(final_one),
        "sixteen_bit_features": str(final_sixteen),
        "sixteen_bit_sha256": sha256_file(final_sixteen),
        "source_index": str(source_index),
        "source_index_sha256": sha256_file(source_index),
    }
    atomic_json(complete_path, marker)
    print(
        f"CACHE_SPLIT_COMPLETE split={split} rows={rows_written} frames={frames_written}",
        flush=True,
    )

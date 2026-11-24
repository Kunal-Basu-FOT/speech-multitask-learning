#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


LANGUAGES = ("bh", "bn", "ch", "kn", "mg", "mr", "mt", "te")
SPLITS = ("train", "dev", "test")

EXPECTED_COUNTS = {
    "train": 146359,
    "dev": 11507,
    "test": 17550,
}

OUTPUT_COLUMNS = [
    "utt_id",
    "audio_path",
    "split",
    "language",
    "dialect",
    "dialect_local",
    "gender",
    "gender_available",
    "speaker_id",
    "duration",
    "text",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def read_cache_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {
                key: "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]

    return rows


def kaldi_directory(dataset_root: Path, language: str, split: str) -> Path:
    language_root = dataset_root / "kaldi_files" / language

    if split == "train":
        name = f"IISc_RESPIN_train_{language}_small"
    elif split == "dev":
        name = f"IISc_RESPIN_dev_{language}"
    elif split == "test":
        name = f"IISc_RESPIN_test_{language}"
    else:
        raise ValueError(f"Unsupported split: {split}")

    return language_root / name


def read_kaldi_text(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}

    if not path.is_file():
        raise FileNotFoundError(f"Missing Kaldi text file: {path}")

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")

            if not stripped:
                continue

            parts = stripped.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line in {path}:{line_number}: {stripped!r}"
                )

            utterance_id, text = parts
            result[utterance_id] = text

    return result


def read_kaldi_scalar_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}

    if not path.is_file():
        return result

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line in {path}:{line_number}: {stripped!r}"
                )

            key, value = parts
            result[key] = value

    return result


def parse_respins_id(filepath: str) -> dict[str, str]:
    full_utterance_id = Path(filepath).stem
    parts = full_utterance_id.split("_")

    if len(parts) < 8:
        raise ValueError(
            f"Unexpected RESPIN utterance ID: {full_utterance_id}"
        )

    if parts[0:2] != ["IISc", "RESPIN"]:
        raise ValueError(
            f"Unexpected RESPIN utterance prefix: {full_utterance_id}"
        )

    language = parts[2]
    dialect_local = parts[3]
    speaker_id = "_".join(parts[:5])

    gender_candidates = [
        token for token in parts[5:9] if token in {"M", "F"}
    ]

    if len(gender_candidates) != 1:
        raise ValueError(
            f"Could not resolve one gender token from {full_utterance_id}"
        )

    return {
        "full_utterance_id": full_utterance_id,
        "language": language,
        "dialect_local": dialect_local,
        "speaker_id": speaker_id,
        "gender": gender_candidates[0],
        "numeric_utterance_id": parts[-1],
    }


def find_metadata_value(
    mapping: dict[str, str],
    full_utterance_id: str,
    numeric_utterance_id: str,
) -> str | None:
    for key in (full_utterance_id, numeric_utterance_id):
        if key in mapping:
            return mapping[key]

    return None


def write_manifest_atomically(
    destination: Path,
    rows: list[dict[str, Any]],
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "policy": "CURRENT_MANIFESTS_EQUAL_EXISTING_CACHE_INDEXES",
        "cache_modified": False,
        "feature_reextraction_performed": False,
        "splits": {},
    }

    all_text_maps: dict[tuple[str, str], dict[str, str]] = {}
    all_duration_maps: dict[tuple[str, str], dict[str, str]] = {}
    all_speaker_maps: dict[tuple[str, str], dict[str, str]] = {}
    all_dialect_maps: dict[tuple[str, str], dict[str, str]] = {}

    print("===== LOADING KALDI METADATA =====")

    for split in SPLITS:
        for language in LANGUAGES:
            directory = kaldi_directory(
                dataset_root=args.dataset_root,
                language=language,
                split=split,
            )

            key = (split, language)

            all_text_maps[key] = read_kaldi_text(directory / "text")
            all_duration_maps[key] = read_kaldi_scalar_map(
                directory / "utt2dur"
            )
            all_speaker_maps[key] = read_kaldi_scalar_map(
                directory / "utt2spk"
            )
            all_dialect_maps[key] = read_kaldi_scalar_map(
                directory / "utt2dialect"
            )

            print(
                f"{split:>5} {language}: "
                f"text={len(all_text_maps[key]):,}"
            )

    print()
    print("===== VERIFYING PRECISION-DOMAIN INDEX EQUALITY =====")

    for split in SPLITS:
        one_index = args.cache_root / "one_bit" / split / "index.csv"
        sixteen_index = (
            args.cache_root / "sixteen_bit" / split / "index.csv"
        )

        one_hash = sha256_file(one_index)
        sixteen_hash = sha256_file(sixteen_index)

        if one_hash != sixteen_hash:
            raise RuntimeError(
                f"1-bit and 16-bit indexes differ for {split}"
            )

        print(f"{split:>5}: identical index sha256={one_hash}")

    print()
    print("===== BUILDING CACHE-ALIGNED MANIFESTS =====")

    for split in SPLITS:
        cache_index = args.cache_root / "one_bit" / split / "index.csv"
        cache_rows = read_cache_index(cache_index)

        if len(cache_rows) != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_COUNTS[split]:,} cache rows, "
                f"found {len(cache_rows):,}"
            )

        manifest_rows: list[dict[str, Any]] = []
        missing_text: list[str] = []
        missing_audio: list[str] = []
        label_mismatches: list[str] = []

        for cache_row in cache_rows:
            filepath = cache_row["filepath"]
            parsed = parse_respins_id(filepath)

            language = cache_row["language"]

            if parsed["language"] != language:
                label_mismatches.append(filepath)
                continue

            key = (split, language)

            full_id = parsed["full_utterance_id"]
            numeric_id = parsed["numeric_utterance_id"]

            text = find_metadata_value(
                all_text_maps[key],
                full_id,
                numeric_id,
            )

            if text is None:
                missing_text.append(full_id)
                continue

            source_dialect = find_metadata_value(
                all_dialect_maps[key],
                full_id,
                numeric_id,
            )

            dialect_local = parsed["dialect_local"]

            if source_dialect and source_dialect != dialect_local:
                label_mismatches.append(full_id)
                continue

            source_speaker = find_metadata_value(
                all_speaker_maps[key],
                full_id,
                numeric_id,
            )

            speaker_id = source_speaker or parsed["speaker_id"]

            source_duration = find_metadata_value(
                all_duration_maps[key],
                full_id,
                numeric_id,
            )

            if source_duration is not None:
                duration = float(source_duration)
            else:
                duration = int(cache_row["audio_samples"]) / 16000.0

            if not Path(filepath).is_file():
                missing_audio.append(filepath)
                continue

            manifest_rows.append(
                {
                    "utt_id": full_id,
                    "audio_path": filepath,
                    "split": split,
                    "language": language,
                    "dialect": f"{language}_{dialect_local}",
                    "dialect_local": dialect_local,
                    "gender": parsed["gender"],
                    "gender_available": 1,
                    "speaker_id": speaker_id,
                    "duration": f"{duration:.6f}",
                    "text": text,
                }
            )

        if missing_text:
            raise RuntimeError(
                f"{split}: {len(missing_text)} cache utterances have no text; "
                f"first examples={missing_text[:10]}"
            )

        if missing_audio:
            raise RuntimeError(
                f"{split}: {len(missing_audio)} audio files are missing; "
                f"first examples={missing_audio[:10]}"
            )

        if label_mismatches:
            raise RuntimeError(
                f"{split}: {len(label_mismatches)} label mismatches; "
                f"first examples={label_mismatches[:10]}"
            )

        if len(manifest_rows) != len(cache_rows):
            raise RuntimeError(
                f"{split}: built {len(manifest_rows):,} rows from "
                f"{len(cache_rows):,} cache rows"
            )

        utterance_ids = [row["utt_id"] for row in manifest_rows]
        audio_paths = [row["audio_path"] for row in manifest_rows]

        duplicate_utterance_ids = [
            value
            for value, count in Counter(utterance_ids).items()
            if count > 1
        ]

        duplicate_audio_paths = [
            value
            for value, count in Counter(audio_paths).items()
            if count > 1
        ]

        if duplicate_utterance_ids:
            raise RuntimeError(
                f"{split}: duplicate full utterance IDs found"
            )

        if duplicate_audio_paths:
            raise RuntimeError(
                f"{split}: duplicate audio paths found"
            )

        cache_paths = [row["filepath"] for row in cache_rows]

        if audio_paths != cache_paths:
            raise RuntimeError(
                f"{split}: manifest order does not exactly match cache order"
            )

        destination = (
            args.manifest_dir
            / f"respin_unique_sentence_{split}.csv"
        )

        write_manifest_atomically(destination, manifest_rows)

        report["splits"][split] = {
            "rows": len(manifest_rows),
            "manifest": str(destination),
            "manifest_sha256": sha256_file(destination),
            "cache_index": str(cache_index),
            "cache_index_sha256": sha256_file(cache_index),
            "exact_audio_path_order_match": True,
            "missing_text": 0,
            "missing_audio": 0,
            "duplicate_utterance_ids": 0,
            "duplicate_audio_paths": 0,
        }

        print(
            f"{split:>5}: wrote {len(manifest_rows):,} rows | "
            f"exact cache order"
        )

    report["status"] = "PASS"
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("CACHE_ALIGNED_MANIFEST_CREATION_PASS")


if __name__ == "__main__":
    main()

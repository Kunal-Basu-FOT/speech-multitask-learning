#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


FEATURE_DIM = 768
BYTES_PER_VALUE = 2
DOMAINS = ("one_bit", "sixteen_bit")
SPLITS = ("train", "dev", "test")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def normalized_path(value: str) -> str:
    return os.path.normpath(str(value).strip())


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            {
                key: "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]

    return fields, rows


def parse_filename_contract(filepath: str) -> dict[str, str]:
    stem = Path(filepath).stem
    parts = stem.split("_")

    result = {
        "stem": stem,
        "language_from_filename": "",
        "dialect_from_filename": "",
        "speaker_from_filename": "",
        "gender_from_filename": "",
        "numeric_utt_id_from_filename": "",
    }

    # Expected form:
    # IISc_RESPIN_<lang>_<D#>_<speaker>_<pin/NA>_<M/F>_...
    if len(parts) >= 8 and parts[0] == "IISc" and parts[1] == "RESPIN":
        result["language_from_filename"] = parts[2]
        result["dialect_from_filename"] = parts[3]
        result["speaker_from_filename"] = "_".join(parts[:5])
        result["gender_from_filename"] = parts[6]
        result["numeric_utt_id_from_filename"] = parts[-1]

    return result


def validate_cache_split(
    domain: str,
    split: str,
    split_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    errors: list[str] = []

    index_path = split_root / "index.csv"
    features_path = split_root / "features.f16"
    metadata_path = split_root / "metadata.json"
    complete_path = split_root / "COMPLETE.json"

    for required in (
        index_path,
        features_path,
        metadata_path,
        complete_path,
    ):
        if not required.is_file():
            errors.append(f"Missing cache artifact: {required}")

    if errors:
        return (
            {
                "domain": domain,
                "split": split,
                "root": str(split_root),
                "status": "FAIL",
            },
            [],
            errors,
        )

    fields, rows = read_csv(index_path)

    expected_fields = {
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

    missing_fields = sorted(expected_fields - set(fields))

    if missing_fields:
        errors.append(
            f"{domain}/{split} missing index fields: {missing_fields}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))

    if metadata.get("dtype") != "float16":
        errors.append(
            f"{domain}/{split}: metadata dtype is "
            f"{metadata.get('dtype')!r}, expected 'float16'"
        )

    if int(metadata.get("feature_dim", -1)) != FEATURE_DIM:
        errors.append(
            f"{domain}/{split}: feature_dim is "
            f"{metadata.get('feature_dim')!r}, expected {FEATURE_DIM}"
        )

    if int(metadata.get("fairseq_zero_based_layer_index", -1)) != 11:
        errors.append(
            f"{domain}/{split}: layer index is "
            f"{metadata.get('fairseq_zero_based_layer_index')!r}, expected 11"
        )

    if not bool(metadata.get("full_utterance", False)):
        errors.append(f"{domain}/{split}: full_utterance is not true")

    expected_start = 0
    total_frames = 0
    invalid_offsets = 0
    invalid_lengths = 0
    invalid_filename_contract = 0
    numeric_id_mismatches = 0
    label_filename_mismatches = 0

    language_to_lid: dict[str, set[int]] = defaultdict(set)
    gender_to_index: dict[str, set[int]] = defaultdict(set)
    dialect_to_index: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    speakers: set[str] = set()

    enriched_rows: list[dict[str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        try:
            start_frame = int(row["start_frame"])
            num_frames = int(row["num_frames"])
            lid = int(row["lid"])
            dialect_index = int(row["dialect"])
            gender_index = int(row["gender"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"{domain}/{split}: malformed index row {row_number}: {exc}"
            )
            continue

        if start_frame != expected_start:
            invalid_offsets += 1

        if num_frames <= 0:
            invalid_lengths += 1

        expected_start = start_frame + num_frames
        total_frames += num_frames

        parsed = parse_filename_contract(row["filepath"])

        language_from_filename = parsed["language_from_filename"]
        dialect_from_filename = parsed["dialect_from_filename"]
        gender_from_filename = parsed["gender_from_filename"]
        speaker_from_filename = parsed["speaker_from_filename"]
        numeric_id_from_filename = parsed["numeric_utt_id_from_filename"]

        if not language_from_filename:
            invalid_filename_contract += 1
        else:
            if language_from_filename != row["language"]:
                label_filename_mismatches += 1

            language_to_lid[row["language"]].add(lid)
            gender_to_index[gender_from_filename].add(gender_index)
            dialect_to_index[row["language"]][dialect_from_filename].add(
                dialect_index
            )

            if speaker_from_filename:
                speakers.add(speaker_from_filename)

        if numeric_id_from_filename != row["utt_id"]:
            numeric_id_mismatches += 1

        enriched = dict(row)
        enriched["_normalized_filepath"] = normalized_path(row["filepath"])
        enriched["_basename"] = Path(row["filepath"]).name
        enriched["_full_utt_stem"] = Path(row["filepath"]).stem
        enriched["_speaker_id"] = speaker_from_filename
        enriched["_gender_token"] = gender_from_filename
        enriched["_dialect_token"] = dialect_from_filename
        enriched_rows.append(enriched)

    binary_size = features_path.stat().st_size
    expected_binary_size = total_frames * FEATURE_DIM * BYTES_PER_VALUE

    if binary_size != expected_binary_size:
        errors.append(
            f"{domain}/{split}: features.f16 size mismatch: "
            f"actual={binary_size}, expected={expected_binary_size}"
        )

    if invalid_offsets:
        errors.append(
            f"{domain}/{split}: {invalid_offsets} non-contiguous offsets"
        )

    if invalid_lengths:
        errors.append(
            f"{domain}/{split}: {invalid_lengths} invalid frame lengths"
        )

    if invalid_filename_contract:
        errors.append(
            f"{domain}/{split}: {invalid_filename_contract} filenames "
            "did not match the expected RESPIN structure"
        )

    if numeric_id_mismatches:
        errors.append(
            f"{domain}/{split}: {numeric_id_mismatches} cache utt_id values "
            "do not match the final filename token"
        )

    if label_filename_mismatches:
        errors.append(
            f"{domain}/{split}: {label_filename_mismatches} language labels "
            "disagree with filenames"
        )

    for language, values in language_to_lid.items():
        if len(values) != 1:
            errors.append(
                f"{domain}/{split}: language {language} maps to "
                f"multiple LID indices: {sorted(values)}"
            )

    for gender_token, values in gender_to_index.items():
        if gender_token in {"M", "F"} and len(values) != 1:
            errors.append(
                f"{domain}/{split}: gender {gender_token} maps to "
                f"multiple indices: {sorted(values)}"
            )

    for language, dialects in dialect_to_index.items():
        for dialect_token, values in dialects.items():
            if len(values) != 1:
                errors.append(
                    f"{domain}/{split}: {language}/{dialect_token} maps to "
                    f"multiple dialect indices: {sorted(values)}"
                )

    summary = {
        "domain": domain,
        "split": split,
        "status": "PASS" if not errors else "FAIL",
        "index_path": str(index_path),
        "local_features_path": str(features_path),
        "metadata_features_path": metadata.get("features_path"),
        "loader_must_use_local_features_path": True,
        "index_sha256": sha256_file(index_path),
        "row_count": len(rows),
        "total_frames": total_frames,
        "binary_bytes": binary_size,
        "expected_binary_bytes": expected_binary_size,
        "binary_integrity": binary_size == expected_binary_size,
        "feature_dim": metadata.get("feature_dim"),
        "dtype": metadata.get("dtype"),
        "layer_index": metadata.get("fairseq_zero_based_layer_index"),
        "full_utterance": metadata.get("full_utterance"),
        "speaker_count": len(speakers),
        "language_to_lid": {
            key: sorted(values)
            for key, values in sorted(language_to_lid.items())
        },
        "gender_to_index": {
            key: sorted(values)
            for key, values in sorted(gender_to_index.items())
            if key in {"M", "F"}
        },
        "dialect_to_index": {
            language: {
                dialect: sorted(values)
                for dialect, values in sorted(dialects.items())
            }
            for language, dialects in sorted(dialect_to_index.items())
        },
        "complete_status": complete.get("status"),
        "complete_rows": (
            complete.get("rows")
            or complete.get("new_rows")
            or complete.get("utterances")
        ),
    }

    return summary, enriched_rows, errors


def manifest_cache_overlap(
    split: str,
    manifest_path: Path,
    cache_rows: list[dict[str, str]],
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    if not manifest_path.is_file():
        return (
            {
                "split": split,
                "manifest_path": str(manifest_path),
                "status": "FAIL",
            },
            [f"Missing manifest: {manifest_path}"],
        )

    manifest_fields, manifest_rows = read_csv(manifest_path)

    audio_column = (
        "audio_path"
        if "audio_path" in manifest_fields
        else "filepath"
        if "filepath" in manifest_fields
        else None
    )

    if audio_column is None:
        return (
            {
                "split": split,
                "manifest_path": str(manifest_path),
                "status": "FAIL",
            },
            [
                f"{manifest_path} has neither audio_path nor filepath column"
            ],
        )

    manifest_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    cache_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in manifest_rows:
        manifest_by_path[normalized_path(row[audio_column])].append(row)

    for row in cache_rows:
        cache_by_path[row["_normalized_filepath"]].append(row)

    manifest_paths = set(manifest_by_path)
    cache_paths = set(cache_by_path)

    shared_paths = manifest_paths & cache_paths
    manifest_only_paths = manifest_paths - cache_paths
    cache_only_paths = cache_paths - manifest_paths

    duplicate_manifest_paths = {
        key: rows
        for key, rows in manifest_by_path.items()
        if len(rows) > 1
    }

    duplicate_cache_paths = {
        key: rows
        for key, rows in cache_by_path.items()
        if len(rows) > 1
    }

    if duplicate_manifest_paths:
        errors.append(
            f"{split}: {len(duplicate_manifest_paths)} duplicate manifest paths"
        )

    if duplicate_cache_paths:
        errors.append(
            f"{split}: {len(duplicate_cache_paths)} duplicate cache paths"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_only_file = output_dir / f"manifest_only_{split}.csv"
    cache_only_file = output_dir / f"cache_only_{split}.csv"

    with manifest_only_file.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "audio_path",
            "utt_id",
            "language",
            "dialect",
            "dialect_local",
            "gender",
            "speaker_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for path in sorted(manifest_only_paths):
            for row in manifest_by_path[path]:
                writer.writerow(
                    {
                        "audio_path": path,
                        "utt_id": row.get("utt_id", ""),
                        "language": row.get("language", ""),
                        "dialect": row.get("dialect", ""),
                        "dialect_local": row.get("dialect_local", ""),
                        "gender": row.get("gender", ""),
                        "speaker_id": row.get("speaker_id", ""),
                    }
                )

    with cache_only_file.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "filepath",
            "utt_id",
            "language",
            "lid",
            "dialect",
            "gender",
            "num_frames",
            "speaker_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for path in sorted(cache_only_paths):
            for row in cache_by_path[path]:
                writer.writerow(
                    {
                        "filepath": path,
                        "utt_id": row.get("utt_id", ""),
                        "language": row.get("language", ""),
                        "lid": row.get("lid", ""),
                        "dialect": row.get("dialect", ""),
                        "gender": row.get("gender", ""),
                        "num_frames": row.get("num_frames", ""),
                        "speaker_id": row.get("_speaker_id", ""),
                    }
                )

    manifest_coverage = (
        len(shared_paths) / len(manifest_paths)
        if manifest_paths
        else 0.0
    )

    cache_coverage = (
        len(shared_paths) / len(cache_paths)
        if cache_paths
        else 0.0
    )

    summary = {
        "split": split,
        "manifest_path": str(manifest_path),
        "manifest_audio_column": audio_column,
        "manifest_rows": len(manifest_rows),
        "cache_rows": len(cache_rows),
        "manifest_unique_paths": len(manifest_paths),
        "cache_unique_paths": len(cache_paths),
        "shared_paths": len(shared_paths),
        "manifest_only_paths": len(manifest_only_paths),
        "cache_only_paths": len(cache_only_paths),
        "manifest_coverage_fraction": manifest_coverage,
        "cache_coverage_fraction": cache_coverage,
        "current_manifest_is_cache_subset": not manifest_only_paths,
        "cache_is_current_manifest_subset": not cache_only_paths,
        "manifest_only_file": str(manifest_only_file),
        "cache_only_file": str(cache_only_file),
    }

    return summary, errors


def derive_global_maps(
    split_rows: dict[str, list[dict[str, str]]]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    language_to_lid: dict[str, set[int]] = defaultdict(set)
    gender_to_index: dict[str, set[int]] = defaultdict(set)
    dialect_to_index: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for rows in split_rows.values():
        for row in rows:
            language = row["language"]
            language_to_lid[language].add(int(row["lid"]))

            gender_token = row["_gender_token"]

            if gender_token in {"M", "F"}:
                gender_to_index[gender_token].add(int(row["gender"]))

            dialect_token = row["_dialect_token"]

            if dialect_token:
                dialect_to_index[language][dialect_token].add(
                    int(row["dialect"])
                )

    resolved_language_map: dict[str, int] = {}
    resolved_gender_map: dict[str, int] = {}
    resolved_dialect_map: dict[str, dict[str, int]] = {}

    for language, values in sorted(language_to_lid.items()):
        if len(values) != 1:
            errors.append(
                f"Language {language} has inconsistent LID indices: "
                f"{sorted(values)}"
            )
        else:
            resolved_language_map[language] = next(iter(values))

    for gender, values in sorted(gender_to_index.items()):
        if len(values) != 1:
            errors.append(
                f"Gender {gender} has inconsistent indices: {sorted(values)}"
            )
        else:
            resolved_gender_map[gender] = next(iter(values))

    for language, dialects in sorted(dialect_to_index.items()):
        resolved_dialect_map[language] = {}

        for dialect, values in sorted(dialects.items()):
            if len(values) != 1:
                errors.append(
                    f"{language}/{dialect} has inconsistent indices: "
                    f"{sorted(values)}"
                )
            else:
                resolved_dialect_map[language][dialect] = next(iter(values))

    return (
        {
            "language_to_lid": resolved_language_map,
            "gender_to_index": resolved_gender_map,
            "dialect_to_index": resolved_dialect_map,
        },
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_errors: list[str] = []
    domain_summaries: dict[str, dict[str, Any]] = {}
    domain_rows: dict[str, dict[str, list[dict[str, str]]]] = {}

    for domain in DOMAINS:
        domain_summaries[domain] = {}
        domain_rows[domain] = {}

        for split in SPLITS:
            summary, rows, errors = validate_cache_split(
                domain=domain,
                split=split,
                split_root=args.cache_root / domain / split,
            )

            domain_summaries[domain][split] = summary
            domain_rows[domain][split] = rows
            all_errors.extend(errors)

    cross_domain_comparison: dict[str, Any] = {}

    for split in SPLITS:
        one_summary = domain_summaries["one_bit"][split]
        sixteen_summary = domain_summaries["sixteen_bit"][split]

        one_rows = domain_rows["one_bit"][split]
        sixteen_rows = domain_rows["sixteen_bit"][split]

        compared_columns = (
            "utt_id",
            "start_frame",
            "num_frames",
            "lid",
            "dialect",
            "gender",
            "language",
            "filepath",
            "audio_samples",
        )

        exact_rows_equal = (
            len(one_rows) == len(sixteen_rows)
            and all(
                all(left[column] == right[column] for column in compared_columns)
                for left, right in zip(one_rows, sixteen_rows)
            )
        )

        cross_domain_comparison[split] = {
            "one_bit_index_sha256": one_summary.get("index_sha256"),
            "sixteen_bit_index_sha256": sixteen_summary.get("index_sha256"),
            "byte_identical_indexes": (
                one_summary.get("index_sha256")
                == sixteen_summary.get("index_sha256")
            ),
            "exact_selected_columns_equal": exact_rows_equal,
        }

        if not exact_rows_equal:
            all_errors.append(
                f"{split}: one-bit and sixteen-bit cache indexes differ"
            )

    inferred_maps, map_errors = derive_global_maps(
        domain_rows["one_bit"]
    )
    all_errors.extend(map_errors)

    overlap_summaries: dict[str, Any] = {}

    manifest_names = {
        "train": "respin_unique_sentence_train.csv",
        "dev": "respin_unique_sentence_dev.csv",
        "test": "respin_unique_sentence_test.csv",
    }

    for split in SPLITS:
        overlap, overlap_errors = manifest_cache_overlap(
            split=split,
            manifest_path=args.manifest_root / manifest_names[split],
            cache_rows=domain_rows["one_bit"][split],
            output_dir=args.output_dir,
        )

        overlap_summaries[split] = overlap
        all_errors.extend(overlap_errors)

    speaker_sets = {
        split: {
            row["_speaker_id"]
            for row in rows
            if row["_speaker_id"]
        }
        for split, rows in domain_rows["one_bit"].items()
    }

    speaker_overlap = {
        "train_dev": sorted(
            speaker_sets["train"] & speaker_sets["dev"]
        ),
        "train_test": sorted(
            speaker_sets["train"] & speaker_sets["test"]
        ),
        "dev_test": sorted(
            speaker_sets["dev"] & speaker_sets["test"]
        ),
    }

    for pair_name, speakers in speaker_overlap.items():
        if speakers:
            all_errors.append(
                f"Cache-native speaker leakage in {pair_name}: "
                f"{len(speakers)} speakers"
            )

    all_current_manifests_covered = all(
        overlap_summaries[split].get(
            "current_manifest_is_cache_subset", False
        )
        for split in SPLITS
    )

    any_cache_extras = any(
        overlap_summaries[split].get("cache_only_paths", 0) > 0
        for split in SPLITS
    )

    if all_current_manifests_covered:
        recommendation = (
            "FILTER_EXISTING_CACHE_TO_CURRENT_MANIFESTS"
            if any_cache_extras
            else "CACHE_EXACTLY_MATCHES_CURRENT_MANIFESTS"
        )
    else:
        recommendation = (
            "USE_CACHE_NATIVE_INDEXES_OR_REEXTRACT_MISSING_MANIFEST_ROWS"
        )

    report = {
        "status": "PASS" if not all_errors else "FAIL",
        "cache_root": str(args.cache_root),
        "domain_summaries": domain_summaries,
        "cross_domain_comparison": cross_domain_comparison,
        "inferred_cache_label_maps": inferred_maps,
        "manifest_cache_overlap": overlap_summaries,
        "cache_native_speaker_overlap_counts": {
            key: len(value)
            for key, value in speaker_overlap.items()
        },
        "recommendation": recommendation,
        "errors": all_errors,
    }

    write_json(args.output_dir / "cache_contract_report.json", report)
    write_json(
        args.output_dir / "inferred_cache_label_maps.json",
        inferred_maps,
    )

    summary_csv = args.output_dir / "cache_manifest_overlap_summary.csv"

    with summary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "split",
            "manifest_rows",
            "cache_rows",
            "shared_paths",
            "manifest_only_paths",
            "cache_only_paths",
            "manifest_coverage_fraction",
            "cache_coverage_fraction",
            "current_manifest_is_cache_subset",
        ]

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for split in SPLITS:
            row = overlap_summaries[split]
            writer.writerow({field: row.get(field) for field in fields})

    print("===== CACHE BINARY INTEGRITY =====")

    for domain in DOMAINS:
        for split in SPLITS:
            item = domain_summaries[domain][split]

            print(
                f"{domain:>11} {split:>5}: "
                f"rows={item.get('row_count', 0):,} | "
                f"frames={item.get('total_frames', 0):,} | "
                f"binary_integrity={item.get('binary_integrity')}"
            )

    print()
    print("===== ONE-BIT / SIXTEEN-BIT INDEX EQUALITY =====")

    for split in SPLITS:
        item = cross_domain_comparison[split]

        print(
            f"{split:>5}: "
            f"byte_identical={item['byte_identical_indexes']} | "
            f"row_contract_equal={item['exact_selected_columns_equal']}"
        )

    print()
    print("===== INFERRED CACHE LABEL MAPS =====")
    print(
        "language_to_lid:",
        json.dumps(
            inferred_maps["language_to_lid"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    print(
        "gender_to_index:",
        json.dumps(
            inferred_maps["gender_to_index"],
            ensure_ascii=False,
            sort_keys=True,
        ),
    )

    print("dialect_to_index:")

    for language, mapping in sorted(
        inferred_maps["dialect_to_index"].items()
    ):
        print(
            f"  {language}: "
            + json.dumps(mapping, ensure_ascii=False, sort_keys=True)
        )

    print()
    print("===== CURRENT MANIFEST / CACHE OVERLAP =====")

    for split in SPLITS:
        item = overlap_summaries[split]

        print(
            f"{split:>5}: "
            f"manifest={item.get('manifest_rows', 0):,} | "
            f"cache={item.get('cache_rows', 0):,} | "
            f"shared={item.get('shared_paths', 0):,} | "
            f"manifest_only={item.get('manifest_only_paths', 0):,} | "
            f"cache_only={item.get('cache_only_paths', 0):,} | "
            f"manifest_coverage="
            f"{item.get('manifest_coverage_fraction', 0.0):.6f}"
        )

    print()
    print("===== CACHE-NATIVE SPEAKER OVERLAP =====")

    for pair_name, speakers in speaker_overlap.items():
        print(f"{pair_name}: {len(speakers)}")

    print()
    print("===== DECISION =====")
    print("recommendation:", recommendation)

    if all_errors:
        print()
        print(f"CACHE_CONTRACT_AUDIT_FAIL: {len(all_errors)} issue(s)")

        for index, error in enumerate(all_errors, start=1):
            print(f"{index}. {error}")

        return 2

    print("CACHE_CONTRACT_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

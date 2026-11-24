#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
    ".md",
}


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


def dialect_sort_key(label: str) -> tuple[int, int | str]:
    cleaned = str(label).strip()

    if cleaned.startswith("D") and cleaned[1:].isdigit():
        return 0, int(cleaned[1:])

    return 1, cleaned


def read_manifest(
    split_name: str,
    path: Path,
    required_columns: set[str],
    configured_languages: set[str],
    configured_genders: set[str],
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    errors: list[str] = []

    if not path.is_file():
        return (
            {
                "split": split_name,
                "path": str(path),
                "exists": False,
            },
            [],
            [f"Missing {split_name} manifest: {path}"],
        )

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])

        missing_columns = sorted(required_columns - columns)

        if missing_columns:
            errors.append(
                f"{split_name} manifest is missing columns: "
                + ", ".join(missing_columns)
            )

        for row_number, row in enumerate(reader, start=2):
            cleaned = {
                str(key): "" if value is None else str(value).strip()
                for key, value in row.items()
            }

            cleaned["_row_number"] = str(row_number)
            rows.append(cleaned)

    utterance_ids = [
        row.get("utt_id", "")
        for row in rows
        if row.get("utt_id", "")
    ]

    duplicate_utterance_ids = sorted(
        value
        for value, count in Counter(utterance_ids).items()
        if count > 1
    )

    if duplicate_utterance_ids:
        errors.append(
            f"{split_name} has {len(duplicate_utterance_ids)} "
            "duplicate utterance IDs"
        )

    language_counts = Counter(row.get("language", "") for row in rows)
    gender_counts = Counter(row.get("gender", "") for row in rows)

    invalid_languages = sorted(
        label
        for label in language_counts
        if label not in configured_languages
    )

    invalid_genders = sorted(
        label
        for label in gender_counts
        if label not in configured_genders
    )

    if invalid_languages:
        errors.append(
            f"{split_name} has invalid languages: {invalid_languages}"
        )

    if invalid_genders:
        errors.append(
            f"{split_name} has invalid gender labels: {invalid_genders}"
        )

    dialects_by_language: dict[str, set[str]] = defaultdict(set)
    composite_dialect_mismatches = 0

    for row in rows:
        language = row.get("language", "")
        dialect_local = row.get("dialect_local", "")
        dialect_composite = row.get("dialect", "")

        if language and dialect_local:
            dialects_by_language[language].add(dialect_local)

        expected_composite = (
            f"{language}_{dialect_local}"
            if language and dialect_local
            else ""
        )

        if (
            dialect_composite
            and expected_composite
            and dialect_composite != expected_composite
        ):
            composite_dialect_mismatches += 1

    if composite_dialect_mismatches:
        errors.append(
            f"{split_name} has {composite_dialect_mismatches} "
            "composite dialect-label mismatches"
        )

    duplicate_language_text_pairs = 0

    if "text" in (rows[0] if rows else {}):
        pair_counts = Counter(
            (row.get("language", ""), row.get("text", ""))
            for row in rows
            if row.get("language", "") and row.get("text", "")
        )

        duplicate_language_text_pairs = sum(
            1 for count in pair_counts.values() if count > 1
        )

        # Repeated transcript strings are permitted when the utterance IDs
        # are distinct. They may represent different speakers, genders, or
        # dialect realizations of the same prompt. Duplicate utt_id values
        # remain a fatal validation error.

    durations: list[float] = []
    invalid_duration_rows = 0

    for row in rows:
        raw_duration = row.get("duration", "")

        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            invalid_duration_rows += 1
            continue

        if duration <= 0:
            invalid_duration_rows += 1
            continue

        durations.append(duration)

    if invalid_duration_rows:
        errors.append(
            f"{split_name} has {invalid_duration_rows} invalid durations"
        )

    speaker_ids = {
        row.get("speaker_id", "")
        for row in rows
        if row.get("speaker_id", "")
    }

    summary = {
        "split": split_name,
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "columns": sorted(
            key for key in (rows[0].keys() if rows else []) if not key.startswith("_")
        ),
        "row_count": len(rows),
        "unique_utterance_count": len(set(utterance_ids)),
        "duplicate_utterance_id_count": len(duplicate_utterance_ids),
        "duplicate_language_text_pair_count": duplicate_language_text_pairs,
        "duplicate_language_text_pair_policy": "retain_distinct_utterance_ids",
        "speaker_count": len(speaker_ids),
        "language_counts": dict(sorted(language_counts.items())),
        "gender_counts": dict(sorted(gender_counts.items())),
        "dialects_by_language": {
            language: sorted(labels, key=dialect_sort_key)
            for language, labels in sorted(dialects_by_language.items())
        },
        "duration": {
            "valid_count": len(durations),
            "invalid_count": invalid_duration_rows,
            "total_hours": (
                round(sum(durations) / 3600.0, 6)
                if durations
                else 0.0
            ),
            "minimum_seconds": min(durations) if durations else None,
            "maximum_seconds": max(durations) if durations else None,
            "mean_seconds": (
                sum(durations) / len(durations)
                if durations
                else None
            ),
        },
    }

    return summary, rows, errors


def inventory_cache(domain: str, root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    if not root.is_dir():
        return (
            {
                "domain": domain,
                "root": str(root),
                "exists": False,
            },
            [f"Missing {domain} cache directory: {root}"],
        )

    files = sorted(path for path in root.rglob("*") if path.is_file())

    extension_counts: Counter[str] = Counter()
    total_bytes = 0
    file_records: list[dict[str, Any]] = []
    candidate_metadata: list[dict[str, Any]] = []

    for path in files:
        size = path.stat().st_size
        suffix = path.suffix.lower() or "<no_extension>"
        relative = path.relative_to(root).as_posix()

        extension_counts[suffix] += 1
        total_bytes += size

        record = {
            "relative_path": relative,
            "size_bytes": size,
            "suffix": suffix,
        }

        file_records.append(record)

        if suffix in TEXT_SUFFIXES and size <= 50 * 1024 * 1024:
            candidate_metadata.append(record)

    top_level_entries = []

    for path in sorted(root.iterdir()):
        top_level_entries.append(
            {
                "name": path.name,
                "type": "directory" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )

    return (
        {
            "domain": domain,
            "root": str(root),
            "resolved_root": str(root.resolve()),
            "exists": True,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "total_gib": round(total_bytes / (1024**3), 6),
            "extension_counts": dict(sorted(extension_counts.items())),
            "top_level_entries": top_level_entries,
            "candidate_metadata_files": candidate_metadata,
            "files": file_records,
        },
        errors,
    )


def build_cache_tree(root: Path, maximum_depth: int = 4) -> list[str]:
    lines: list[str] = []

    if not root.is_dir():
        return lines

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)

        if len(relative.parts) > maximum_depth:
            continue

        marker = "d" if path.is_dir() else "f"

        if path.is_file():
            size_text = f" [{path.stat().st_size} bytes]"
        else:
            size_text = ""

        lines.append(f"{marker} {relative.as_posix()}{size_text}")

    return lines


def build_text_previews(
    domain: str,
    root: Path,
    inventory: dict[str, Any],
    maximum_lines: int = 12,
    maximum_characters: int = 5000,
) -> list[str]:
    previews: list[str] = []

    for record in inventory.get("candidate_metadata_files", []):
        path = root / record["relative_path"]

        previews.append("=" * 78)
        previews.append(f"DOMAIN: {domain}")
        previews.append(f"FILE: {record['relative_path']}")
        previews.append(f"SIZE: {record['size_bytes']} bytes")
        previews.append("-" * 78)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            preview_lines = text.splitlines()[:maximum_lines]
            preview = "\n".join(preview_lines)[:maximum_characters]
            previews.append(preview)
        except Exception as exc:
            previews.append(f"PREVIEW_ERROR: {type(exc).__name__}: {exc}")

        previews.append("")

    return previews


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    args = parser.parse_args()

    configuration = yaml.safe_load(
        args.data_config.read_text(encoding="utf-8")
    )

    data_config = configuration["data"]
    cache_config = configuration["cache"]

    required_columns = set(data_config["required_manifest_columns"])
    configured_languages = set(data_config["languages"])
    configured_genders = set(data_config["gender_labels"])

    errors: list[str] = []
    manifest_summaries: dict[str, Any] = {}
    manifest_rows: dict[str, list[dict[str, str]]] = {}

    for split_name in ("train", "dev", "test"):
        path = Path(data_config["manifests"][split_name])

        summary, rows, split_errors = read_manifest(
            split_name=split_name,
            path=path,
            required_columns=required_columns,
            configured_languages=configured_languages,
            configured_genders=configured_genders,
        )

        manifest_summaries[split_name] = summary
        manifest_rows[split_name] = rows
        errors.extend(split_errors)

    speaker_sets = {
        split_name: {
            row.get("speaker_id", "")
            for row in rows
            if row.get("speaker_id", "")
        }
        for split_name, rows in manifest_rows.items()
    }

    speaker_overlap = {
        "train_dev": sorted(
            speaker_sets.get("train", set())
            & speaker_sets.get("dev", set())
        ),
        "train_test": sorted(
            speaker_sets.get("train", set())
            & speaker_sets.get("test", set())
        ),
        "dev_test": sorted(
            speaker_sets.get("dev", set())
            & speaker_sets.get("test", set())
        ),
    }

    for pair_name, overlap in speaker_overlap.items():
        if overlap:
            errors.append(
                f"Speaker leakage in {pair_name}: {len(overlap)} speakers"
            )

    actual_dialect_maps: dict[str, set[str]] = defaultdict(set)

    for rows in manifest_rows.values():
        for row in rows:
            language = row.get("language", "")
            dialect_local = row.get("dialect_local", "")

            if language and dialect_local:
                actual_dialect_maps[language].add(dialect_local)

    actual_dialect_maps_sorted = {
        language: sorted(
            actual_dialect_maps.get(language, set()),
            key=dialect_sort_key,
        )
        for language in data_config["languages"]
    }

    configured_dialect_maps = data_config["dialect_local_labels"]

    for language in data_config["languages"]:
        actual = actual_dialect_maps_sorted[language]
        configured = configured_dialect_maps[language]

        if actual != configured:
            errors.append(
                f"Dialect map mismatch for {language}: "
                f"configured={configured}, actual={actual}"
            )

    language_map = {
        "labels": data_config["languages"],
        "names": data_config["language_names"],
        "label_to_index": data_config["language_to_index"],
        "index_to_label": {
            str(index): label
            for label, index in data_config["language_to_index"].items()
        },
    }

    gender_map = {
        "labels": data_config["gender_labels"],
        "label_to_index": data_config["gender_to_index"],
        "index_to_label": {
            str(index): label
            for label, index in data_config["gender_to_index"].items()
        },
    }

    dialect_maps = {}

    for language in data_config["languages"]:
        labels = actual_dialect_maps_sorted[language]

        dialect_maps[language] = {
            "labels": labels,
            "number_of_classes": len(labels),
            "label_to_index": {
                label: index for index, label in enumerate(labels)
            },
            "index_to_label": {
                str(index): label for index, label in enumerate(labels)
            },
            "composite_labels": [
                f"{language}_{label}" for label in labels
            ],
        }

    task_definition = {
        "tasks": {
            "language": {
                "number_of_heads": 1,
                "number_of_classes": 8,
            },
            "gender": {
                "number_of_heads": 1,
                "number_of_classes": 2,
            },
            "dialect": {
                "number_of_heads": 8,
                "routing": "oracle_ground_truth_language",
                "total_classes_across_heads": sum(
                    item["number_of_classes"]
                    for item in dialect_maps.values()
                ),
                "heads": {
                    language: item["number_of_classes"]
                    for language, item in dialect_maps.items()
                },
            },
        },
        "total_classifier_modules": 10,
    }

    cache_inventories: dict[str, Any] = {}
    cache_trees: list[str] = []
    cache_previews: list[str] = []

    for domain in ("one_bit", "sixteen_bit"):
        root = Path(cache_config["domains"][domain]["root"])

        inventory, cache_errors = inventory_cache(domain, root)
        cache_inventories[domain] = inventory
        errors.extend(cache_errors)

        cache_trees.append("=" * 78)
        cache_trees.append(f"CACHE DOMAIN: {domain}")
        cache_trees.append(f"ROOT: {root}")
        cache_trees.append("=" * 78)
        cache_trees.extend(build_cache_tree(root))
        cache_trees.append("")

        cache_previews.extend(
            build_text_previews(
                domain=domain,
                root=root,
                inventory=inventory,
            )
        )

    one_files = {
        record["relative_path"]
        for record in cache_inventories.get("one_bit", {}).get("files", [])
    }

    sixteen_files = {
        record["relative_path"]
        for record in cache_inventories.get("sixteen_bit", {}).get("files", [])
    }

    cache_relative_path_comparison = {
        "same_relative_file_set": one_files == sixteen_files,
        "one_bit_only": sorted(one_files - sixteen_files),
        "sixteen_bit_only": sorted(sixteen_files - one_files),
        "shared_relative_paths": len(one_files & sixteen_files),
    }

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)

    write_json(args.metadata_dir / "language_map.json", language_map)
    write_json(args.metadata_dir / "gender_map.json", gender_map)
    write_json(args.metadata_dir / "dialect_maps.json", dialect_maps)
    write_json(args.metadata_dir / "task_definition.json", task_definition)

    write_json(
        args.metadata_dir / "manifest_summary.json",
        {
            "manifests": manifest_summaries,
            "speaker_overlap_counts": {
                key: len(value)
                for key, value in speaker_overlap.items()
            },
        },
    )

    metadata_fingerprint_source = json.dumps(
        {
            "language_map": language_map,
            "gender_map": gender_map,
            "dialect_maps": dialect_maps,
            "task_definition": task_definition,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    write_json(
        args.metadata_dir / "labels_fingerprint.json",
        {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(
                metadata_fingerprint_source
            ).hexdigest(),
        },
    )

    write_json(
        args.validation_dir / "input_validation.json",
        {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "manifest_summaries": manifest_summaries,
            "speaker_overlap": speaker_overlap,
            "actual_dialect_maps": actual_dialect_maps_sorted,
        },
    )

    cache_validation_dir = args.validation_dir / "cache"
    cache_validation_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        cache_validation_dir / "cache_inventory.json",
        cache_inventories,
    )

    write_json(
        cache_validation_dir / "cache_relative_path_comparison.json",
        cache_relative_path_comparison,
    )

    (cache_validation_dir / "cache_tree.txt").write_text(
        "\n".join(cache_trees) + "\n",
        encoding="utf-8",
    )

    (cache_validation_dir / "cache_text_previews.txt").write_text(
        "\n".join(cache_previews) + "\n",
        encoding="utf-8",
    )

    print("===== MANIFEST SUMMARY =====")

    for split_name in ("train", "dev", "test"):
        summary = manifest_summaries[split_name]

        if not summary.get("exists", False):
            print(f"{split_name:>5}: MISSING")
            continue

        print(
            f"{split_name:>5}: "
            f"rows={summary['row_count']:,} | "
            f"speakers={summary['speaker_count']:,} | "
            f"hours={summary['duration']['total_hours']:.3f}"
        )

    print()
    print("===== DIALECT HEADS =====")

    for language, item in dialect_maps.items():
        print(
            f"{language}: "
            f"{item['number_of_classes']} classes | "
            f"{', '.join(item['labels'])}"
        )

    print()
    print("===== SPEAKER OVERLAP =====")

    for pair_name, overlap in speaker_overlap.items():
        print(f"{pair_name}: {len(overlap)}")

    print()
    print("===== CACHE INVENTORY =====")

    for domain in ("one_bit", "sixteen_bit"):
        inventory = cache_inventories[domain]

        if not inventory.get("exists", False):
            print(f"{domain}: MISSING")
            continue

        print(
            f"{domain}: "
            f"{inventory['file_count']} files | "
            f"{inventory['total_gib']:.3f} GiB | "
            f"extensions={inventory['extension_counts']}"
        )

        candidates = inventory.get("candidate_metadata_files", [])
        print(f"  candidate metadata/index files: {len(candidates)}")

        for item in candidates[:20]:
            print(
                f"  - {item['relative_path']} "
                f"({item['size_bytes']} bytes)"
            )

    print()
    print("===== CACHE FILE-SET COMPARISON =====")
    print(
        "same relative file set:",
        cache_relative_path_comparison["same_relative_file_set"],
    )
    print(
        "shared relative paths:",
        cache_relative_path_comparison["shared_relative_paths"],
    )
    print(
        "one-bit-only paths:",
        len(cache_relative_path_comparison["one_bit_only"]),
    )
    print(
        "sixteen-bit-only paths:",
        len(cache_relative_path_comparison["sixteen_bit_only"]),
    )

    print()
    print("===== VALIDATION RESULT =====")

    if errors:
        print(f"INPUT_VALIDATION_FAIL: {len(errors)} error(s)")

        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")

        return 2

    print("INPUT_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

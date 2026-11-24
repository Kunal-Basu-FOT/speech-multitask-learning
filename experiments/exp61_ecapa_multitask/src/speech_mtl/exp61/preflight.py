from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from speech_mtl.common.data.cache_index import (
    load_cache_index,
    validate_binary_size,
)
from speech_mtl.common.utils.atomic_io import sha256_file

from .config import Exp61Config, VALID_DOMAINS, VALID_SPLITS


DIALECT_CLASS_COUNTS = {
    "bh": 3,
    "bn": 5,
    "ch": 4,
    "kn": 5,
    "mg": 4,
    "mr": 4,
    "mt": 4,
    "te": 4,
}


def _validate_split_labels(
    records,
    *,
    split: str,
) -> dict[str, Any]:
    languages = {record.language for record in records}
    lids = {record.lid for record in records}
    genders = {record.gender for record in records}

    if languages != set(DIALECT_CLASS_COUNTS):
        raise ValueError(
            f"{split}: language set mismatch: {sorted(languages)}"
        )

    if lids != set(range(8)):
        raise ValueError(
            f"{split}: LID labels must be exactly 0..7"
        )

    if genders != {0, 1}:
        raise ValueError(
            f"{split}: gender labels must be exactly 0 and 1"
        )

    language_to_lids: dict[str, set[int]] = defaultdict(set)
    lid_to_languages: dict[int, set[str]] = defaultdict(set)
    dialects_by_language: dict[str, set[int]] = defaultdict(set)

    for record in records:
        language_to_lids[record.language].add(record.lid)
        lid_to_languages[record.lid].add(record.language)

        class_count = DIALECT_CLASS_COUNTS[record.language]

        if not 0 <= record.dialect < class_count:
            raise ValueError(
                f"{split}: dialect={record.dialect} is invalid "
                f"for language={record.language} "
                f"with {class_count} local classes"
            )

        dialects_by_language[record.language].add(
            record.dialect
        )

    if any(
        len(values) != 1
        for values in language_to_lids.values()
    ):
        raise ValueError(
            f"{split}: a language maps to multiple LID labels"
        )

    if any(
        len(values) != 1
        for values in lid_to_languages.values()
    ):
        raise ValueError(
            f"{split}: a LID label maps to multiple languages"
        )

    if split == "train":
        for language, class_count in (
            DIALECT_CLASS_COUNTS.items()
        ):
            expected = set(range(class_count))
            actual = dialects_by_language[language]

            if actual != expected:
                raise ValueError(
                    f"train: missing dialect classes for {language}; "
                    f"expected={sorted(expected)}, "
                    f"actual={sorted(actual)}"
                )

    return {
        "languages": sorted(languages),
        "lid_values": sorted(lids),
        "gender_values": sorted(genders),
        "language_to_lid": {
            language: next(iter(values))
            for language, values
            in sorted(language_to_lids.items())
        },
        "dialect_values_by_language": {
            language: sorted(values)
            for language, values
            in sorted(dialects_by_language.items())
        },
    }


def run_exp61_preflight(
    config: Exp61Config,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "PASS",
        "cache_root": str(config.cache_root),
        "splits": {},
    }

    for split in VALID_SPLITS:
        domain_reports = {}

        for domain in VALID_DOMAINS:
            split_root = config.cache_root / domain / split
            index_path = split_root / "index.csv"
            features_path = split_root / "features.f16"

            records = load_cache_index(index_path)

            expected_rows = config.expected_split_counts[split]

            if len(records) != expected_rows:
                raise ValueError(
                    f"{domain}/{split}: expected "
                    f"{expected_rows:,} rows, found "
                    f"{len(records):,}"
                )

            total_frames = validate_binary_size(
                features_path=features_path,
                records=records,
                feature_dimension=config.feature_dimension,
                bytes_per_value=2,
            )

            domain_reports[domain] = {
                "rows": len(records),
                "frames": total_frames,
                "minimum_frames": min(
                    record.num_frames for record in records
                ),
                "maximum_frames": max(
                    record.num_frames for record in records
                ),
                "index_sha256": sha256_file(index_path),
                "feature_bytes": features_path.stat().st_size,
            }

            if domain == "one_bit":
                label_report = _validate_split_labels(
                    records,
                    split=split,
                )

        one_hash = domain_reports["one_bit"][
            "index_sha256"
        ]
        sixteen_hash = domain_reports["sixteen_bit"][
            "index_sha256"
        ]

        if one_hash != sixteen_hash:
            raise ValueError(
                f"{split}: one-bit and sixteen-bit indexes differ"
            )

        if (
            domain_reports["one_bit"]["frames"]
            != domain_reports["sixteen_bit"]["frames"]
        ):
            raise ValueError(
                f"{split}: frame totals differ across domains"
            )

        report["splits"][split] = {
            "domains": domain_reports,
            "indexes_identical": True,
            "labels": label_report,
        }

    return report

"""Derive and freeze label mappings from the cache indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

from .utils import natural_key


@dataclass(frozen=True)
class LabelSpace:
    languages: tuple[str, ...]
    language_to_index: dict[str, int]
    gender_values: tuple[str, ...]
    gender_to_index: dict[str, int]
    dialect_values_by_language: dict[str, tuple[str, ...]]
    dialect_to_index_by_language: dict[str, dict[str, int]]

    @property
    def num_lid_classes(self) -> int:
        return len(self.languages)

    @property
    def num_gender_classes(self) -> int:
        return len(self.gender_values)

    def num_dialect_classes(self, language: str) -> int:
        return len(self.dialect_values_by_language[language])

    def to_dict(self) -> dict[str, Any]:
        return {
            "languages": list(self.languages),
            "gender_values": list(self.gender_values),
            "dialect_values_by_language": {
                key: list(value) for key, value in self.dialect_values_by_language.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelSpace":
        languages = tuple(payload["languages"])
        genders = tuple(str(x) for x in payload["gender_values"])
        dialects = {
            str(lang): tuple(str(x) for x in values)
            for lang, values in payload["dialect_values_by_language"].items()
        }
        return cls(
            languages=languages,
            language_to_index={x: i for i, x in enumerate(languages)},
            gender_values=genders,
            gender_to_index={x: i for i, x in enumerate(genders)},
            dialect_values_by_language=dialects,
            dialect_to_index_by_language={
                lang: {x: i for i, x in enumerate(values)} for lang, values in dialects.items()
            },
        )


def derive_label_space(cache_root: Path) -> LabelSpace:
    index_path = cache_root / "one_bit" / "train" / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    languages: set[str] = set()
    genders: set[str] = set()
    dialects: dict[str, set[str]] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"language", "dialect", "gender"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing index columns in {index_path}: {sorted(missing)}")
        for row in reader:
            lang = str(row["language"])
            dialect = str(row["dialect"])
            gender = str(row["gender"])
            languages.add(lang)
            genders.add(gender)
            dialects.setdefault(lang, set()).add(dialect)

    language_values = tuple(sorted(languages, key=natural_key))
    gender_values = tuple(sorted(genders, key=natural_key))
    dialect_values = {
        lang: tuple(sorted(dialects[lang], key=natural_key)) for lang in language_values
    }
    space = LabelSpace.from_dict(
        {
            "languages": language_values,
            "gender_values": gender_values,
            "dialect_values_by_language": dialect_values,
        }
    )
    expected_languages = {"bh", "bn", "ch", "kn", "mg", "mr", "mt", "te"}
    if set(space.languages) != expected_languages:
        raise ValueError(
            f"Expected eight project languages {sorted(expected_languages)}, got {space.languages}"
        )
    expected_counts = {"bh": 3, "bn": 5, "ch": 4, "kn": 5, "mg": 4, "mr": 4, "mt": 4, "te": 4}
    observed = {lang: space.num_dialect_classes(lang) for lang in space.languages}
    if observed != expected_counts:
        raise ValueError(f"Dialect class contract mismatch: expected {expected_counts}, got {observed}")
    if space.num_gender_classes != 2:
        raise ValueError(f"Expected two gender labels, got {space.gender_values}")
    return space


def save_label_space(space: LabelSpace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(space.to_dict(), indent=2, sort_keys=True) + "\n")


def load_label_space(path: Path) -> LabelSpace:
    return LabelSpace.from_dict(json.loads(path.read_text()))

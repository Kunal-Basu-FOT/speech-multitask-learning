"""Frozen condition registry for the reduced EXP 6.2 study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Condition:
    name: str
    active_tasks: tuple[str, ...]
    selection_metrics: tuple[str, ...]


CONDITIONS: Mapping[str, Condition] = {
    "lid_only": Condition(
        name="lid_only",
        active_tasks=("lid",),
        selection_metrics=("lid_macro_f1",),
    ),
    "dialect_only": Condition(
        name="dialect_only",
        active_tasks=("dialect",),
        selection_metrics=("dialect_language_macro_f1",),
    ),
    "lid_dialect": Condition(
        name="lid_dialect",
        active_tasks=("lid", "dialect"),
        selection_metrics=("lid_macro_f1", "dialect_language_macro_f1"),
    ),
}

CONDITION_ORDER = ("lid_only", "dialect_only", "lid_dialect")
DOMAINS = ("one_bit", "sixteen_bit")


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError as exc:
        allowed = ", ".join(CONDITION_ORDER)
        raise ValueError(f"Unknown condition {name!r}; expected one of: {allowed}") from exc

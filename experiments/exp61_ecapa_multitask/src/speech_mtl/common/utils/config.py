from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {source}"
        )

    value = yaml.safe_load(source.read_text(encoding="utf-8"))

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(
            f"Top-level YAML value must be a mapping: {source}"
        )

    return value


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def get_required(
    mapping: Mapping[str, Any],
    dotted_path: str,
) -> Any:
    """Read a required nested configuration value using dotted notation."""

    if not dotted_path:
        raise ValueError("dotted_path cannot be empty")

    current: Any = mapping

    for component in dotted_path.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(
                f"Cannot descend into {component!r} while reading "
                f"{dotted_path!r}"
            )

        if component not in current:
            raise KeyError(
                f"Missing required configuration key: {dotted_path}"
            )

        current = current[component]

    return current

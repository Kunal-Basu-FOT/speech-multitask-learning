from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write text using fsync and atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding=encoding,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination


def atomic_write_json(
    path: Path | str,
    value: Any,
    *,
    indent: int = 2,
) -> Path:
    return atomic_write_text(
        path,
        json.dumps(
            value,
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )


def sha256_file(
    path: Path | str,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"File does not exist: {source}")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()

    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()

"""Write or verify the SHA-256 manifest of released repository files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"


def files() -> list[Path]:
    return sorted(
        (p for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST and ".git" not in p.parts),
        key=lambda p: p.relative_to(ROOT).as_posix(),
    )


def lines() -> list[str]:
    return [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(ROOT).as_posix()}"
        for p in files()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate SHA256SUMS")
    args = parser.parse_args()
    actual = lines()
    if args.write:
        MANIFEST.write_text("\n".join(actual) + "\n", encoding="utf-8")
        print(f"Wrote {len(actual)} hashes")
    else:
        expected = MANIFEST.read_text(encoding="utf-8").splitlines()
        if expected != actual:
            raise SystemExit("Manifest differs from released files")
        print(f"Verified {len(actual)} files")


if __name__ == "__main__":
    main()

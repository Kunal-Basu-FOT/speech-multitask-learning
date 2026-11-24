"""Best-effort importer for completed EXP 6.1 matched test metrics.

The exact EXP 6.1 JSON layout is not assumed. The importer searches recursively,
extracts common metric aliases, scores matched-test candidates, and refuses
ambiguous results. Manual review remains mandatory before report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

from .conditions import DOMAINS
from .utils import atomic_json_dump


ALIASES = {
    "lid_accuracy": ("lid_accuracy", "language_accuracy"),
    "lid_macro_f1": ("lid_macro_f1", "language_macro_f1"),
    "dialect_language_macro_accuracy": (
        "dialect_language_macro_accuracy",
        "language_macro_dialect_accuracy",
    ),
    "dialect_language_macro_f1": (
        "dialect_language_macro_f1",
        "language_macro_dialect_f1",
    ),
}


def _walk(payload: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from _walk(value, path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk(value, f"{prefix}[{index}]")


def _extract(payload: dict[str, Any]) -> dict[str, float] | None:
    flattened = list(_walk(payload))
    result: dict[str, float] = {}
    for target, aliases in ALIASES.items():
        matches: list[float] = []
        for path, value in flattened:
            leaf = path.split(".")[-1]
            if leaf not in aliases:
                continue
            if isinstance(value, (int, float)):
                matches.append(float(value))
        if not matches:
            return None
        unique = sorted(set(matches))
        if len(unique) != 1:
            return None
        result[target] = unique[0]
    return result


def _candidate_score(path: Path, domain: str) -> int:
    text = str(path).lower()
    score = 0
    if domain in text:
        score += 5
    if "test" in text:
        score += 4
    if "matched" in text:
        score += 4
    if "cross" in text or "one_to_sixteen" in text or "sixteen_to_one" in text:
        score -= 8
    if path.name.lower() in {"metrics.json", "test_metrics.json", "matched_results.json"}:
        score += 2
    return score


def discover_exp61_summary(results_root: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    summary: dict[str, dict[str, float]] = {}
    provenance: dict[str, str] = {}
    json_paths = sorted(results_root.rglob("*.json"))
    for domain in DOMAINS:
        candidates: list[tuple[int, Path, dict[str, float]]] = []
        for path in json_paths:
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            metrics = _extract(payload)
            if metrics is None:
                continue
            score = _candidate_score(path, domain)
            if score > 0:
                candidates.append((score, path, metrics))
        if not candidates:
            raise RuntimeError(f"No usable EXP 6.1 matched metric candidate found for {domain}")
        candidates.sort(key=lambda item: (-item[0], str(item[1])))
        best_score = candidates[0][0]
        best = [item for item in candidates if item[0] == best_score]
        if len(best) != 1:
            paths = "\n".join(str(item[1]) for item in best)
            raise RuntimeError(
                f"Ambiguous EXP 6.1 candidates for {domain} at score {best_score}:\n{paths}\n"
                "Use the manual summary template instead."
            )
        _, path, metrics = best[0]
        summary[domain] = metrics
        provenance[domain] = str(path)
    return summary, provenance


def import_to_file(results_root: Path, output_path: Path) -> Path:
    summary, provenance = discover_exp61_summary(results_root)
    payload = dict(summary)
    payload["_provenance"] = provenance
    payload["_review_required"] = (
        "Verify both source files are matched test evaluations from the completed EXP 6.1 best checkpoints."
    )
    atomic_json_dump(payload, output_path)
    return output_path

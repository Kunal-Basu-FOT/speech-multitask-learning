from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .cross_precision import cross_precision_frame
from .io import ensure_parent


def save_experimental_configuration_table(
    configuration: dict[str, Any],
    *,
    output_path: Path | str,
) -> str:
    rows = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(name, nested)
        elif isinstance(value, (list, tuple)):
            rows.append(
                {
                    "parameter": prefix,
                    "value": ", ".join(str(item) for item in value),
                }
            )
        else:
            rows.append(
                {
                    "parameter": prefix,
                    "value": value,
                }
            )

    visit("", configuration)

    destination = ensure_parent(output_path)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return str(destination)


def _matched_row(
    domain: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    summary = evaluation["summary"]

    return {
        "domain": domain,
        "language_accuracy": summary["language"]["accuracy"],
        "language_macro_f1": summary["language"]["macro_f1"],
        "dialect_language_macro_accuracy": (
            summary["dialect"]["language_macro_accuracy"]
        ),
        "dialect_language_macro_f1": (
            summary["dialect"]["language_macro_f1"]
        ),
        "gender_accuracy": summary["gender"]["accuracy"],
        "gender_macro_f1": summary["gender"]["macro_f1"],
        "three_task_mean_macro_f1": (
            summary["selection_metric"]["value"]
        ),
        "examples": evaluation["examples"],
    }


def save_matched_results_table(
    *,
    one_bit_evaluation: dict[str, Any],
    sixteen_bit_evaluation: dict[str, Any],
    output_path: Path | str,
) -> str:
    frame = pd.DataFrame(
        [
            _matched_row("1-bit", one_bit_evaluation),
            _matched_row("16-bit", sixteen_bit_evaluation),
        ]
    )

    delta = {
        "domain": "16-bit minus 1-bit",
        "examples": "",
    }

    for column in frame.columns:
        if column in {"domain", "examples"}:
            continue
        delta[column] = (
            float(frame.loc[1, column])
            - float(frame.loc[0, column])
        )

    frame = pd.concat(
        [frame, pd.DataFrame([delta])],
        ignore_index=True,
    )

    destination = ensure_parent(output_path)
    frame.to_csv(destination, index=False)
    return str(destination)


def save_per_language_dialect_table(
    *,
    one_bit_evaluation: dict[str, Any],
    sixteen_bit_evaluation: dict[str, Any],
    output_path: Path | str,
) -> str:
    one = one_bit_evaluation["summary"]["dialect"]["per_language"]
    sixteen = sixteen_bit_evaluation["summary"]["dialect"]["per_language"]

    if set(one) != set(sixteen):
        raise ValueError(
            "One-bit and sixteen-bit dialect language sets differ"
        )

    rows = []

    for language in sorted(one):
        rows.append(
            {
                "language": language,
                "one_bit_examples": one[language]["examples"],
                "one_bit_accuracy": one[language]["accuracy"],
                "one_bit_macro_f1": one[language]["macro_f1"],
                "sixteen_bit_examples": sixteen[language]["examples"],
                "sixteen_bit_accuracy": sixteen[language]["accuracy"],
                "sixteen_bit_macro_f1": sixteen[language]["macro_f1"],
                "accuracy_delta_16_minus_1": (
                    sixteen[language]["accuracy"]
                    - one[language]["accuracy"]
                ),
                "macro_f1_delta_16_minus_1": (
                    sixteen[language]["macro_f1"]
                    - one[language]["macro_f1"]
                ),
            }
        )

    destination = ensure_parent(output_path)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return str(destination)


def save_cross_precision_table(
    evaluations: dict[str, dict[str, Any]],
    *,
    output_path: Path | str,
) -> str:
    destination = ensure_parent(output_path)
    cross_precision_frame(evaluations).to_csv(
        destination,
        index=False,
    )
    return str(destination)


def save_runtime_table(
    rows: list[dict[str, Any]],
    *,
    output_path: Path | str,
) -> str:
    required = {
        "run",
        "examples",
        "elapsed_seconds",
        "examples_per_second",
    }

    for index, row in enumerate(rows):
        missing = sorted(required - set(row))

        if missing:
            raise ValueError(
                f"Runtime row {index} is missing fields: {missing}"
            )

    destination = ensure_parent(output_path)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return str(destination)

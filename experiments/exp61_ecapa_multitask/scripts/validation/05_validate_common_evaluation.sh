#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/common_evaluation_validation.log"
OUT="$ROOT/validation/evaluation/common_evaluation_validation.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — COMMON EVALUATION VALIDATION"
echo "============================================================"
echo "Python: $PYTHON"
echo "Mode:   CPU validation; no GPU reservation"
echo

{
    echo "===== ENVIRONMENT ====="

    "$PYTHON" - <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA visible in this validation:", torch.cuda.is_available())
PY

    echo
    echo "===== UNIT TESTS ====="

    "$PYTHON" -m unittest discover \
        -s "$ROOT/tests/unit" \
        -p "test_common_evaluation.py" \
        -v

    echo
    echo "===== INTEGRATED EVALUATION CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.common.evaluation.validation import (
    validate_evaluation_contract,
)

report = validate_evaluation_contract()

output = Path("$OUT")
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("Examples:", report["examples"])
print(
    "Selection metric:",
    report["selection_metric"],
)
print(
    "Language accuracy:",
    report["language_accuracy"],
)
print(
    "Gender accuracy:",
    report["gender_accuracy"],
)
print(
    "Dialect language-macro accuracy:",
    report["dialect_language_macro_accuracy"],
)
print(
    "Dialect language-macro F1:",
    report["dialect_language_macro_f1"],
)
print(
    "Confusion matrices:",
    report["confusion_matrix_contract"],
)
print(
    "Prediction CSV:",
    report["prediction_csv_contract"],
)
print("EVALUATION_CONTRACT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " COMMON_EVALUATION_VALIDATION_PASS"
echo "============================================================"

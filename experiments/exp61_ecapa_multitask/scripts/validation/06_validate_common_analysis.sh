#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/common_analysis_validation.log"
OUT="$ROOT/validation/analysis/common_analysis_validation.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — COMMON ANALYSIS VALIDATION"
echo "============================================================"
echo "Python: $PYTHON"
echo "Mode:   CPU validation; no GPU reservation"
echo

{
    echo "===== ENVIRONMENT ====="

    "$PYTHON" - <<'PY'
import sys
import matplotlib
import numpy
import pandas

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Matplotlib backend:", matplotlib.get_backend())
PY

    echo
    echo "===== UNIT TESTS ====="

    "$PYTHON" -m unittest discover \
        -s "$ROOT/tests/unit" \
        -p "test_common_analysis.py" \
        -v

    echo
    echo "===== INTEGRATED ANALYSIS CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.common.analysis.validation import (
    validate_analysis_contract,
)

report = validate_analysis_contract()

output = Path("$OUT")
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("Generated artifacts:", report["generated_artifact_count"])
print("Synthetic figures:", report["figures"])
print("Synthetic tables:", report["tables"])
print("Curve contract:", report["curve_contract"])
print("Confusion contract:", report["confusion_contract"])
print(
    "Cross-precision contract:",
    report["cross_precision_contract"],
)
print("ANALYSIS_CONTRACT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " COMMON_ANALYSIS_VALIDATION_PASS"
echo "============================================================"

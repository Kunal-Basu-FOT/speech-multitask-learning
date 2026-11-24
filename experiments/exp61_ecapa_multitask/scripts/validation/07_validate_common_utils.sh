#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/common_utils_validation.log"
OUT="$ROOT/validation/utils/common_utils_validation.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — COMMON UTILS VALIDATION"
echo "============================================================"
echo "Python: $PYTHON"
echo "Mode:   CPU validation plus read-only nvidia-smi query"
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
        -p "test_common_utils.py" \
        -v

    echo
    echo "===== INTEGRATED UTILS CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.common.utils.validation import (
    validate_utils_contract,
)

report = validate_utils_contract()

output = Path("$OUT")
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("Configuration:", report["configuration_contract"])
print("Atomic I/O:", report["atomic_io_contract"])
print("SHA-256:", report["sha256_contract"])
print("Logging:", report["logging_contract"])
print("Reproducibility:", report["reproducibility_contract"])
print("GPU parser:", report["gpu_parser_contract"])
print(
    "Live physical GPU1 assessment:",
    report["live_physical_gpu1_assessment"],
)
print("UTILS_CONTRACT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " COMMON_UTILS_VALIDATION_PASS"
echo "============================================================"

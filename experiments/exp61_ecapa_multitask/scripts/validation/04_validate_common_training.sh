#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/common_training_validation.log"
OUT="$ROOT/validation/training/common_training_validation.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — COMMON TRAINING VALIDATION"
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
        -p "test_common_training.py" \
        -v

    echo
    echo "===== INTEGRATED TRAINING CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.common.training.validation import (
    validate_training_contract,
)

report = validate_training_contract()

output = Path("$OUT")
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("Total parameters:", f"{report['parameter_total']:,}")
print("Phase boundaries:", report["phase_boundaries"])
print("Initial learning rates:", report["initial_learning_rates"])
print("Phase-3 learning rates:", report["phase3_learning_rates"])
print("Raw task losses:", report["losses"])
print("Weighted total loss:", report["weighted_total_loss"])
print("Gradient norm:", report["gradient_norm"])
print("Checkpoint roundtrip:", report["checkpoint_roundtrip"])
print("Phase-3 freeze:", report["phase3_freeze"])
print("TRAINING_CONTRACT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " COMMON_TRAINING_VALIDATION_PASS"
echo "============================================================"

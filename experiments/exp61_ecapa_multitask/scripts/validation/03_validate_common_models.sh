#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/common_models_validation.log"
OUT="$ROOT/validation/model/common_models_validation.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — COMMON MODELS VALIDATION"
echo "============================================================"
echo "Python: $PYTHON"
echo "Mode:   CPU validation; no GPU reservation"
echo

{
    echo "===== ENVIRONMENT ====="

    "$PYTHON" - <<'PY'
import sys
import speechbrain
import torch

print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("SpeechBrain:", speechbrain.__version__)
print("CUDA visible in this validation:", torch.cuda.is_available())
PY

    echo
    echo "===== UNIT TESTS ====="

    "$PYTHON" -m unittest discover \
        -s "$ROOT/tests/unit" \
        -p "test_common_models.py" \
        -v

    echo
    echo "===== FULL MODEL CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.common.models.validation import validate_model_forward

report = validate_model_forward()

output = Path("$OUT")
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

parameters = report["parameter_report"]

print("Input normalization:", f"{parameters['input_normalization']:,}")
print("SpeechBrain ECAPA:", f"{parameters['ecapa']:,}")
print("Encoder total:", f"{parameters['encoder_total']:,}")
print("Language head:", f"{parameters['language_head']:,}")
print("Gender head:", f"{parameters['gender_head']:,}")
print("Eight dialect heads:", f"{parameters['dialect_heads']:,}")
print("All classifier heads:", f"{parameters['all_classifier_heads']:,}")
print("Task weighting:", f"{parameters['task_weighting']:,}")
print("Total model:", f"{parameters['total']:,}")
print("Trainable model:", f"{parameters['trainable']:,}")
print(
    "Oracle-routed examples:",
    report["oracle_routing_covered_examples"],
)
print(
    "Phase-3 encoder freeze:",
    report["phase3_encoder_freeze_check"],
)
print("MODEL_FORWARD_CONTRACT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " COMMON_MODELS_VALIDATION_PASS"
echo "============================================================"

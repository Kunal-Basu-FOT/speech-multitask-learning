#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CONFIG="$ROOT/configs/exp61/experiment.yaml"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/integration_validation.log"
OUT="$ROOT/validation/exp61/integration_validation.json"
PREFLIGHT="$ROOT/validation/exp61/real_cache_preflight.json"

mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"

echo "============================================================"
echo " EXP 6.1 — INTEGRATION VALIDATION"
echo "============================================================"
echo "Python: $PYTHON"
echo "Mode:   CPU synthetic integration + real-cache read-only scan"
echo

{
    echo "===== UNIT TESTS ====="

    "$PYTHON" -m unittest discover \
        -s "$ROOT/tests/unit" \
        -p "test_exp61_integration.py" \
        -v

    echo
    echo "===== SYNTHETIC END-TO-END CONTRACT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.exp61.validation import (
    validate_exp61_integration_contract,
)

report = validate_exp61_integration_contract()

Path("$OUT").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, indent=2))
print("EXP61_SYNTHETIC_INTEGRATION_PASS")
PY

    echo
    echo "===== REAL CACHE PREFLIGHT ====="

    "$PYTHON" - <<PY
import json
from pathlib import Path

from speech_mtl.exp61.config import load_exp61_config
from speech_mtl.exp61.preflight import run_exp61_preflight

config = load_exp61_config("$CONFIG")
report = run_exp61_preflight(config)

Path("$PREFLIGHT").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

for split, item in report["splits"].items():
    one = item["domains"]["one_bit"]
    sixteen = item["domains"]["sixteen_bit"]
    print(
        f"{split:>5}: rows={one['rows']:,} | "
        f"frames={one['frames']:,} | "
        f"range={one['minimum_frames']}-{one['maximum_frames']} | "
        f"indexes_identical={item['indexes_identical']}"
    )
    assert one["rows"] == sixteen["rows"]
    assert one["frames"] == sixteen["frames"]

print("EXP61_REAL_CACHE_PREFLIGHT_PASS")
PY

} 2>&1 | tee "$LOG"

echo
echo "============================================================"
echo " EXP61_INTEGRATION_VALIDATION_PASS"
echo "============================================================"

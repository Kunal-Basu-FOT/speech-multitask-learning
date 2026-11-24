#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/input_inventory.log"

mkdir -p "$(dirname "$LOG")"

echo "============================================================"
echo " EXP 6.1 — BUILD METADATA AND INSPECT INPUTS"
echo "============================================================"
echo "Started: $(date --iso-8601=seconds)"
echo "Host:    $(hostname)"
echo "Python:  $PYTHON"
echo "Log:     $LOG"
echo "============================================================"

set +e

"$PYTHON" -m speech_mtl.exp61.build_metadata \
    --data-config "$ROOT/configs/common/data.yaml" \
    --metadata-dir "$ROOT/metadata" \
    --validation-dir "$ROOT/validation" \
    2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

set -e

echo
echo "============================================================"

if [[ "$STATUS" -eq 0 ]]; then
    echo " METADATA_AND_INPUT_INVENTORY_PASS"
else
    echo " METADATA_AND_INPUT_INVENTORY_FAIL"
    echo " Exit status: $STATUS"
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"

exit "$STATUS"

#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/cache_contract_audit.log"
OUT="$ROOT/validation/cache/contract"

mkdir -p "$(dirname "$LOG")" "$OUT"

echo "============================================================"
echo " EXP 6.1 — CACHE CONTRACT AUDIT"
echo "============================================================"
echo "Started: $(date --iso-8601=seconds)"
echo "Host:    $(hostname)"
echo "Log:     $LOG"
echo "============================================================"

set +e

"$PYTHON" -m speech_mtl.exp61.validate_cache_contract \
    --cache-root /data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16 \
    --manifest-root /data/Kunal/manifests \
    --output-dir "$OUT" \
    2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}
set -e

echo
echo "============================================================"

if [[ "$STATUS" -eq 0 ]]; then
    echo " CACHE_CONTRACT_VALIDATION_PASS"
else
    echo " CACHE_CONTRACT_VALIDATION_REQUIRES_DECISION"
    echo " Exit status: $STATUS"
fi

echo "Finished: $(date --iso-8601=seconds)"
echo "============================================================"

exit "$STATUS"

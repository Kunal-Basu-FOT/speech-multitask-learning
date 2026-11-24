#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CACHE=/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo " EXP 6.1 — COMMON DATA VALIDATION"
echo "============================================================"

cd "$ROOT"

echo
echo "===== UNIT TESTS ====="
"$PYTHON" -m unittest discover \
    -s "$ROOT/tests/unit" \
    -p "test_common_data.py" \
    -v

echo
echo "===== REAL CACHE CHECK ====="
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from speech_mtl.common.data.validation import validate_real_cache

report = validate_real_cache(
    "/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16"
)

output = Path(
    "/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK/"
    "validation/cache/common_data_validation.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

for domain, splits in report["domains"].items():
    for split, item in splits.items():
        print(
            f"{domain:>11} {split:>5}: "
            f"rows={item['rows']:,} | "
            f"frames={item['total_frames']:,} | "
            f"range={item['minimum_frames']}-{item['maximum_frames']}"
        )

print("REAL_CACHE_VALIDATION_PASS")
PY

echo
echo "============================================================"
echo " COMMON_DATA_VALIDATION_PASS"
echo "============================================================"

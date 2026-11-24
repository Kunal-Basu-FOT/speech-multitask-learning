#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/Kunal/env_speech_mtl/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
"$PYTHON_BIN" -m speech_mtl.exp62.cli --config configs/exp62/reduced_experiment.yaml import-exp61 "$@"
echo "Open reports/exp62_reduced/exp61_matched_summary.json and verify _provenance before report generation."

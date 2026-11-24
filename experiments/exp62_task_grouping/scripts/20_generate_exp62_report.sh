#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/Kunal/env_speech_mtl/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
"$PYTHON_BIN" -m speech_mtl.exp62.cli --config configs/exp62/reduced_experiment.yaml report "$@"

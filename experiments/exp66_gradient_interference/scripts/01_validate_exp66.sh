#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
"$PY" -m compileall -q src
"$PY" -m speech_mtl.exp66.cli --config configs/exp66/experiment.yaml validate

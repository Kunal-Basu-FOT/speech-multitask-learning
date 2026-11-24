#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
cd "$ROOT"; export PYTHONPATH="$ROOT/src"
/data/Kunal/env_speech_mtl/bin/python -m speech_mtl.exp66.cli --config configs/exp66/experiment.yaml domain-report --domain one_bit

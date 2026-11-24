#!/usr/bin/env bash
set -Eeuo pipefail
cd /data/Kunal/FORMAL_EXP6_7_Hierarchical_LID_Dialect_IndicWav2Vec_FinalLayer
export PYTHONPATH="$PWD/src"
/data/Kunal/env_speech_mtl/bin/python -m speech_mtl.exp67.cli \
  --config configs/exp67/experiment.yaml report

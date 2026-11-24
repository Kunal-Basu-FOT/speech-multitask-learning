#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" -u -m exp69.cli --config "$ROOT/configs/exp69.json" verify

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CONFIG="$ROOT/configs/exp69.json"

[[ "$(whoami)" == "wtc9" ]] || { echo "ERROR: expected user wtc9"; exit 1; }
[[ "$(hostname)" == "wtc9-server" ]] || { echo "ERROR: expected host wtc9-server"; exit 1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON"; exit 1; }
[[ -d "$EXP61/src" ]] || { echo "ERROR: missing EXP 6.1 source: $EXP61/src"; exit 1; }

# Installation validation is deliberately CPU-only so it can run while both
# WTC9 GPUs are occupied without creating CUDA contexts or touching per-process
# CUDA RNG state.
export CUDA_VISIBLE_DEVICES=""

find "$ROOT/scripts" -type f -name '*.sh' -exec bash -n {} \;
bash -n "$ROOT/RUN_ALL_GPU1.sh"

PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" -m compileall -q "$ROOT/src" "$ROOT/tests"

PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" - <<'PY'
import numpy, pandas, torch, torchaudio, matplotlib, speechbrain
import exp69
print("EXP69 package:", exp69.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Torch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Matplotlib:", matplotlib.__version__)
print("SpeechBrain:", speechbrain.__version__)
PY

PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" -m unittest discover -s "$ROOT/tests" -v
PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" -u -m exp69.cli --config "$CONFIG" preflight-cpu

mkdir -p "$ROOT/runtime"
PYTHONPATH="$ROOT/src:$EXP61/src" "$PYTHON" - <<'PY'
from pathlib import Path
from exp69.utils import atomic_json, utc_now
root = Path('/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA')
atomic_json(root / 'runtime/PACKAGE_INSTALL_VALIDATED.json', {
    'status': 'PACKAGE_INSTALL_VALIDATED',
    'created_utc': utc_now(),
    'version': '1.0.0',
})
print(root / 'runtime/PACKAGE_INSTALL_VALIDATED.json')
PY

echo "EXP69_PACKAGE_INSTALL_PASS"

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/Kunal/env_speech_mtl/bin/python}"

echo "===== HOST ====="
whoami
hostname
date --iso-8601=seconds

echo "===== PYTHON ====="
"$PYTHON_BIN" - <<'PY'
import sys
import numpy
import torch
import speechbrain
import yaml
import matplotlib
print("Python:", sys.version)
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("SpeechBrain:", speechbrain.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

test -d /data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16
test -d /data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK

echo "EXP62_ENVIRONMENT_PASS"

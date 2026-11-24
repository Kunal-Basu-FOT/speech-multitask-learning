#!/usr/bin/env bash
set -Eeuo pipefail

PY=/data/Kunal/env_speech_mtl/bin/python
CACHE=/data/Kunal/shared_cache/exp61_indicwav2vec_final_layer_fp16

[[ "$(hostname)" == "wtc7-server" ]] || { echo "ERROR: run on wtc7-server"; exit 1; }
[[ "$(whoami)" == "wtc7" ]] || { echo "ERROR: run as wtc7"; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: missing $PY"; exit 1; }
[[ -d "$CACHE" ]] || { echo "ERROR: missing $CACHE"; exit 1; }

"$PY" - <<'PY'
import sys
import numpy
import pandas
import torch
import torchaudio
import speechbrain
import yaml
import matplotlib

print("Python:", sys.version)
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("SpeechBrain:", speechbrain.__version__)
print("PyYAML:", yaml.__version__)
print("Matplotlib:", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Physical CUDA devices visible:", torch.cuda.device_count())

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 2, "Expected both WTC7 RTX 3090 GPUs before lane isolation"
assert torch.__version__.startswith("2.6."), torch.__version__
for index in range(torch.cuda.device_count()):
    print("GPU", index, torch.cuda.get_device_name(index))
print("EXP67_WTC7_ENVIRONMENT_PASS")
PY

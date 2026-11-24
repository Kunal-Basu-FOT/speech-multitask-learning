#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python
[[ "$(hostname)" == "wtc10-server" ]] || { echo "ERROR: expected wtc10-server"; exit 1; }
[[ "$(whoami)" == "wtc10" ]] || { echo "ERROR: expected user wtc10"; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: missing $PY"; exit 1; }
[[ -d "$ROOT" ]] || { echo "ERROR: missing $ROOT"; exit 1; }
"$PY" - <<'PY'
import sys, numpy, pandas, torch, torchaudio, speechbrain, yaml, matplotlib
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("PyTorch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("SpeechBrain:", speechbrain.__version__)
print("PyYAML:", yaml.__version__)
print("Matplotlib:", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
assert sys.version_info[:2] == (3, 10)
assert torch.__version__ == "2.6.0+cu124"
assert torchaudio.__version__ == "2.6.0+cu124"
assert speechbrain.__version__ == "1.0.3"
assert numpy.__version__ == "2.2.6"
assert pandas.__version__ == "2.3.3"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
assert all(torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3090" for i in range(2))
print("EXP66_WTC10_ENVIRONMENT_PASS")
PY

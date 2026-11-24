#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ' | tr ',' ' ')
(( USED <= 1024 && UTIL <= 10 )) || { echo "ERROR: physical GPU0 busy: memory=${USED}MiB util=${UTIL}%"; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT/src" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
"$PY" - <<'PY'
import torch
from speech_mtl.exp66.validate import run
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"
run("configs/exp66/experiment.yaml", device=torch.device("cuda:0"))
torch.cuda.synchronize()
print("EXP66_GPU0_SMOKE_PASS")
PY

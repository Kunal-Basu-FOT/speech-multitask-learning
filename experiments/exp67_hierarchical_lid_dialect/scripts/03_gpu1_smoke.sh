#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_7_Hierarchical_LID_Dialect_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python

read -r USED UTIL < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i 1 | tr -d ' ' | tr ',' ' '
)
(( USED <= 1024 && UTIL <= 10 )) || {
  echo "ERROR: physical GPU1 is busy: memory=${USED}MiB util=${UTIL}%"
  exit 1
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

"$PY" - <<'PY'
import torch
from speech_mtl.exp67.validate import run

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"
run("configs/exp67/experiment.yaml", device=torch.device("cuda:0"))
torch.cuda.synchronize()
print("EXP67_GPU1_SMOKE_PASS")
PY

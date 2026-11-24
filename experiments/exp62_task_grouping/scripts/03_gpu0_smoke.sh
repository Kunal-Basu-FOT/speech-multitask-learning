#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/Kunal/env_speech_mtl/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
"$PYTHON_BIN" - <<'PY'
import torch
from speech_mtl.exp62.config import load_config
from speech_mtl.exp62.validate import run_synthetic_validation
cfg=load_config('configs/exp62/reduced_experiment.yaml')
assert torch.cuda.is_available() and torch.cuda.device_count()==1
run_synthetic_validation(cfg, torch.device('cuda:0'))
print('EXP62_GPU0_SYNTHETIC_SMOKE_PASS')
PY

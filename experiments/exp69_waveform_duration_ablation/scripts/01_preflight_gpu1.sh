#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CONFIG="$ROOT/configs/exp69.json"
GPU_PROCESSES="$(nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "ERROR: physical GPU 1 is occupied:"
  echo "$GPU_PROCESSES"
  exit 1
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$ROOT/src:$EXP61/src" \
"$PYTHON" -u -m exp69.cli --config "$CONFIG" preflight-gpu

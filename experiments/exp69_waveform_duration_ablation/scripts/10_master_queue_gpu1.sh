#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CONFIG="$ROOT/configs/exp69.json"
LOG="$ROOT/runtime/logs/exp69_gpu1.screen.log"
LOCK=/tmp/speech_mtl_exp69_gpu1.lock

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "ERROR: another EXP 6.9 queue owns $LOCK"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/src:$EXP61/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

{
  echo "============================================================"
  echo " EXP 6.9 — WAVEFORM/LOG-MEL DURATION ABLATION — GPU 1"
  echo "============================================================"
  date --iso-8601=seconds
  whoami
  hostname
  "$PYTHON" --version
  nvidia-smi -i 1

  "$PYTHON" -u -m exp69.cli --config "$CONFIG" preflight-gpu
  "$PYTHON" -u -m exp69.cli --config "$CONFIG" build-cache
  "$PYTHON" -u -m exp69.cli --config "$CONFIG" run-queue
  "$PYTHON" -u -m exp69.cli --config "$CONFIG" report
  "$PYTHON" -u -m exp69.cli --config "$CONFIG" verify

  echo
  echo "EXP69_GPU1_MASTER_QUEUE_COMPLETE"
  date --iso-8601=seconds
} 2>&1 | tee -a "$LOG"

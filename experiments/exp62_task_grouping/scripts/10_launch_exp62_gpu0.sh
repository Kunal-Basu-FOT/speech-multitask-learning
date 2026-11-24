#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/Kunal/env_speech_mtl/bin/python}"
SCREEN_NAME="exp62_gpu0"
LOCK_FILE="/tmp/speech_mtl_exp62_gpu0.lock"
LOG_DIR="$ROOT/runtime/logs/exp62_reduced"
LOG_FILE="$LOG_DIR/master_queue.screen.log"
mkdir -p "$LOG_DIR"

test -f "$ROOT/validation/EXP62_VALIDATION_PASS" || { echo "Run scripts/01_validate_exp62.sh first." >&2; exit 1; }

if screen -list 2>/dev/null | grep -q "\.${SCREEN_NAME}[[:space:]]"; then
  echo "Screen $SCREEN_NAME already exists." >&2
  exit 1
fi

GPU_LINE="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '$1 ~ /^0[[:space:]]*$/ {gsub(/ /, "", $0); print $0}')"
IFS=',' read -r GPU_INDEX GPU_MEM GPU_UTIL <<<"$GPU_LINE"
[[ "$GPU_INDEX" == "0" ]] || { echo "Could not inspect physical GPU 0." >&2; exit 1; }
(( GPU_MEM <= 1024 )) || { echo "GPU 0 memory use is ${GPU_MEM} MiB; refusing launch." >&2; exit 1; }
(( GPU_UTIL <= 10 )) || { echo "GPU 0 utilization is ${GPU_UTIL}%; refusing launch." >&2; exit 1; }
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -q .; then
  UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F',' '$1 ~ /^0[[:space:]]*$/ {gsub(/ /,"",$2); print $2}')"
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$UUID"; then
    echo "GPU 0 has an active compute process; refusing launch." >&2
    exit 1
  fi
fi

COMMAND="cd '$ROOT' && exec 9>'$LOCK_FILE' && flock -n 9 || exit 99; export PYTHONPATH='$ROOT/src'; export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=0; export PYTHONUNBUFFERED=1; export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'; export OMP_NUM_THREADS=8; '$PYTHON_BIN' -m speech_mtl.exp62.cli --config configs/exp62/reduced_experiment.yaml run-queue 2>&1 | tee -a '$LOG_FILE'"
screen -dmS "$SCREEN_NAME" bash -lc "$COMMAND"
sleep 2
screen -list | grep "$SCREEN_NAME"
echo "EXP 6.2 reduced queue launched."
echo "Screen: $SCREEN_NAME"
echo "Log: $LOG_FILE"

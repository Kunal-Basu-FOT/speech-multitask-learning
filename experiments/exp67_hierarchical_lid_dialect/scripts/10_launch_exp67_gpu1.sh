#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_7_Hierarchical_LID_Dialect_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python
SCREEN=exp67_gpu1
LOG="$ROOT/runtime/logs/exp67/master_queue.screen.log"
LOCK=/tmp/speech_mtl_exp67_gpu1.lock

[[ "$(hostname)" == "wtc7-server" ]] || { echo "ERROR: wrong host"; exit 1; }
[[ "$(whoami)" == "wtc7" ]] || { echo "ERROR: wrong user"; exit 1; }
[[ -d "$ROOT" ]] || { echo "ERROR: missing repository $ROOT"; exit 1; }

if screen -list 2>/dev/null | grep -q "[.]$SCREEN"; then
  echo "ERROR: screen already exists: $SCREEN"
  exit 1
fi

read -r USED UTIL < <(
  nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i 1 | tr -d ' ' | tr ',' ' '
)
(( USED <= 1024 && UTIL <= 10 )) || {
  echo "ERROR: GPU1 busy: memory=${USED}MiB util=${UTIL}%"
  exit 1
}

mkdir -p "$(dirname "$LOG")" "$ROOT/runtime/runs/exp67"

CMD="cd '$ROOT' && flock -n '$LOCK' bash -lc 'export PYTHONPATH=\"$ROOT/src\" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 OMP_NUM_THREADS=8; '$PY' -m speech_mtl.exp67.cli --config configs/exp67/experiment.yaml run-queue 2>&1 | tee -a '$LOG''"

screen -dmS "$SCREEN" bash -lc "$CMD"
sleep 2
if ! screen -list 2>/dev/null | grep -q "[.]$SCREEN"; then
  echo "ERROR: screen terminated immediately. Inspect: $LOG"
  tail -n 80 "$LOG" 2>/dev/null || true
  exit 1
fi

echo "EXP 6.7 launched on WTC7 physical GPU1."
echo "Screen: $SCREEN"
echo "Log: $LOG"
echo "Queue: one_bit train+matched+cross, then sixteen_bit train+matched+cross"

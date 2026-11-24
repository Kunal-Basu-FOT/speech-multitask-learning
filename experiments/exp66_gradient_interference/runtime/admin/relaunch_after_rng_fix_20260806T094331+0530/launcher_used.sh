#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
PY=/data/Kunal/env_speech_mtl/bin/python
SCREEN=exp66_gpu0
LOG="$ROOT/runtime/logs/exp66/master_queue.screen.log"
LOCK=/tmp/speech_mtl_exp66_gpu0.lock
[[ "$(hostname)" == "wtc10-server" ]] || { echo "ERROR: wrong host"; exit 1; }
[[ "$(whoami)" == "wtc10" ]] || { echo "ERROR: wrong user"; exit 1; }
[[ -d "$ROOT" ]] || { echo "ERROR: missing repository"; exit 1; }
if screen -list 2>/dev/null | grep -q "[.]$SCREEN"; then echo "ERROR: screen exists: $SCREEN"; exit 1; fi
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ' | tr ',' ' ')
(( USED <= 1024 && UTIL <= 10 )) || { echo "ERROR: GPU0 busy: memory=${USED}MiB util=${UTIL}%"; exit 1; }
mkdir -p "$(dirname "$LOG")" "$ROOT/runtime/runs/exp66"
CMD="cd '$ROOT' && flock -n '$LOCK' bash -lc 'export PYTHONPATH=\"$ROOT/src\" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 OMP_NUM_THREADS=8; '$PY' -m speech_mtl.exp66.cli --config configs/exp66/experiment.yaml run-queue 2>&1 | tee -a '$LOG''"
screen -dmS "$SCREEN" bash -lc "$CMD"
sleep 2
if ! screen -list 2>/dev/null | grep -q "[.]$SCREEN"; then echo "ERROR: screen terminated. Inspect $LOG"; tail -n 100 "$LOG" 2>/dev/null || true; exit 1; fi
echo "EXP 6.6 launched on WTC10 physical GPU0."
echo "Screen: $SCREEN"
echo "Log: $LOG"
echo "Queue: one-bit all methods + evaluations + report, then sixteen-bit all methods + evaluations + report."

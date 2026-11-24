#!/usr/bin/env bash
set -Eeuo pipefail
umask 002

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
CONFIG="$ROOT/configs/exp61/experiment.yaml"
SCREEN_NAME=exp61_gpu1
MASTER_LOG="$ROOT/runtime/logs/ecapa_7p2m/exp61/master_queue.screen.log"
LOCK_FILE=/tmp/speech_mtl_exp61_gpu1.lock

[[ "$(hostname)" == "wtc9-server" ]] || {
    echo "ERROR: launch must run on WTC9."
    exit 1
}

[[ -x "$PYTHON" ]] || {
    echo "ERROR: missing Python: $PYTHON"
    exit 1
}

[[ -f "$CONFIG" ]] || {
    echo "ERROR: missing config: $CONFIG"
    exit 1
}

if screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "ERROR: screen '$SCREEN_NAME' already exists."
    echo "Attach with: screen -r $SCREEN_NAME"
    exit 1
fi

mkdir -p "$(dirname "$MASTER_LOG")"

GPU_LINE="$(
    nvidia-smi \
      --query-gpu=index,name,memory.used,utilization.gpu \
      --format=csv,noheader,nounits |
    awk -F',' '$1 ~ /^[[:space:]]*1[[:space:]]*$/ {print}'
)"

[[ -n "$GPU_LINE" ]] || {
    echo "ERROR: physical GPU 1 was not found."
    exit 1
}

GPU_USED="$(
    awk -F',' '{gsub(/ /,"",$3); print $3}' <<<"$GPU_LINE"
)"
GPU_UTIL="$(
    awk -F',' '{gsub(/ /,"",$4); print $4}' <<<"$GPU_LINE"
)"

GPU_UUID="$(
    nvidia-smi \
      --query-gpu=index,uuid \
      --format=csv,noheader,nounits |
    awk -F',' '$1 ~ /^[[:space:]]*1[[:space:]]*$/ {
        gsub(/^[ \t]+|[ \t]+$/, "", $2);
        print $2
    }'
)"

GPU_PROCESSES="$(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null |
    awk -F',' -v uuid="$GPU_UUID" '
      {
        value=$1;
        gsub(/^[ \t]+|[ \t]+$/, "", value);
        if (value == uuid) print
      }
    ' || true
)"

echo "Physical GPU 1:"
echo "$GPU_LINE"

if [[ -n "$GPU_PROCESSES" ]]; then
    echo "ERROR: physical GPU 1 already has compute processes:"
    echo "$GPU_PROCESSES"
    exit 1
fi

if (( GPU_USED > 1024 )); then
    echo "ERROR: GPU 1 memory use is ${GPU_USED} MiB (>1024 MiB)."
    exit 1
fi

if (( GPU_UTIL > 10 )); then
    echo "ERROR: GPU 1 utilization is ${GPU_UTIL}% (>10%)."
    exit 1
fi

COMMAND=$(cat <<EOF
set -Eeuo pipefail
exec 9>"$LOCK_FILE"
flock -n 9 || {
    echo "ERROR: EXP 6.1 GPU1 lock is already held."
    exit 1
}
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
"$PYTHON" -m speech_mtl.exp61.cli \
    --config "$CONFIG" \
    queue 2>&1 | tee -a "$MASTER_LOG"
EOF
)

screen -dmS "$SCREEN_NAME" bash -lc "$COMMAND"

sleep 2

if ! screen -list 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "ERROR: screen exited immediately."
    tail -n 100 "$MASTER_LOG" 2>/dev/null || true
    exit 1
fi

echo
echo "EXP 6.1 queue launched."
echo "Screen: $SCREEN_NAME"
echo "Log:    $MASTER_LOG"
echo
echo "Attach now:"
echo "screen -r $SCREEN_NAME"

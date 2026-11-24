#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
SCRIPT="$ROOT/scripts/extract_exp61_references.py"
LOG="$ROOT/runtime/logs/exp61_reference_extract_gpu1.screen.log"
LOCK=/tmp/speech_mtl_exp61_reference_gpu1.lock

mkdir -p "$(dirname "$LOG")"

exec 9>"$LOCK"

if ! flock -n 9; then
    echo "Another EXP 6.1 reference extraction owns $LOCK"
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$EXP61/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

{
    echo "============================================================"
    echo " EXP 6.1 REFERENCE EXTRACTION"
    echo "============================================================"
    date --iso-8601=seconds
    hostname
    whoami
    nvidia-smi -i 1

    ionice -c2 -n7 \
    nice -n 10 \
    "$PYTHON" -u "$SCRIPT" --workers 4

    echo
    echo "EXP61_REFERENCE_EXTRACT_GPU1_QUEUE_COMPLETE"
    date --iso-8601=seconds
} 2>&1 | tee -a "$LOG"

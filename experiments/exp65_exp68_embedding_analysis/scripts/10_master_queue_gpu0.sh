#!/usr/bin/env bash
set -Eeuo pipefail

# Shared source roots required by EXP 6.5/6.8.
EXP61_SRC=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK/src
COMBINED_SRC=/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer/src
export PYTHONPATH="$COMBINED_SRC:$EXP61_SRC${PYTHONPATH:+:$PYTHONPATH}"

ROOT=/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
EXP61=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK
PYTHON=/data/Kunal/env_speech_mtl/bin/python
LOG="$ROOT/runtime/logs/exp65_exp68_gpu0.screen.log"
LOCK=/tmp/speech_mtl_exp65_exp68_gpu0.lock

mkdir -p "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "Another EXP 6.5/6.8 queue owns $LOCK"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/src:$EXP61/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=8
export MPLBACKEND=Agg

{
  echo "============================================================"
  echo " EXP 6.5 + EXP 6.8 — COMPLETE MASTER QUEUE"
  echo "============================================================"
  whoami
  hostname
  date --iso-8601=seconds
  "$PYTHON" --version
  nvidia-smi -i 0

  echo
  echo "===== PREFLIGHT ====="
  "$PYTHON" -u -m exp65_68.cli preflight

  echo
  echo "===== REAL-CHECKPOINT SMOKE ====="
  "$PYTHON" -u -m exp65_68.cli smoke --device cuda:0

  echo
  echo "===== EXP 6.2 + RANDOM EMBEDDING EXTRACTION ====="
  ionice -c2 -n4 nice -n 5 "$PYTHON" -u -m exp65_68.cli extract --device cuda:0 --workers 4

  echo
  echo "===== EXP 6.5 + EXP 6.8 ANALYSIS ====="
  ionice -c2 -n4 nice -n 5 "$PYTHON" -u -m exp65_68.cli analyze

  echo
  echo "===== FINAL VERIFICATION ====="
  "$PYTHON" -u -m exp65_68.cli verify

  echo
  echo "EXP65_EXP68_GPU0_QUEUE_COMPLETE"
  date --iso-8601=seconds
} 2>&1 | tee -a "$LOG"

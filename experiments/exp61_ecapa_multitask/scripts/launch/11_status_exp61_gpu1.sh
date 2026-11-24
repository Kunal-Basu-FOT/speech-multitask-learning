#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_ECAPA_MULTITASK

echo "===== SCREEN ====="
screen -list 2>/dev/null | grep -E 'exp61_gpu1|No Sockets' || true

echo
echo "===== GPU 1 ====="
nvidia-smi -i 1 || true

echo
echo "===== QUEUE PROGRESS ====="
cat \
  "$ROOT/runtime/runs/ecapa_7p2m/exp61/QUEUE_PROGRESS.json" \
  2>/dev/null || echo "No queue progress yet."

echo
echo "===== ONE-BIT PROGRESS ====="
cat \
  "$ROOT/runtime/runs/ecapa_7p2m/exp61/one_bit/progress.json" \
  2>/dev/null || echo "No one-bit progress yet."

echo
echo "===== SIXTEEN-BIT PROGRESS ====="
cat \
  "$ROOT/runtime/runs/ecapa_7p2m/exp61/sixteen_bit/progress.json" \
  2>/dev/null || echo "No sixteen-bit progress yet."

echo
echo "===== MASTER LOG TAIL ====="
tail -n 60 \
  "$ROOT/runtime/logs/ecapa_7p2m/exp61/master_queue.screen.log" \
  2>/dev/null || true

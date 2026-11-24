#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "===== SCREEN ====="
screen -list 2>/dev/null | grep -E 'exp62_gpu0|No Sockets' || true
echo "===== GPU 0 ====="
nvidia-smi -i 0
echo "===== QUEUE ====="
cat "$ROOT/runtime/runs/exp62_reduced/QUEUE_PROGRESS.json" 2>/dev/null || echo "No queue progress yet."
echo "===== LAST LOG ====="
tail -n 60 "$ROOT/runtime/logs/exp62_reduced/master_queue.screen.log" 2>/dev/null || true

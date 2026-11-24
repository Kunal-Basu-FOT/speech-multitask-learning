#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/Kunal/FORMAL_EXP6_6_GradientInterference_IndicWav2Vec_FinalLayer
echo "===== SCREEN ====="; screen -ls 2>&1 | grep -E 'exp66_gpu0|No Sockets' || true
echo "===== GPU0 ====="; nvidia-smi -i 0
echo "===== QUEUE PROGRESS ====="; cat "$ROOT/runtime/runs/exp66/QUEUE_PROGRESS.json" 2>/dev/null || echo "No queue progress."
echo "===== DOMAIN COMPLETION ====="
for domain in one_bit sixteen_bit; do
  echo "--- $domain ---"
  cat "$ROOT/runtime/runs/exp66/$domain/DOMAIN_COMPLETE.json" 2>/dev/null || echo "Not complete."
done
echo "===== CONDITION PROGRESS ====="
for domain in one_bit sixteen_bit; do
  for condition in standard_logged pcgrad gradnorm; do
    p="$ROOT/runtime/runs/exp66/$domain/$condition/progress.json"
    [[ -f "$p" ]] && { echo "--- $domain/$condition ---"; cat "$p"; }
  done
done
echo "===== LOG TAIL ====="; tail -n 80 "$ROOT/runtime/logs/exp66/master_queue.screen.log" 2>/dev/null || true

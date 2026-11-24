#!/usr/bin/env bash
ROOT=/data/Kunal/FORMAL_EXP6_7_Hierarchical_LID_Dialect_IndicWav2Vec_FinalLayer

echo "===== SCREEN ====="
screen -list 2>/dev/null | grep -E 'exp67_gpu1|No Sockets' || true

echo
echo "===== PHYSICAL GPU1 ====="
nvidia-smi -i 1

echo
echo "===== QUEUE PROGRESS ====="
cat "$ROOT/runtime/runs/exp67/QUEUE_PROGRESS.json" 2>/dev/null || echo "No queue progress file."

for domain in one_bit sixteen_bit; do
  echo
  echo "===== $domain PROGRESS ====="
  cat "$ROOT/runtime/runs/exp67/$domain/hierarchical_lid_dialect/progress.json" \
    2>/dev/null || echo "No progress file."
  echo
  echo "===== $domain COMPLETION ====="
  cat "$ROOT/runtime/runs/exp67/$domain/hierarchical_lid_dialect/TRAINING_COMPLETE.json" \
    2>/dev/null || echo "Training not complete."
done

echo
echo "===== RECENT LOG ====="
tail -n 60 "$ROOT/runtime/logs/exp67/master_queue.screen.log" 2>/dev/null || true

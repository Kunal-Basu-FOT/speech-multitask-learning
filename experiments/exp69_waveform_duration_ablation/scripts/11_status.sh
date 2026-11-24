#!/usr/bin/env bash
set +e
ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
RUN="$ROOT/runtime/runs/exp69"

echo "============================================================"
echo " EXP 6.9 STATUS"
echo "============================================================"
date --iso-8601=seconds

echo
echo "===== GPU 1 ====="
nvidia-smi -i 1 --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo
echo "===== SCREEN ====="
screen -ls 2>&1 | grep -E 'exp69_gpu1|There (is|are)|No Sockets' || true

echo
echo "===== QUEUE ====="
cat "$RUN/QUEUE_PROGRESS.json" 2>/dev/null || echo "QUEUE_NOT_STARTED"

echo
echo "===== CONDITIONS ====="
for domain in one_bit sixteen_bit; do
  for duration in 3s 7s 9s full; do
    root="$RUN/$domain/$duration"
    epochs=0
    [[ -s "$root/history.csv" ]] && epochs=$(( $(wc -l < "$root/history.csv") - 1 ))
    status=NOT_STARTED
    [[ -d "$root" ]] && status=PARTIAL
    [[ -s "$root/TRAINING_COMPLETE.json" ]] && status=TRAINING_COMPLETE
    [[ -s "$root/CONDITION_COMPLETE.json" ]] && status=COMPLETE
    printf '%-12s %-5s epochs=%-2s status=%s\n' "$domain" "$duration" "$epochs" "$status"
  done
done

echo
echo "===== CACHE ====="
cat /data/Kunal/shared_cache/exp69_waveform_logmel80/CACHE_COMPLETE.json 2>/dev/null || echo "CACHE_NOT_COMPLETE"

echo
echo "===== LOG TAIL ====="
tail -n 80 "$ROOT/runtime/logs/exp69_gpu1.screen.log" 2>/dev/null || true

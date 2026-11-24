#!/usr/bin/env bash
set +e
ROOT=/data/Kunal/FORMAL_EXP6_5_EXP6_8_Combined_Embedding_Analysis_IndicWav2Vec_FinalLayer
LOG="$ROOT/runtime/logs/exp65_exp68_gpu0.screen.log"
echo "===== TIME ====="; date --iso-8601=seconds
echo "===== SCREEN ====="; screen -ls 2>&1 | grep -E 'exp65_exp68_gpu0|No Sockets|There (is|are)' || true
echo "===== GPU 0 ====="; nvidia-smi -i 0
echo "===== PROCESS ====="; ps -eo user,pid,ppid,etime,%cpu,%mem,args | grep -E '[e]xp65_68|[e]xp65_exp68_gpu0' || true
echo "===== EXTRACTION MARKER ====="; cat "$ROOT/runtime/embeddings/EXP62_RANDOM_EXTRACTION_COMPLETE.json" 2>/dev/null | head -n 30 || echo PENDING
echo "===== REPORT MARKER ====="; cat "$ROOT/reports/combined/REPORT_COMPLETE.json" 2>/dev/null || echo PENDING
echo "===== FINAL MARKER ====="; cat "$ROOT/runtime/EXPERIMENT_COMPLETE.json" 2>/dev/null | head -n 40 || echo PENDING
echo "===== LOG TAIL ====="; tail -n 80 "$LOG" 2>/dev/null || true

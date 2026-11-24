#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/Kunal/FORMAL_EXP6_9_Waveform_Duration_Ablation_ECAPA
SCREEN=exp69_gpu1
LAUNCHER="$ROOT/scripts/10_master_queue_gpu1.sh"
LOG="$ROOT/runtime/logs/exp69_gpu1.screen.log"

[[ "$(whoami)" == "wtc9" ]] || { echo "ERROR: run as wtc9"; exit 1; }
[[ "$(hostname)" == "wtc9-server" ]] || { echo "ERROR: run on wtc9-server"; exit 1; }
[[ -x "$LAUNCHER" ]] || { echo "ERROR: missing launcher: $LAUNCHER"; exit 1; }

if pgrep -af '[p]ython.*exp69' >/tmp/exp69_live.txt; then
  echo "ERROR: EXP 6.9 is already running:"
  cat /tmp/exp69_live.txt
  exit 1
fi

GPU_PROCESSES="$(nvidia-smi -i 1 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "ERROR: physical GPU 1 is occupied:"
  echo "$GPU_PROCESSES"
  exit 1
fi

if screen -ls 2>/dev/null | grep -qE "\.${SCREEN}[[:space:]]"; then
  echo "ERROR: screen already exists: $SCREEN"
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
screen -dmS "$SCREEN" bash -lc "cd '$ROOT' && bash '$LAUNCHER'"
sleep 10

echo "===== SCREEN ====="
screen -ls 2>&1
echo
echo "===== GPU 1 ====="
nvidia-smi -i 1
echo
echo "===== LOG TAIL ====="
tail -n 60 "$LOG" 2>/dev/null || true

echo
echo "EXP69_GPU1_LAUNCHED"
echo "screen: $SCREEN"
echo "log:    $LOG"

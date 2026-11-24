#!/usr/bin/env bash
set -Eeuo pipefail
if screen -ls 2>/dev/null | grep -qE '\.exp69_gpu1[[:space:]]'; then
  screen -S exp69_gpu1 -X quit
  echo "EXP69_SCREEN_STOPPED"
else
  echo "No exp69_gpu1 screen exists."
fi

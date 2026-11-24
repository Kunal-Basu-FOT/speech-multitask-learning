#!/usr/bin/env bash
set -Eeuo pipefail
if screen -list 2>/dev/null | grep -q '[.]exp67_gpu1'; then
  screen -S exp67_gpu1 -X stuff $'\003'
  echo "Sent Ctrl-C to exp67_gpu1. The current incomplete epoch will restart on resume."
else
  echo "No exp67_gpu1 screen exists."
fi

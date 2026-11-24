#!/usr/bin/env bash
set -Eeuo pipefail
screen -S exp66_gpu0 -X stuff $'\003' 2>/dev/null || true
echo "Ctrl-C sent to exp66_gpu0. The incomplete epoch restarts on resume."

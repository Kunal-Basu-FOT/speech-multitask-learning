#!/usr/bin/env bash
set -Eeuo pipefail
screen -S exp62_gpu0 -X stuff $'\003' || true
echo "Sent Ctrl-C to exp62_gpu0. The incomplete epoch will restart on resume."

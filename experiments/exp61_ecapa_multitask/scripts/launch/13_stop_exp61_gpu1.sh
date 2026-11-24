#!/usr/bin/env bash
set -Eeuo pipefail

if screen -list 2>/dev/null | grep -q '[.]exp61_gpu1[[:space:]]'; then
    screen -S exp61_gpu1 -X stuff $'\003'
    echo "Sent Ctrl-C to exp61_gpu1."
    echo "The current epoch may need to restart from its beginning."
else
    echo "No exp61_gpu1 screen is running."
fi

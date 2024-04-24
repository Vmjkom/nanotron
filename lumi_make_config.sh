#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 CONFIG.PY" >&2
    exit 1
fi
CONFIG_PY="$1"

source lumi_settings.sh

singularity_exec "python3 $CONFIG_PY"

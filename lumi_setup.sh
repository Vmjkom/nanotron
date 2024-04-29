#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

source lumi_settings.sh

rm -rf "$PYTHONUSERBASE"
mkdir -p "$PYTHONUSERBASE"

CMD="
python3 -m pip install -r $SCRIPT_DIR/lumi_requirements.txt --user
echo 'Installation finished'
"

singularity_exec "$CMD"

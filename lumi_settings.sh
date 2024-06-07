#!/bin/bash

# Common settings and functionality for running on LUMI

export PYTHONUSERBASE="pythonuserbase" 
export PYTHONPATH="$PYTHONPATH:src"

export CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"

REAL_PWD="$(realpath "$PWD")"
export SINGULARITY_BIND="$REAL_PWD"
export SINGULARITY_BIND=$SINGULARITY_BIND,"/pfs,/scratch,/projappl,/project,/flash,/appl"
singularity_exec() {
    cmd="$1"
    real_pwd="$(realpath "$PWD")"
    cmd="source /opt/miniconda3/bin/activate pytorch; $cmd"
    singularity exec --pwd "$real_pwd" "$CONTAINER" bash -c "$cmd"
}

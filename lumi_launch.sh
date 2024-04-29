#!/bin/bash

source lumi_settings.sh

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1

echo "Run on $SLURMD_NODENAME ($SLURM_PROCID/$WORLD_SIZE)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"

CMD="python3 $@"

echo "CMD $CMD"

singularity_exec "$CMD"

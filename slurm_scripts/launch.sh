#!/bin/bash
#Tämän skriptin kaikki hommelit ajetaan lumi_train.sh skriptistä tulevasta "n_tasks" kertaa
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID


echo "Run on $SLURMD_NODENAME ($SLURM_PROCID/$WORLD_SIZE)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"


python3 "$@"
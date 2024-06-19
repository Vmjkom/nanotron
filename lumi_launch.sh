#!/bin/bash
#Tämän skriptin kaikki hommelit ajetaan lumi_train.sh skriptistä tulevasta "n_tasks" kertaa
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export OMP_NUM_THREADS=16

#Activate the python env from inside the container
source /opt/miniconda3/bin/activate pytorch

echo "Run on $SLURMD_NODENAME ($SLURM_PROCID/$WORLD_SIZE)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE"

CMD="python3 -u -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3"

python3 "$@"
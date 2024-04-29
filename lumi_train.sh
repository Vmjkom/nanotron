#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --partition=dev-g
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000558
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

source lumi_config.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

CONFIG="config_tiny_llama.yaml"

# create hostfile
HOSTFILE="hostfiles/$SLURM_JOB_ID.txt"
mkdir -p $(dirname "$HOSTFILE")
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read n; do
    echo "$n slots=$SLURM_GPUS_ON_NODE" >> "$HOSTFILE"
done

echo "START: $(date)"

srun --label lumi_launch.sh run_train.py --config-file "$CONFIG"
#--hostfile "$HOSTFILE"

echo "END: $(date)"

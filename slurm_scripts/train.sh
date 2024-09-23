#!/bin/bash
#SBATCH -J debug_nanotron_LLAMA2_7B_2N
#SBATCH --cpus-per-task=7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive
#SBATCH --account=project_462000615
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eox pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

DIR="/projappl/project_462000353/villekom/nanotron"

module purge
ml use /appl/local/csc/modulefiles/
ml pytorch/2.4
source $DIR/.venv/bin/activate
export PYTHONPATH=/projappl/project_462000353/villekom/nanotron/.venv/lib/python3.10/site-packages
echo "NGPUS" $SLURM_GPUS_ON_NODE
export NCCL_IFNAME=hsn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
#export NANOTRON_BENCHMARK=1 #Logs throughput
#export HF_TOKEN="TOKEN HERE FROM HF"
export CONFIG=$DIR/configs/llama_2B.yaml

#Debugging
#export NCCL_DEBUG=INFO

#Masks for binding cpu cores to right numa nodes and therefor to right gpu's
#c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START: $(date)"

srun --label $DIR/slurm_scripts/launch.sh \
    $DIR/run_train.py --config-file "$CONFIG"

echo "END: $(date)"
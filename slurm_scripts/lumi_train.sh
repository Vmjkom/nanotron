#!/bin/bash
#SBATCH -J debug_nanotron_LLAMA2_7B_2N
#SBATCH --cpus-per-task=7
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive
#SBATCH --account=project_462000558
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eox pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
#export NANOTRON_BENCHMARK=1 #Logs throughput
export HF_TOKEN="TOKEN HERE FROM HF"
export CONFIG="examples/config_llama.yaml"

#Prepend variables that you want singularity to export at runtime
#SINGULARITYENV_ is needed to override variables of the same name inside the container
#More info https://docs.sylabs.io/guides/3.7/user-guide/environment_and_metadata.html#environment-overview
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false #Disable forking the FastTokenizer
export SINGULARITYENV_TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_HF_HOME=/scratch/project_462000353/transformers_cache
export SINGULARITYENV_PYTHONWARNINGS=ignore #Decrease verbosity in logging. Pytorch warnings log on every rank
export SINGULARITYENV_PYTHONUSERBASE="pythonuserbase" 
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH:src"
export SINGULARITYENV_TRANSFORMERS_VERBOSITY=error
export SINGULARITYENV_TRANSFORMERS_NO_ADVISORY_WARNINGS=1


#SINGULARITY
export CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"
export SINGULARITY_BIND="/pfs,/scratch,/projappl,/project,/flash,/appl,/var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4"

#Masks for binding cpu cores to right numa nodes and therefor to right gpu's
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START: $(date)"
PWD=$(realpath "$PWD")
srun --cpu-bind=mask_cpu:$MYMASKS --label singularity exec \
    -B $PWD:$PWD \
    $CONTAINER \
    $PWD/lumi_launch.sh \
    $PWD/run_train.py --config-file "$CONFIG"

echo "END: $(date)"

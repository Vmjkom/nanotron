#!/bin/bash
#SBATCH -A project_462000615
#SBATCH -J data_preprocess
#SBATCH -N 1
#SBATCH -p small
#SBATCH --cpus-per-task=64
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=900G

set -xeo
module purge
module load pytorch/2.2
source /projappl/project_462000353/villekom/nanotron/.venv/bin/activate
# Print the task index.
#export RANK=$SLURM_PROCID
#export WORLD_SIZE=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

echo "Starting preprocessing"
torchrun --nproc-per-node=$SLURM_CPUS_PER_TASK --standalone preprocess_data.py \
    --input /scratch/project_462000353/jburdge/data/fineweb-100B \
    --tokenizer_name_or_path HuggingFaceFW/ablation-model-fineweb-edu \
    --output-prefix /scratch/project_462000353/data/fineweb/100BT
echo "Done"
#python3 preprocess_data.py \
#    --input /scratch/project_462000353/jburdge/data/fineweb-100B \
#    --tokenizer_name_or_path HuggingFaceFW/ablation-model-fineweb-v1 \
#    --output-prefix /scratch/project_462000353/data/fineweb/100BT/
#echo "Done"
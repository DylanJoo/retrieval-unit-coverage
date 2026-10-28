#!/bin/bash -l
#SBATCH --job-name=claimgen
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

WORK=${HOME}/scratch/retrieval-unit-coverage

METHODS=(
"dice"
"wice"
)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

python -m claimgen.generate_claims \
    --method $METHOD \
    --corpus ${WORK}/units/pool.jsonl.gz \
    --output ${WORK}/claims/pool.${METHOD}.Qwen3-32B.jsonl \
    --model Qwen/Qwen3-32B \
    --tp 2

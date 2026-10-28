#!/bin/bash -l
#SBATCH --job-name=claimgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Submit from the repo root. Adapted from ~/ClaimEmbeddings/scripts/generate_claims_lumi.sh.
#   sbatch scripts/generate_claims.sh                                  # dice, Qwen3-32B on 2 GPUs
#   METHOD=wice sbatch scripts/generate_claims.sh                      # WiCE Claim-Split
#   MODEL=Qwen/Qwen3-4B TP=1 sbatch --gres=gpu:l40s:1 scripts/generate_claims.sh
#   NUM_SHARDS=4 sbatch --array=0-3 scripts/generate_claims.sh         # shards -> *.shard{i}of4.jsonl

source ${HOME}/.bashrc; initconda; conda activate ${ENV:-inference}

WORK=${WORK:-$HOME/scratch/retrieval-unit-coverage}
METHOD=${METHOD:-dice}  # dice | wice
MODEL=${MODEL:-Qwen/Qwen3-32B}
TP=${TP:-2}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
CORPUS=${CORPUS:-$WORK/units/pool.jsonl.gz}
NAME=$(basename $CORPUS); NAME=${NAME%.gz}; NAME=${NAME%.jsonl}
OUTPUT=${OUTPUT:-$WORK/claims/${NAME}.${METHOD}.${MODEL##*/}.jsonl}

python -m claimgen.generate_claims \
    --method $METHOD \
    --corpus $CORPUS \
    --output $OUTPUT \
    --model $MODEL \
    --tp $TP \
    --num_shards $NUM_SHARDS \
    --shard_id $SHARD_ID

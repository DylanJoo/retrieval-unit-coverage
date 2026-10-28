#!/bin/bash -l
#SBATCH --job-name=retrieve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Submit from the repo root. Writes runs/ragtime1.$NAME.trec (unit ids; evaluate.py maps them to docs)
#   NAME=sent3 sbatch scripts/retrieve.sh
#   NAME=doc REPS="$HOME/scratch/ragtime1/modernbert-base.cover-5k/docs_emb/*.pkl" sbatch scripts/retrieve.sh  # full-corpus docs

source ${HOME}/.bashrc; initconda; conda activate ${ENV:-inference}

WORK=${WORK:-$HOME/scratch/retrieval-unit-coverage}
MODEL=${MODEL:-DylanJHJ/modernbert-base.cover-5k}
NAME=${NAME:?set NAME}
REPS=${REPS:-$WORK/emb/${MODEL##*/}/$NAME/*.pkl}

python run_retrieval.py \
    --queries data/ragtime1/topics.jsonl \
    --passage_reps "$REPS" \
    --model_name_or_path $MODEL \
    --query_prefix "search_query: " \
    --pooling mean \
    --query_max_len 128 \
    --depth ${DEPTH:-1000} \
    --run_name $NAME \
    --output runs/ragtime1.$NAME.trec

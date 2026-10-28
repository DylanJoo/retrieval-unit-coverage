#!/bin/bash -l
#SBATCH --job-name=retrieve
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

WORK=${HOME}/scratch/retrieval-unit-coverage
MODEL=DylanJHJ/modernbert-base.cover-5k

# "doc" is the full-corpus doc baseline (existing embeddings); the rest are pool units
NAMES=(
"doc"
"pool"
"sent1"
"sent3"
"stmt"
"claim-dice"
"claim-wice"
)
REPS=(
"${HOME}/scratch/ragtime1/${MODEL##*/}/docs_emb/*.pkl"
"${WORK}/emb/${MODEL##*/}/pool/*.pkl"
"${WORK}/emb/${MODEL##*/}/sent1/*.pkl"
"${WORK}/emb/${MODEL##*/}/sent3/*.pkl"
"${WORK}/emb/${MODEL##*/}/stmt/*.pkl"
"${WORK}/emb/${MODEL##*/}/claim-dice/*.pkl"
"${WORK}/emb/${MODEL##*/}/claim-wice/*.pkl"
)

for i in "${!NAMES[@]}"; do
    name=${NAMES[i]}
    if [[ -f runs/ragtime1.${name}.trec ]]; then
        echo "Skip ${name}: runs/ragtime1.${name}.trec exists"
        continue
    fi
    python run_retrieval.py \
        --queries data/ragtime1/topics.jsonl \
        --passage_reps "${REPS[i]}" \
        --model_name_or_path $MODEL \
        --query_prefix "search_query: " \
        --pooling mean \
        --query_max_len 128 \
        --depth 1000 \
        --run_name $name \
        --output runs/ragtime1.${name}.trec
done

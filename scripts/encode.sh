#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

# Submit from the repo root. Encodes $WORK/units/$NAME.jsonl.gz -> $WORK/emb/$MODEL/$NAME/emb.{shard}.pkl
#   NAME=sent3 sbatch scripts/encode.sh
#   NAME=sent1 NUM_SHARDS=4 sbatch --array=0-3 scripts/encode.sh

source ${HOME}/.bashrc; initconda; conda activate ${ENV:-inference}

WORK=${WORK:-$HOME/scratch/retrieval-unit-coverage}
MODEL=${MODEL:-DylanJHJ/modernbert-base.cover-5k}
NAME=${NAME:?set NAME}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
OUT_DIR=$WORK/emb/${MODEL##*/}/$NAME
mkdir -p $OUT_DIR

python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path $MODEL \
    --tokenizer_name ${TOKENIZER:-answerdotai/ModernBERT-base} \
    --pooling mean --normalize --bf16 \
    --passage_prefix "search_document: " \
    --passage_max_len ${PASSAGE_MAX_LEN:-1024} \
    --per_device_eval_batch_size ${BATCH_SIZE:-512} \
    --dataset_path $WORK/units/$NAME.jsonl.gz \
    --dataset_number_of_shards $NUM_SHARDS \
    --dataset_shard_index $SHARD_ID \
    --encode_output_path $OUT_DIR/emb.$SHARD_ID.pkl

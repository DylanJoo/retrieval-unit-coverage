#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

WORK=${HOME}/scratch/retrieval-unit-coverage
MODEL=DylanJHJ/modernbert-base.cover-5k

NAMES=(
"pool"
"sent1"
"sent3"
"stmt"
"claim-dice"
"claim-wice"
)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
output_dir=${WORK}/emb/${MODEL##*/}/${NAME}
mkdir -p $output_dir

python -m tevatron.retriever.driver.encode \
    --output_dir temp \
    --model_name_or_path $MODEL \
    --tokenizer_name answerdotai/ModernBERT-base \
    --pooling mean --normalize --bf16 \
    --passage_prefix "search_document: " \
    --passage_max_len 1024 \
    --per_device_eval_batch_size 512 \
    --dataset_path ${WORK}/units/${NAME}.jsonl.gz \
    --encode_output_path ${output_dir}/emb.0.pkl

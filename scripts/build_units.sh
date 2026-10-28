#!/bin/bash -l
#SBATCH --job-name=build-units
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# Submit from the repo root. Writes $WORK/units/$NAME.jsonl.gz
#   NAME=pool UNIT=doc RUN=runs/ragtime1.doc.trec TOPK=100 sbatch scripts/build_units.sh  # doc pool of a baseline
#   NAME=sent1 UNIT=sentence sbatch scripts/build_units.sh
#   NAME=sent3 UNIT=sentence SIZE=3 STRIDE=1 sbatch scripts/build_units.sh
#   NAME=claim-dice UNIT=claim CLAIMS=$WORK/claims/pool.dice.jsonl sbatch scripts/build_units.sh
#   NAME=stmt UNIT=claim sbatch scripts/build_units.sh                                    # corpus "statements" field

source ${HOME}/.bashrc; initconda; conda activate ${ENV:-inference}

WORK=${WORK:-$HOME/scratch/retrieval-unit-coverage}
CORPUS=${CORPUS:-$WORK/units/pool.jsonl.gz}   # use $HOME/scratch/ragtime1/*.processed-claims.jsonl.gz for the full corpus
NAME=${NAME:?set NAME}
UNIT=${UNIT:?set UNIT (doc|sentence|claim)}

python build_units.py \
    --corpus $CORPUS \
    --unit $UNIT \
    --size ${SIZE:-1} \
    ${STRIDE:+--stride $STRIDE} \
    ${CLAIMS:+--claims $CLAIMS} \
    ${RUN:+--run $RUN --topk ${TOPK:-100}} \
    --output $WORK/units/$NAME.jsonl.gz

#!/bin/bash -l
#SBATCH --job-name=build-units
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

# ENV
source ${HOME}/.bashrc
initconda
conda activate inference

WORK=${HOME}/scratch/retrieval-unit-coverage
POOL=${WORK}/units/pool.jsonl.gz

# doc units = pool: top-100 docs per topic of the full-corpus doc baseline (runs/ragtime1.doc.trec)
if [[ ! -f $POOL ]]; then
    python build_units.py \
        --corpus ${HOME}/scratch/ragtime1/*.processed-claims.jsonl.gz \
        --unit doc \
        --run runs/ragtime1.doc.trec \
        --topk 100 \
        --output $POOL
fi

# sentence units
python build_units.py --corpus $POOL --unit sentence --size 1 --output ${WORK}/units/sent1.jsonl.gz
python build_units.py --corpus $POOL --unit sentence --size 3 --stride 1 --output ${WORK}/units/sent3.jsonl.gz

# claim units: corpus "statements" field, then claimgen outputs (scripts/generate_claims.sh)
python build_units.py --corpus $POOL --unit claim --size 1 --output ${WORK}/units/stmt.jsonl.gz
for method in dice wice; do
    claims=${WORK}/claims/pool.${method}.Qwen3-32B.jsonl
    if [[ -f $claims ]]; then
        python build_units.py --corpus $POOL --unit claim --size 1 --claims $claims \
            --output ${WORK}/units/claim-${method}.jsonl.gz
    fi
done

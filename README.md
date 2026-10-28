# retrieval-unit-coverage

How does the **retrieval unit** (document, sentence, multi-sentence window, claim, ...) affect **information coverage**?
We index the same documents as different units, retrieve with a dense retriever (tevatron), map the ranked
units back to documents, and measure nugget coverage on TREC RAGTIME 2025 (ragtime1) with α-nDCG and subtopic recall.

```
build_units.py        (1) documents -> retrieval units (doc / sentence windows / claim windows)
claimgen/             (1) LLM claim extraction with vLLM (copied from ~/ClaimEmbeddings/claimgen)
run_retrieval.py      (2) dense retrieval over tevatron embeddings -> TREC run (adapted from ~/trec2026)
evaluate.py           (3) α-nDCG / StRecall on nugget qrels (adapted from ~/trec2026/src/evaluator/rac_eval.py)
scripts/*.sh          slurm wrappers for each step
data/ragtime1/        topics.jsonl (122 topics), nuggets.qrel (nugget-level judgments, 34 topics)
```

## Setup

One conda env (`ENV`, default `inference`) with:
- tevatron from Dylan's fork: `pip install -e ~/tevatron` (github.com/DylanJoo/tevatron)
- `faiss`, `transformers`, `flash-attn` (retrieval), `vllm` (claimgen), `ir_measures`, `pandas` (evaluation)

All scripts are submitted **from the repo root**. Large files go to `WORK` (default `~/scratch/retrieval-unit-coverage`):
```
$WORK/units/<NAME>.jsonl.gz            unit corpus ({docid, title, text})
$WORK/claims/<corpus>.<method>.<model>.jsonl   claimgen outputs
$WORK/emb/<model>/<NAME>/emb.<shard>.pkl       tevatron embeddings
runs/ragtime1.<NAME>.trec              TREC runs (in the repo)
```

## Conventions

- **Unit ids** are `{docid}#{k}` (k = unit index within the document). The `doc` unit keeps the plain docid.
- **Evaluation is document-level**: the qrels judge documents, so `evaluate.py` max-pools each run back to documents
  (a document is ranked at its best unit, duplicates dropped). `@10`/`@20` therefore count **documents**, not units.
- Qrels format: `qid nugget_id docid rel`. The nugget id is the subtopic for α-nDCG/StRecall.
  Only 34/122 topics have nuggets. Topics missing from a run score 0.

## Pipeline

The default setup compares units over a **shared document pool**: the top-100 documents per topic from the
doc-level baseline. Every unit type then sees the same documents, and LLM claim extraction stays affordable
(~12k docs instead of the full ragtime1 corpus; full-corpus sentence units would be ~100M+ vectors).
Set `CORPUS=$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz` to build units over the full corpus instead.

**0. Doc baseline + pool.** The full ragtime1 corpus is already encoded with `modernbert-base.cover-5k`:
```bash
NAME=doc REPS="$HOME/scratch/ragtime1/modernbert-base.cover-5k/docs_emb/*.pkl" sbatch scripts/retrieve.sh
NAME=pool UNIT=doc CORPUS="$HOME/scratch/ragtime1/*.processed-claims.jsonl.gz" \
    RUN=runs/ragtime1.doc.trec TOPK=100 sbatch scripts/build_units.sh
```

**1. Retrieval units** (`build_units.py`; `--size` items per unit, `--stride` defaults to `--size`, i.e. no overlap):
```bash
NAME=doc-pool UNIT=doc                     sbatch scripts/build_units.sh   # documents (pool baseline)
NAME=sent1    UNIT=sentence                sbatch scripts/build_units.sh   # single sentences
NAME=sent3    UNIT=sentence SIZE=3 STRIDE=1 sbatch scripts/build_units.sh  # sliding 3-sentence windows
NAME=stmt     UNIT=claim                   sbatch scripts/build_units.sh   # existing "statements" field of the corpus

# your own claims (claimgen: dice = passage-level decontextualized claims, wice = WiCE per-sentence split)
sbatch scripts/generate_claims.sh                      # -> $WORK/claims/pool.dice.Qwen3-32B.jsonl
METHOD=wice sbatch scripts/generate_claims.sh
NAME=claim-dice UNIT=claim CLAIMS=$WORK/claims/pool.dice.Qwen3-32B.jsonl sbatch scripts/build_units.sh
NAME=claim-dice3 UNIT=claim SIZE=3 CLAIMS=... sbatch scripts/build_units.sh   # groups of 3 claims
```
Sentences are split with the regex splitter in `claimgen/generate_claims.py` (the same one WiCE mode uses).
claimgen resumes from an existing output file. With `NUM_SHARDS>1` it writes `*.shard{i}of{N}.jsonl`;
`cat` them into one file before `build_units.sh`.

**2. Encode + retrieve** (tevatron; defaults match the existing doc embeddings: mean pooling, normalized,
`search_document: ` / `search_query: ` prefixes):
```bash
NAME=sent3 sbatch scripts/encode.sh                               # add NUM_SHARDS=4 --array=0-3 for large unit sets
NAME=sent3 sbatch --dependency=afterok:<encode_jobid> scripts/retrieve.sh
```
`run_retrieval.py` searches shards one at a time and keeps a running top-`DEPTH` (default 1000 units), so memory
is bounded by the largest shard. When units are small, 1000 units may cover fewer than 1000 distinct documents.
That is fine for @10/@20, but raise `DEPTH` if you need deeper document cutoffs.

**3. Evaluate** (seconds on CPU, no slurm needed):
```bash
python evaluate.py --runs runs/ragtime1.*.trec                    # --cutoffs 10 20 by default
```
Prints a markdown table. Sanity check: the full-corpus doc baseline from `~/trec2026/runs/runs.ragtime2025.cover.test.txt`
gives α-nDCG@10 0.5230, α-nDCG@20 0.5491, StRecall@10 0.6592, StRecall@20 0.7364 (identical to `rac_eval`).

## Notes / next steps

- Different units have different lengths. When feeding units to a generator (planned RAG step), compare at
  equal token budgets as well as at equal unit counts.
- Encoding uses `PASSAGE_MAX_LEN=1024` for every unit type. Documents longer than that are truncated, so the doc baseline
  only sees each document's beginning, while sentence/claim units cover the whole document.
- Using a different retriever (e.g. Qwen3-Embedding): set `MODEL`/`TOKENIZER` for encoding and pass
  `--pooling last --append_eos` plus that model's prefixes to `run_retrieval.py`.

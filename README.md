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

One conda env, `inference` (the name is hard-coded in `scripts/*.sh`), with:
- tevatron from Dylan's fork: `pip install -e ~/tevatron` (github.com/DylanJoo/tevatron)
- `faiss`, `transformers`, `flash-attn` (retrieval), `vllm` (claimgen), `ir_measures`, `pandas` (evaluation)

All scripts are submitted **from the repo root**. All values (paths, model, unit types) are fixed in the scripts; edit them there.
Large files go to `~/scratch/retrieval-unit-coverage` (`WORK` in the scripts):
```
$WORK/units/<name>.jsonl.gz                       unit corpus ({docid, title, text})
$WORK/claims/pool.<method>.Qwen3-32B.jsonl        claimgen outputs
$WORK/emb/modernbert-base.cover-5k/<name>/*.pkl   tevatron embeddings
runs/ragtime1.<name>.trec                         TREC runs (in the repo)
```

## Conventions

- **Unit ids** are `{docid}#{k}` (k = unit index within the document). The `doc` unit keeps the plain docid.
- **Evaluation is document-level**: the qrels judge documents, so `evaluate.py` max-pools each run back to documents
  (a document is ranked at its best unit, duplicates dropped). `@10`/`@20` therefore count **documents**, not units.
- Qrels format: `qid nugget_id docid rel`. The nugget id is the subtopic for α-nDCG/StRecall.
  Only 34/122 topics have nuggets. Topics missing from a run score 0.

## Pipeline

Units are compared over a **shared document pool**: the top-100 documents per topic from the full-corpus
doc-level baseline. Every unit type then sees the same documents, and LLM claim extraction stays affordable
(~12k docs instead of the full ragtime1 corpus; full-corpus sentence units would be ~100M+ vectors).

| name | unit | built by |
|---|---|---|
| `doc` | full-corpus documents (existing `docs_emb`, baseline only) | - |
| `pool` | pool documents | `build_units.py --unit doc --run runs/ragtime1.doc.trec --topk 100` |
| `sent1` | single sentences | `--unit sentence --size 1` |
| `sent3` | sliding 3-sentence windows | `--unit sentence --size 3 --stride 1` |
| `stmt` | single claims from the corpus `statements` field | `--unit claim --size 1` |
| `claim-dice` / `claim-wice` | single claims from claimgen | `--unit claim --size 1 --claims ...` |

Run in this order. Each step re-runs safely: finished pool, claims and runs are skipped or resumed.
```bash
sbatch scripts/retrieve.sh          # 1. doc baseline -> runs/ragtime1.doc.trec (other units fail: not encoded yet)
sbatch scripts/build_units.sh       # 2. pool + sent1/sent3/stmt units
sbatch scripts/generate_claims.sh   # 3. dice + wice claims on the pool (array 0-1, Qwen3-32B, 2 GPUs)
sbatch scripts/build_units.sh       # 4. again, now also builds claim-dice / claim-wice
sbatch scripts/encode.sh            # 5. encode all units (array 0-5, one per unit type)
sbatch scripts/retrieve.sh          # 6. retrieve with every unit type
python evaluate.py --runs runs/ragtime1.*.trec   # 7. seconds on CPU, no slurm needed
```

To add a unit type (e.g. groups of 3 claims, `--size 3`), add a `build_units.py` line in `scripts/build_units.sh`
and add its name to the arrays in `scripts/encode.sh` (also update `--array`) and `scripts/retrieve.sh`.

Details:
- Sentences are split with the regex splitter in `claimgen/generate_claims.py` (the same one WiCE mode uses).
  claimgen methods: `dice` = passage-level decontextualized claims, `wice` = WiCE Claim-Split per sentence.
- Encoding matches the existing doc embeddings: mean pooling, normalized, `search_document: ` / `search_query: ` prefixes.
- `run_retrieval.py` searches embedding shards one at a time and keeps a running top-1000 units, so memory is
  bounded by the largest shard. With small units, 1000 units may cover fewer than 1000 distinct documents.
  That is fine for @10/@20, but raise `--depth` for deeper document cutoffs.
- `evaluate.py` prints a markdown table. Sanity check: the full-corpus doc baseline
  (`~/trec2026/runs/runs.ragtime2025.cover.test.txt`) gives α-nDCG@10 0.5230, α-nDCG@20 0.5491,
  StRecall@10 0.6592, StRecall@20 0.7364 (identical to `rac_eval`).

## Notes / next steps

- Different units have different lengths. When feeding units to a generator (planned RAG step), compare at
  equal token budgets as well as at equal unit counts.
- Encoding uses `--passage_max_len 1024` for every unit type. Documents longer than that are truncated, so the doc baseline
  only sees each document's beginning, while sentence/claim units cover the whole document.
- Using a different retriever (e.g. Qwen3-Embedding): change `MODEL` and `--tokenizer_name` in the scripts and pass
  `--pooling last --append_eos` plus that model's prefixes to `run_retrieval.py`.

"""Coverage evaluation: alpha-nDCG and subtopic recall (StRecall) on nugget qrels.

Adapted from ~/trec2026/src/evaluator/rac_eval.py. Qrels are "qid nugget_id docid rel"; each nugget
acts as a subtopic. Qrels are document-level, so unit ids "{docid}#{k}" are max-pooled to their
document (a document is ranked at its best unit) before scoring; cutoffs therefore count documents.

Example:
    python evaluate.py --qrels data/ragtime1/nuggets.qrel --runs runs/ragtime1.*.trec
"""
import argparse
import logging

import ir_measures
import pandas as pd
from ir_measures import StRecall, alpha_nDCG


def load_run(path):
    run = pd.read_csv(path, sep=r"\s+", names=["query_id", "q0", "doc_id", "rank", "score", "tag"],
                      dtype={"query_id": str, "doc_id": str})
    run["doc_id"] = run["doc_id"].str.split("#").str[0]
    run = run.sort_values(["query_id", "score"], ascending=[True, False])
    return run.drop_duplicates(["query_id", "doc_id"])[["query_id", "doc_id", "score"]]


def main(args):
    qrels = pd.read_csv(args.qrels, sep=r"\s+", names=["query_id", "iteration", "doc_id", "relevance"],
                        dtype={"query_id": str, "iteration": str, "doc_id": str})
    measures = [m @ k for m in (alpha_nDCG, StRecall) for k in args.cutoffs]
    print("| run | " + " | ".join(map(str, measures)) + " |")
    print("|---|" + "---:|" * len(measures))
    for path in args.runs:
        run = load_run(path)
        missing = set(qrels.query_id) - set(run.query_id)
        if missing:
            logging.warning(f"{path}: {len(missing)} topics missing from run; scored as 0")
        # average over all judged topics, so missing topics count as 0
        per_query = {m: {q: 0.0 for q in qrels.query_id.unique()} for m in measures}
        for r in ir_measures.iter_calc(measures, qrels, run):
            per_query[r.measure][r.query_id] = r.value
        means = [sum(v.values()) / len(v) for v in per_query.values()]
        print(f"| {path} | " + " | ".join(f"{x:.4f}" for x in means) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default="data/ragtime1/nuggets.qrel")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 20])
    main(parser.parse_args())

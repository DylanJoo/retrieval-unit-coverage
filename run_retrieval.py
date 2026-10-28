"""Dense retrieval with tevatron: encode queries, search pre-encoded unit shards, write a TREC run.

Adapted from ~/trec2026/run_retrieval.py. Shards are searched one at a time and merged, so
large unit corpora (e.g. sentences) never need to fit in memory at once.

Example:
    python run_retrieval.py --queries data/ragtime1/topics.jsonl \
        --passage_reps "emb/modernbert-base.cover-5k/sent3/*.pkl" --output runs/ragtime1.sent3.trec
"""
import argparse
import glob
import json
import os
import pickle
from contextlib import nullcontext

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.retriever.modeling import DenseModel


def load_queries(path):
    with open(path) as f:
        return {str(q["qid"]): q["query"] for q in map(json.loads, f) if q}


@torch.no_grad()
def encode_queries(args, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.no_bf16 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = DenseModel.load(
        args.model_name_or_path,
        pooling=args.pooling,
        normalize=args.normalize,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
    ).to(device).eval()

    autocast = torch.amp.autocast("cuda") if dtype != torch.float32 and device.type == "cuda" else nullcontext()
    max_len = args.query_max_len - 1 if args.append_eos else args.query_max_len
    reps = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding queries"):
        enc = tokenizer(texts[start : start + args.batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        if args.append_eos:  # e.g. last-token pooling models (Qwen3-Embedding)
            n = enc["input_ids"].shape[0]
            enc["input_ids"] = torch.cat([enc["input_ids"], torch.full((n, 1), tokenizer.eos_token_id)], 1)
            enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones((n, 1), dtype=torch.long)], 1)
        with autocast:
            reps.append(model(query={k: v.to(device) for k, v in enc.items()}).q_reps.float().cpu().numpy())
    return np.concatenate(reps)


def search(q_reps, pattern, depth):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No embedding files match: {pattern}")
    scores = np.full((len(q_reps), 0), -np.inf, dtype=np.float32)
    ids = np.empty((len(q_reps), 0), dtype=object)
    for path in tqdm(files, desc="Searching shards"):
        with open(path, "rb") as f:
            reps, lookup = pickle.load(f)
        index = faiss.IndexFlatIP(q_reps.shape[1])
        index.add(np.asarray(reps, dtype=np.float32))
        s, i = index.search(q_reps, min(depth, len(lookup)))
        # merge with the running top-k
        scores = np.concatenate([scores, s], 1)
        ids = np.concatenate([ids, np.asarray(lookup, dtype=object)[i]], 1)
        top = np.argsort(-scores, 1, kind="stable")[:, :depth]
        scores, ids = np.take_along_axis(scores, top, 1), np.take_along_axis(ids, top, 1)
    return scores, ids


def main(args):
    queries = load_queries(args.queries)
    qids = list(queries)
    q_reps = encode_queries(args, [args.query_prefix + queries[q] for q in qids])
    scores, ids = search(q_reps, args.passage_reps, args.depth)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for qid, q_scores, q_ids in zip(qids, scores, ids):
            for rank, (s, uid) in enumerate(zip(q_scores, q_ids), 1):
                f.write(f"{qid} Q0 {uid} {rank} {s:.6f} {args.run_name}\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="jsonl with qid, query")
    parser.add_argument("--passage_reps", required=True, help="glob of tevatron encode outputs (*.pkl)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name_or_path", default="DylanJHJ/modernbert-base.cover-5k")
    parser.add_argument("--query_prefix", default="search_query: ")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query_max_len", type=int, default=128)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--depth", type=int, default=1000, help="units retrieved per query")
    parser.add_argument("--run_name", default="tevatron")
    main(parser.parse_args())

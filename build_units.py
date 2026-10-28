"""Turn a document corpus into a retrieval-unit corpus in tevatron format ({docid, title, text}).

Units of a document are ids "{docid}#{k}" so rankings can be mapped back to documents.
  doc       one unit per document; keeps the original docid (also usable as claimgen input)
  sentence  sentences, grouped into windows of --size with --stride (size=1: single sentences)
  claim     claims from claimgen output (--claims), or the corpus "statements" field, grouped the same way

Example:
    python build_units.py --corpus eng-docs.jsonl.gz --unit sentence --size 3 --stride 1 --output units/sent3.jsonl.gz
    python build_units.py --corpus pool.jsonl.gz --unit claim --claims pool_claims.jsonl --output units/claim.jsonl.gz
"""
import argparse
import gzip
import json
import os

from claimgen.generate_claims import get_text, open_text, split_sentences


def windows(items, size, stride):
    # slide until a window reaches the end; the last one may be shorter, every item is covered
    out = []
    for i in range(0, len(items), stride):
        out.append(items[i : i + size])
        if i + size >= len(items):
            break
    return out


def load_docids(run_path, topk):
    # restrict to documents in the top-k of a TREC run (e.g. a doc-level baseline)
    docids = set()
    for line in open(run_path):
        qid, _, docid, rank = line.split()[:4]
        if int(rank) <= topk:
            docids.add(docid.split("#")[0])
    return docids


def main(args):
    claims = None
    if args.claims:
        claims = {}
        for line in open_text(args.claims):
            r = json.loads(line)
            claims[r["docid"]] = r["claims"]

    keep = load_docids(args.run, args.topk) if args.run else None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fout = gzip.open(args.output, "wt") if args.output.endswith(".gz") else open(args.output, "w")
    n_docs = n_units = 0
    for line in (l for path in args.corpus for l in open_text(path)):
        doc = json.loads(line)
        docid = doc.get("docid") or doc["id"]
        if keep is not None and docid not in keep:
            continue
        title = doc.get("title") or ""
        if args.unit == "doc":
            units = {docid: get_text(doc)}
        else:
            if args.unit == "sentence":
                items = split_sentences(get_text(doc))
            else:
                items = claims.get(docid, []) if claims is not None else doc.get("statements", [])
            units = {f"{docid}#{k}": " ".join(w) for k, w in enumerate(windows(items, args.size, args.stride))}
        for uid, text in units.items():
            fout.write(json.dumps({"docid": uid, "title": title, "text": text}, ensure_ascii=False) + "\n")
        n_docs += 1
        n_units += len(units)
    fout.close()
    print(f"{n_docs} docs -> {n_units} units ({n_units / max(n_docs, 1):.1f}/doc) -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True, help="jsonl(.gz) files with id|docid, title, text")
    parser.add_argument("--unit", choices=["doc", "sentence", "claim"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=1, help="items (sentences/claims) per unit")
    parser.add_argument("--stride", type=int, default=None, help="default: --size (non-overlapping)")
    parser.add_argument("--claims", default=None, help="claimgen output; default: corpus 'statements' field")
    parser.add_argument("--run", default=None, help="only keep documents retrieved in this TREC run")
    parser.add_argument("--topk", type=int, default=100, help="cutoff for --run")
    args = parser.parse_args()
    args.stride = args.stride or args.size
    main(args)

"""Extract standalone claims from corpus passages with vLLM (offline batch inference).

Example:
    python -m claimgen.generate_claims \
        --corpus data/rag25/rag25_top100_corpus.jsonl \
        --output outputs/rag25_claims.Qwen3-4B.jsonl \
        --model Qwen/Qwen3-4B --limit 200

    # WiCE Claim-Split (Kamoi et al., 2023): split passages into sentences, decompose each sentence
    python -m claimgen.generate_claims --method wice ...
"""
import argparse
import gzip
import itertools
import json
import os
import re

from . import dice_prompts, wice_prompts

CLAIM_RE = re.compile(r"<c>(.*?)</c>", re.DOTALL)
# split after .!? when followed by whitespace and an uppercase letter/digit/opening quote or bracket
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[A-Z0-9])")


def postprocess(text):
    claims = [c.strip() for c in CLAIM_RE.findall(text)]
    return [c for c in claims if c]


def postprocess_wice(text):
    # keep "- " bullet lines; stop at the first non-bullet line after the list
    claims = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            claims.append(line[2:].strip())
        elif claims and line:
            break
    return [c for c in claims if c]


def split_sentences(text):
    sents = []
    for para in text.splitlines():
        sents.extend(s.strip() for s in SENT_RE.split(para))
    return [s for s in sents if s]


def open_text(path):
    # read plain or gzip-compressed jsonl transparently
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def get_text(doc):
    # MS MARCO v2.1 segmented corpus uses "segment" instead of "text"
    return doc.get("text") or doc.get("segment") or ""


def build_messages(doc):
    title = (doc.get("title") or "").strip()
    if title:
        user = dice_prompts.user_prompt_claim.format(title=title, text=get_text(doc))
    else:
        user = dice_prompts.user_prompt_claim_no_title.format(text=get_text(doc))
    return [
        {"role": "system", "content": dice_prompts.system_prompt},
        {"role": "user", "content": user},
    ]


def build_messages_wice(sentence):
    # WiCE uses a plain few-shot prompt, so no system prompt
    return [{"role": "user", "content": wice_prompts.user_prompt_claim.format(sentence=sentence)}]


def run_dice(llm, batch, sampling_params, chat_kwargs):
    outputs = llm.chat([build_messages(d) for d in batch], sampling_params, chat_template_kwargs=chat_kwargs)
    for doc, out in zip(batch, outputs):
        o = out.outputs[0]
        yield {
            "docid": doc["docid"],
            "claims": postprocess(o.text),
            "finish_reason": o.finish_reason,
        }


def run_wice(llm, batch, sampling_params, chat_kwargs):
    # one request per sentence, flattened across the batch, then regrouped per doc
    doc_sents = [split_sentences(get_text(d)) for d in batch]
    flat = [s for sents in doc_sents for s in sents]
    outputs = iter(llm.chat([build_messages_wice(s) for s in flat], sampling_params, chat_template_kwargs=chat_kwargs))
    for doc, sents in zip(batch, doc_sents):
        per_sent = []
        for sent in sents:
            o = next(outputs).outputs[0]
            per_sent.append({"sentence": sent, "claims": postprocess_wice(o.text), "finish_reason": o.finish_reason})
        yield {
            "docid": doc["docid"],
            "claims": [c for s in per_sent for c in s["claims"]],
            # "length" if any sentence was truncated
            "finish_reason": "length" if any(s["finish_reason"] == "length" for s in per_sent) else "stop",
            "sentences": per_sent,
        }


def shard_output_path(path, shard_id, num_shards):
    # outputs/x.jsonl -> outputs/x.shard0of4.jsonl so shards never append to the same file
    if num_shards == 1:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.shard{shard_id}of{num_shards}{ext}"


def main(args):
    assert 0 <= args.shard_id < args.num_shards, "--shard_id must be in [0, --num_shards)"
    args.output = shard_output_path(args.output, args.shard_id, args.num_shards)

    # islice stops reading early when --limit is set (avoids decompressing the whole file)
    with open_text(args.corpus) as fin:
        lines = itertools.islice(fin, args.limit)
        # round-robin sharding over the (limited) corpus: shard k takes lines k, k+N, k+2N, ...
        lines = itertools.islice(lines, args.shard_id, None, args.num_shards)
        docs = [json.loads(line) for line in lines]

    # ragtime1 uses "id" instead of "docid"
    for d in docs:
        d.setdefault("docid", d.get("id"))
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(docs)} docs -> {args.output}")

    # resume: skip docids already in the output file
    done = set()
    if os.path.exists(args.output):
        done = {json.loads(line)["docid"] for line in open(args.output)}
        print(f"Found {len(done)} finished docs in {args.output}")
    docs = [d for d in docs if d["docid"] not in done]
    print(f"{len(docs)} docs to process")
    if not docs:
        return

    import vllm

    llm = vllm.LLM(
        args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        # text-only: skip the vision encoder of multimodal models (e.g. Qwen3.5); harmless for text-only models
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    # Qwen3 recommended non-thinking settings; seed makes reruns reproducible
    sampling_params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        seed=args.seed,
        max_tokens=args.max_tokens,
        # wice: stop if the model starts another few-shot example
        stop=["\nSentence:"] if args.method == "wice" else None,
    )
    run = run_wice if args.method == "wice" else run_dice
    chat_kwargs = {"enable_thinking": args.enable_thinking}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a") as f:
        for start in range(0, len(docs), args.chunk_size):
            batch = docs[start : start + args.chunk_size]
            for record in run(llm, batch, sampling_params, chat_kwargs):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{start + len(batch)}/{len(docs)}] written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dice", "wice"], default="dice",
                        help="dice: passage-level decontextualized claims; wice: WiCE Claim-Split per sentence")
    parser.add_argument("--corpus", default="data/rag25/rag25_top100_corpus.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=10240)
    parser.add_argument("--max_tokens", type=int, default=None, help="default: fill the rest of the context window")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N docs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true", help="for hybrid-thinking models (Qwen3/3.5)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="docs per llm.chat call (checkpoint interval)")
    parser.add_argument("--num_shards", type=int, default=1, help="split the corpus into N round-robin shards")
    parser.add_argument("--shard_id", type=int, default=0, help="which shard to process, in [0, num_shards)")
    main(parser.parse_args())

"""Collect the distribution of the model's first *meaningful* response token.

For a SciKnowEval-style dataset the model is asked to answer as::

    <reasoning>
    ...
    </reasoning>
    <answer>
    A
    </answer>

so response position 0 is a structural scaffold token, not content. This script
generates short continuations and records the first token *after* that scaffold,
i.e. the token the token-roll branch should be forcing.

Output is written in the same schema as the existing pools under ``datasets/``
so it can be passed straight to ``actor.token_roll.token_pool_path``.
"""

import argparse
import json
import os
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B")
    p.add_argument(
        "--data_path",
        default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/sdpo/SDPO/datasets/sciknoweval/chemistry/train.json",
    )
    p.add_argument("--out_path", default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_tokens_chemistry_model.json")
    p.add_argument("--scaffold", default="<reasoning>\n", help="format prefix to skip before the first meaningful token")
    p.add_argument("--n", type=int, default=8, help="samples per prompt (matches rollout.n)")
    p.add_argument("--max_tokens", type=int, default=24, help="only the head of the response is needed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="cap number of prompts (0 = all)")
    p.add_argument("--tensor_parallel_size", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return p.parse_args()


def load_records(path):
    """The dataset files are JSONL-ish; tolerate both a JSON array and one-per-line."""
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def is_structural(text: str) -> bool:
    """Whitespace or scaffold punctuation carries no semantics."""
    stripped = text.strip()
    return stripped == "" or all(ch in "<>/\\|-=*#`\"'.,:;!?()[]{}" for ch in stripped)


def first_meaningful_token(token_ids, tokenizer, scaffold_ids):
    """Return the first token id after the scaffold, plus whether the scaffold matched."""
    if scaffold_ids and token_ids[: len(scaffold_ids)] == scaffold_ids:
        rest = token_ids[len(scaffold_ids) :]
        matched = True
    else:
        rest = token_ids
        matched = False

    for tid in rest:
        if not is_structural(tokenizer.decode([tid])):
            return tid, matched
    return None, matched


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    scaffold_ids = tokenizer.encode(args.scaffold, add_special_tokens=False) if args.scaffold else []
    print(f"scaffold {args.scaffold!r} -> {scaffold_ids}", flush=True)

    records = load_records(args.data_path)
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} prompts from {args.data_path}", flush=True)

    prompts = []
    for rec in records:
        messages = []
        if rec.get("system"):
            messages.append({"role": "system", "content": rec["system"]})
        messages.append({"role": "user", "content": rec["prompt"]})
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
    )
    sampling = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    outputs = llm.generate(prompts, sampling)

    counter = Counter()
    total = skipped = scaffold_hits = 0
    scaffold_examples = Counter()

    for out in outputs:
        for cand in out.outputs:
            total += 1
            ids = list(cand.token_ids)
            scaffold_examples[tokenizer.decode(ids[: len(scaffold_ids)] or ids[:4])] += 1
            tid, matched = first_meaningful_token(ids, tokenizer, scaffold_ids)
            if matched:
                scaffold_hits += 1
            if tid is None:
                skipped += 1
                continue
            counter[tid] += 1

    tokens = [
        {"token_id": tid, "token_text": tokenizer.decode([tid]), "count": cnt}
        for tid, cnt in counter.most_common()
    ]
    payload = {
        "total_solutions": total,
        "skipped": skipped,
        "unique_tokens": len(tokens),
        "scaffold": args.scaffold,
        "scaffold_match_rate": round(scaffold_hits / max(total, 1), 4),
        "tokens": tokens,
    }
    with open(args.out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n=== wrote {args.out_path} ===")
    print(f"generations: {total}, skipped: {skipped}, unique first-meaningful tokens: {len(tokens)}")
    print(f"scaffold match rate: {payload['scaffold_match_rate']:.2%}")
    print("\ntop-25 first meaningful tokens:")
    for t in tokens[:25]:
        print(f"  {t['count']:6d}  id={t['token_id']:<7} {t['token_text']!r}")
    print("\nmost common response heads:")
    for head, cnt in scaffold_examples.most_common(5):
        print(f"  {cnt:6d}  {head!r}")


if __name__ == "__main__":
    main()

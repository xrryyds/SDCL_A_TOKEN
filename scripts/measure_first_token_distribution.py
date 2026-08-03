"""Measure the model's first *meaningful* token distribution on a dataset.

Unlike ``collect_first_meaningful_tokens.py`` (which samples and counts), this
reads the actual next-token distribution at the position the token-roll branch
forces, i.e. right after the format scaffold. It answers:

* how sharp is that distribution (true full-vocab entropy, rank-k probabilities)?
* how far off-distribution are the pool tokens (mean -log p, comparable to the
  observed ``token_roll/ce_loss``)?
* how much headroom is there for a soft-target beta?

A single forward pass per prompt is all that is needed, so this uses plain
transformers rather than an inference engine: the logits at the last position of
``chat_template(prompt) + scaffold`` *are* the distribution of interest, and the
full vocabulary is available (exact entropy, exact pool mass).
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B")
    p.add_argument(
        "--data_path",
        default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/sdpo/SDPO/datasets/sciknoweval/chemistry/train.json",
    )
    p.add_argument("--pool_path", default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_tokens_test.json")
    p.add_argument(
        "--out_path",
        default="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_distribution_chemistry.json",
    )
    p.add_argument("--scaffold", default="<reasoning>\n")
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def load_records(path):
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    scaffold_ids = tok.encode(args.scaffold, add_special_tokens=False) if args.scaffold else []
    print(f"scaffold {args.scaffold!r} -> {scaffold_ids}", flush=True)

    pool = json.load(open(args.pool_path))
    pool_ids = sorted({t["token_id"] for t in (pool["tokens"] if isinstance(pool, dict) else pool)})
    print(f"pool {os.path.basename(args.pool_path)}: {len(pool_ids)} tokens", flush=True)

    records = load_records(args.data_path)
    if args.limit:
        records = records[: args.limit]
    print(f"loaded {len(records)} prompts", flush=True)

    seqs = []
    for rec in records:
        messages = []
        if rec.get("system"):
            messages.append({"role": "system", "content": rec["system"]})
        messages.append({"role": "user", "content": rec["prompt"]})
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        seqs.append(list(ids) + scaffold_ids)

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    pool_idx = torch.tensor(pool_ids, device=model.device)

    entropies, top1_prob, pool_mass, pool_best, pool_mean_nll = [], [], [], [], []
    rank_probs = defaultdict(list)
    top1_counter = Counter()
    per_token_prob = defaultdict(list)

    n_done = 0
    for start in range(0, len(seqs), args.batch_size):
        chunk = seqs[start : start + args.batch_size]
        maxlen = max(len(s) for s in chunk)
        input_ids = torch.full((len(chunk), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, s in enumerate(chunk):  # left padding keeps the last position real
            input_ids[i, maxlen - len(s) :] = torch.tensor(s, dtype=torch.long)
            attn[i, maxlen - len(s) :] = 1
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()

        ent = -(probs * logprobs).sum(dim=-1)                       # exact, full vocab
        pmass = probs.index_select(1, pool_idx).sum(dim=-1)
        pbest = probs.index_select(1, pool_idx).max(dim=-1).values
        pnll = -logprobs.index_select(1, pool_idx).mean(dim=-1)     # mean -log p over pool
        topv, topi = probs.topk(args.topk, dim=-1)

        for b in range(len(chunk)):
            entropies.append(ent[b].item())
            pool_mass.append(pmass[b].item())
            pool_best.append(pbest[b].item())
            pool_mean_nll.append(pnll[b].item())
            ids_b, ps_b = topi[b].tolist(), topv[b].tolist()
            for rank, (tid, p) in enumerate(zip(ids_b, ps_b), start=1):
                rank_probs[rank].append(p)
                per_token_prob[tid].append(p)
            top1_counter[ids_b[0]] += 1
            top1_prob.append(ps_b[0])

        n_done += len(chunk)
        if n_done % (args.batch_size * 20) == 0 or n_done == len(seqs):
            print(f"  {n_done}/{len(seqs)}", flush=True)

    n = len(entropies)
    report = {
        "n_prompts": n,
        "scaffold": args.scaffold,
        "topk": args.topk,
        "entropy_mean": sum(entropies) / n,
        "entropy_p50": pct(entropies, 0.5),
        "entropy_p90": pct(entropies, 0.9),
        "top1_prob_mean": sum(top1_prob) / n,
        "rank_prob_mean": {r: sum(v) / len(v) for r, v in sorted(rank_probs.items())},
        "pool_mass_mean": sum(pool_mass) / n,
        "pool_mass_p50": pct(pool_mass, 0.5),
        "pool_best_mean": sum(pool_best) / n,
        "pool_mean_nll": sum(pool_mean_nll) / n,
        "top1_tokens": [
            {
                "token_id": tid,
                "token_text": tok.decode([tid]),
                "argmax_count": c,
                "prob_mean": sum(per_token_prob[tid]) / len(per_token_prob[tid]),
            }
            for tid, c in top1_counter.most_common(20)
        ],
        "frequent_tokens": [
            {
                "token_id": tid,
                "token_text": tok.decode([tid]),
                "appear_in_topk": len(v),
                "prob_mean": sum(v) / len(v),
            }
            for tid, v in sorted(per_token_prob.items(), key=lambda kv: -len(kv[1]))[:40]
        ],
    }
    with open(args.out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== wrote {args.out_path} ===")
    print(f"prompts measured: {n}")
    print(
        f"first-meaningful-token entropy (full vocab): mean {report['entropy_mean']:.4f} nats "
        f"(p50 {report['entropy_p50']:.4f}, p90 {report['entropy_p90']:.4f})"
    )
    print(f"top-1 prob mean: {report['top1_prob_mean']:.4f}")

    print("\nrank -> mean prob (sharpness profile):")
    for r in [1, 2, 3, 5, 10, 20, 30]:
        if r in report["rank_prob_mean"]:
            p = report["rank_prob_mean"][r]
            nll = -math.log(p) if p > 0 else float("inf")
            print(f"  rank {r:>2}: {p:.3e}   (-log p = {nll:.2f})")

    print(f"\nMATH pool ({os.path.basename(args.pool_path)}) at this position:")
    print(f"  total prob mass  mean {report['pool_mass_mean']:.3e}   p50 {report['pool_mass_p50']:.3e}")
    print(f"  best pool token  mean {report['pool_best_mean']:.3e}")
    print(f"  mean -log p over pool tokens: {report['pool_mean_nll']:.2f}   (compare token_roll/ce_loss)")

    print("\nmost frequent argmax tokens:")
    for t in report["top1_tokens"][:10]:
        print(f"  {t['argmax_count']:6d}x  p̄={t['prob_mean']:.4f}  id={t['token_id']:<7} {t['token_text']!r}")

    print("\ntokens most often in top-k (diversity headroom):")
    for t in report["frequent_tokens"][:20]:
        print(f"  seen {t['appear_in_topk']:6d}  p̄={t['prob_mean']:.3e}  id={t['token_id']:<7} {t['token_text']!r}")


if __name__ == "__main__":
    main()

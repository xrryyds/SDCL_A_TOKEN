"""Measure the actual first-token logit gap: log p_top1 - log p_forced, per prompt.

This is the quantity the FILL branch's first-token term uses. It must be computed
per prompt and then averaged; averaging probabilities first and taking logs after
gives a different (wrong) number.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"
DATA = "/home/xiongrengrong.xrr/SDCL_A_TOKEN/sdpo/SDPO/datasets/sciknoweval/chemistry/train.json"
POOL = "/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_chemistry_8.json"
SCAFFOLD = "<reasoning>\n"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def load_records(path):
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    pool = json.load(open(POOL))["tokens"]
    pool_ids = [t["token_id"] for t in pool]
    pool_txt = [t["token_text"] for t in pool]

    scaffold_ids = tok.encode(SCAFFOLD, add_special_tokens=False)
    records = load_records(DATA)[:LIMIT]

    seqs = []
    for rec in records:
        msgs = []
        if rec.get("system"):
            msgs.append({"role": "system", "content": rec["system"]})
        msgs.append({"role": "user", "content": rec["prompt"]})
        ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        seqs.append(list(ids) + scaffold_ids)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    gaps = {t: [] for t in pool_txt}
    top1_nll = []
    bs = 8
    for start in range(0, len(seqs), bs):
        chunk = seqs[start : start + bs]
        maxlen = max(len(s) for s in chunk)
        input_ids = torch.full((len(chunk), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, s in enumerate(chunk):
            input_ids[i, maxlen - len(s) :] = torch.tensor(s, dtype=torch.long)
            attn[i, maxlen - len(s) :] = 1
        input_ids, attn = input_ids.to(model.device), attn.to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        top1 = logp.max(dim=-1).values
        top1_nll.extend((-top1).tolist())
        for tid, txt in zip(pool_ids, pool_txt):
            gaps[txt].extend((top1 - logp[:, tid]).tolist())

    n = len(top1_nll)
    print(f"\n=== per-prompt first-token gap, n={n} prompts (chemistry, base Qwen3-8B) ===")
    print(f"mean(-log p_top1) = {sum(top1_nll)/n:.3f} nats\n")
    print(f"{'slot':>4} {'token':10s} {'mean':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'zero%':>6}")
    for i, txt in enumerate(pool_txt, 1):
        v = sorted(gaps[txt])
        m = sum(v) / len(v)
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        zero = 100.0 * sum(1 for x in v if x < 1e-6) / len(v)
        print(f"{i:>4} {txt:10s} {m:7.2f} {q(0.1):7.2f} {q(0.5):7.2f} {q(0.9):7.2f} {zero:5.1f}%")


if __name__ == "__main__":
    main()

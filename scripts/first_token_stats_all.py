"""统计 LoRA 在 eval 输出的所有数据集上的首 token 分布 (greedy).

复用 first_token_stats.py 的逻辑, 但一次扫描多个数据集 (corr / roll / pool / math500 / math_test).
对比 LoRA 在不同分布上的首 token 偏好坍缩情况.

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/first_token_stats_all.py \
    --eval_dir output/eval_v3_<ts> \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stats(items: List[dict], tokenizer) -> Counter:
    cnt: Counter = Counter()
    for it in items:
        pred = it.get("pred_raw", "") or ""
        if not pred:
            tid = -1
        else:
            ids = tokenizer.encode(pred, add_special_tokens=False)
            tid = int(ids[0]) if ids else -1
        cnt[tid] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=15)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    datasets = ["corr", "roll", "pool", "math500", "math_test"]

    print(f"\n{'=' * 100}")
    print(f" LoRA 首 token 分布 (eval_dir={args.eval_dir})")
    print(f"{'=' * 100}")
    print(f" {'Dataset':<12} {'n':>5} {'unique':>7} {'top1 pct':>9} {'top1 token':<20} {'entropy':>9}")
    print(" " + "-" * 96)

    all_results = {}

    for ds in datasets:
        path = os.path.join(args.eval_dir, f"{ds}_lora_greedy.jsonl")
        if not os.path.isfile(path):
            print(f" {ds:<12} {'(missing)':>30}")
            continue
        items = load_jsonl(path)
        cnt = stats(items, tok)
        n_total = len(items)
        unique = len(cnt)
        top1_tid, top1_count = cnt.most_common(1)[0]
        top1_pct = top1_count / n_total * 100
        top1_text = tok.decode([top1_tid]) if top1_tid >= 0 else "<EMPTY>"
        # entropy
        probs = [c / n_total for c in cnt.values()]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        all_results[ds] = (cnt, n_total, unique, top1_text, top1_pct, ent)
        print(
            f" {ds:<12} {n_total:>5} {unique:>7} {top1_pct:>8.2f}% {repr(top1_text):<20} {ent:>9.4f}"
        )

    # 各数据集 Top-N 详情
    print()
    for ds in datasets:
        if ds not in all_results:
            continue
        cnt, n_total, unique, _, _, _ = all_results[ds]
        print(f"\n {ds.upper()}  (n={n_total}, unique={unique})  Top-{args.top_n}:")
        print(f"   {'rank':>4} {'tid':>7} {'count':>5} {'pct':>7}  token_text")
        print("   " + "-" * 58)
        for i, (tid, c) in enumerate(cnt.most_common(args.top_n), 1):
            text = tok.decode([tid]) if tid >= 0 else "<EMPTY>"
            print(f"   {i:>4} {tid:>7} {c:>5} {c/n_total*100:>6.2f}%  {text!r}")
        if unique > args.top_n:
            rest = sum(c for _, c in cnt.most_common()[args.top_n:])
            print(f"   {'...':>4} {'rest':>7} {rest:>5} {rest/n_total*100:>6.2f}%  ({unique - args.top_n} 个其他 token)")
    print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()

"""first_token_distribution_full_math.py — 全量 MATH train 上每题首 token top-5 汇总.

跑 stage2 merged 模型在 MATH train (~7500 题) 的原始 prompt 末尾, 取每题首 token
位置的 top-5 token, 汇总:
  - top-1 token 全集频率 (Counter)
  - top-5 里出现的 token 全集频率 (Counter)
  - top-1 平均概率
  - top-1 概率分布 (min / p25 / median / p75 / max)
  - 多少题 top-1 是 'Okay' / 'Alright' (R1-Distill 默认起手词)

单卡 vLLM, batch 推理, max_tokens=1, logprobs=5.

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/first_token_distribution_full_math.py \
    --model_path output/sdft_v3_stage2_merged \
    --output_dir output/first_token_dist_stage2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                    help="模型目录 (Base 或 merged LoRA 都行)")
    ap.add_argument("--n_questions", type=int, default=-1,
                    help="-1 = 全量 (~7500), 正数 = 限抽前 N 题")
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="vLLM 总窗口预算 (实际只生 1 token)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载 MATH train 全量 ----
    from data_math.MATH_util import Math_All
    print(f"[load] Math_All(train=True, subset='all') ...", flush=True)
    data = Math_All(train=True, subset_name="all", shuffle=False)
    problems = data.problems
    if args.n_questions > 0 and args.n_questions < len(problems):
        problems = problems[: args.n_questions]
    print(f"[load] n_questions = {len(problems)}", flush=True)

    # ---- tokenizer + 构造 prompts ----
    from transformers import AutoTokenizer
    print(f"[load] tokenizer ← {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = []
    for q in problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(q)},
        ]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(prompt)
    print(f"[setup] total prompts = {len(prompts)}", flush=True)
    print(f"[setup] prompt[0] tail (last 200 chars):\n  {prompts[0][-200:]!r}",
          flush=True)

    # ---- vLLM 推理 ----
    from vllm import LLM, SamplingParams
    print(f"[vllm] init engine (TP={args.tensor_parallel_size}) ...", flush=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=args.top_k,
        seed=args.seed,
    )
    print(f"[vllm] generate {len(prompts)} prompts (max_tokens=1, top_k={args.top_k}) ...",
          flush=True)
    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)

    # ---- 汇总 ----
    top1_counter: Counter = Counter()      # text -> count
    top5_counter: Counter = Counter()      # text -> count (top-5 出现次数, 1 题最多 +5)
    top5_in_top5_counter: Counter = Counter()  # 每题去重后 top-5 出现的 token (跟 top5_counter 一样, 但每题只算一次)
    top1_probs = []
    n_total = 0
    n_okay = 0
    n_alright = 0
    n_okay_or_alright = 0

    for o in outs:
        out0 = o.outputs[0]
        if (not out0.token_ids
                or out0.logprobs is None
                or len(out0.logprobs) == 0):
            continue
        # logprobs[0] 就是首 token 位置 dict {tid: Logprob}
        lp_dict = out0.logprobs[0]
        entries = []
        for tid, lp_obj in lp_dict.items():
            lp = float(getattr(lp_obj, "logprob", lp_obj))
            entries.append((int(tid), lp))
        entries.sort(key=lambda x: -x[1])
        topk = entries[: args.top_k]
        if not topk:
            continue

        n_total += 1

        # top-1
        top1_tid, top1_lp = topk[0]
        top1_text = tok.decode([top1_tid])
        top1_prob = math.exp(top1_lp)
        top1_counter[top1_text] += 1
        top1_probs.append(top1_prob)

        if top1_text == "Okay":
            n_okay += 1
        if top1_text == "Alright":
            n_alright += 1
        if top1_text in ("Okay", "Alright"):
            n_okay_or_alright += 1

        # top-5 (含 top-1)
        seen_in_this_q = set()
        for tid, lp in topk:
            txt = tok.decode([tid])
            if txt not in seen_in_this_q:
                top5_counter[txt] += 1
                seen_in_this_q.add(txt)

    # ---- 打印 ----
    top1_probs.sort()
    n = len(top1_probs)
    if n == 0:
        print("[error] no valid outputs", flush=True)
        return

    p25 = top1_probs[n // 4]
    p50 = top1_probs[n // 2]
    p75 = top1_probs[3 * n // 4]
    avg_top1 = sum(top1_probs) / n

    print(flush=True)
    print("=" * 90, flush=True)
    print(f" 全量 MATH train 首 token top-{args.top_k} 汇总  (n={n_total} 题)", flush=True)
    print("=" * 90, flush=True)
    print(f" top-1 概率: avg={avg_top1:.4f}  p25={p25:.4f}  median={p50:.4f}  p75={p75:.4f}",
          flush=True)
    print(f" top-1 == 'Okay'      : {n_okay}/{n_total} ({100*n_okay/n_total:.2f}%)", flush=True)
    print(f" top-1 == 'Alright'   : {n_alright}/{n_total} ({100*n_alright/n_total:.2f}%)", flush=True)
    print(f" top-1 in (Okay,Alright): {n_okay_or_alright}/{n_total} "
          f"({100*n_okay_or_alright/n_total:.2f}%)", flush=True)
    print(flush=True)

    print(f" Top-30 top-1 token 分布:", flush=True)
    print(f" {'token':<24} {'count':>8} {'pct':>8}", flush=True)
    print(" " + "-" * 44, flush=True)
    for txt, cnt in top1_counter.most_common(30):
        print(f" {txt!r:<24} {cnt:>8} {100*cnt/n_total:>7.2f}%", flush=True)
    print(flush=True)

    print(f" Top-30 出现在 top-{args.top_k} 的 token 分布 (按出现题数, 1 题 1 token 只算 1 次):",
          flush=True)
    print(f" {'token':<24} {'in_top5':>8} {'pct':>8}", flush=True)
    print(" " + "-" * 44, flush=True)
    for txt, cnt in top5_counter.most_common(30):
        print(f" {txt!r:<24} {cnt:>8} {100*cnt/n_total:>7.2f}%", flush=True)
    print("=" * 90, flush=True)

    # ---- 落盘 summary ----
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_path": args.model_path,
            "n_questions": n_total,
            "top_k": args.top_k,
            "avg_top1_prob": avg_top1,
            "top1_prob_p25": p25,
            "top1_prob_median": p50,
            "top1_prob_p75": p75,
            "top1_okay_count": n_okay,
            "top1_okay_pct": 100 * n_okay / n_total,
            "top1_alright_count": n_alright,
            "top1_alright_pct": 100 * n_alright / n_total,
            "top1_okay_or_alright_count": n_okay_or_alright,
            "top1_okay_or_alright_pct": 100 * n_okay_or_alright / n_total,
            "top1_top30": [
                {"text": t, "count": c, "pct": 100 * c / n_total}
                for t, c in top1_counter.most_common(30)
            ],
            "top5_top30": [
                {"text": t, "in_top5_count": c, "pct": 100 * c / n_total}
                for t, c in top5_counter.most_common(30)
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()

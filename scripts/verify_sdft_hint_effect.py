"""验证 SDFT hint prompt 是否真的让 Base 模型听话.

实验:
  对 MATH train 题, 给 Base 模型 (DS-R1-Distill-Qwen-7B) 加 hint:
    user content = f'Please start your answer with "{fill_token_text}". {question}'

  其中 fill_token_text 从 376 候选首 token 中随机/轮转抽取。
  生成 1 个 token, 看是不是真的等于 fill_token_text。

  对照: 同一道题, 不加 hint, 看默认首 token。

关键问题:
  - 加 hint 后, 第一个 token 多少比例真的等于 hint 指定的 token?
  - 不同 fill_token (We / Let / By / Since) 听话率有差异吗?
  - 不同 fill_token 之间, top1 不一致还是都是 hint?

用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/verify_sdft_hint_effect.py \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from typing import List

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument(
        "--pool_token_path", type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "first_tokens_test.json"),
    )
    ap.add_argument("--n_questions", type=int, default=300,
                    help="抽多少题 (每题用一个 hint token, 不必跑全集)")
    ap.add_argument("--n_top_hint_tokens", type=int, default=20,
                    help="只用前 N 个候选 hint token (跑全 376 慢, 用 N 个看趋势就够)")
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="vLLM 总窗口预算 (实际只生 1 token)")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "output", "verify_sdft_hint"),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # ----- 1) 加载 pool token + MATH train -----
    print(f"[load] pool tokens ← {args.pool_token_path}")
    with open(args.pool_token_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cand = d["tokens"] if isinstance(d, dict) else d
    # 取前 N 个候选 (按原列表顺序, 前面的是高频)
    candidates = cand[: args.n_top_hint_tokens]
    print(f"[load] using {len(candidates)} hint tokens, first 10 = "
          f"{[c['token_text'] for c in candidates[:10]]}")

    from data_math.MATH_util import Math_All
    print(f"[load] Math_All(train=True) ...")
    data = Math_All(train=True, subset_name="all")
    problems = data.problems[: args.n_questions]
    print(f"[load] n_questions = {len(problems)}")

    # ----- 2) tokenizer + 应用 chat template -----
    from transformers import AutoTokenizer
    print(f"[load] tokenizer ← {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

    # 构造 prompts: (hint_text, question) 对的笛卡尔积或轮转
    # 简化: 每个 hint token 配 n_questions 题, 共 n_top_hint * n_questions
    # 用户要求 n_questions=300, n_top=20 → 6000 个 prompt, 还可以
    all_prompts: List[str] = []
    all_meta: List[dict] = []  # {hint_tid, hint_text, q_idx}

    for hi, c_obj in enumerate(candidates):
        hint_tid = int(c_obj["token_id"])
        hint_text = c_obj["token_text"]
        for qi, q in enumerate(problems):
            user_content = f'Please start your answer with "{hint_text}". {q}'
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            all_prompts.append(prompt)
            all_meta.append({
                "hint_idx": hi,
                "hint_tid": hint_tid,
                "hint_text": hint_text,
                "q_idx": qi,
            })

    print(f"[setup] total prompts = {len(all_prompts)} "
          f"({len(candidates)} hints × {len(problems)} questions)")

    # 关键: 打印前 3 条 prompt 末尾, 确认 hint 真的拼上了
    print()
    print("=" * 90)
    print(" 前 3 条 prompt 末尾 200 字符 (验证 hint 是否真的拼进去):")
    print("=" * 90)
    for i in range(min(3, len(all_prompts))):
        meta = all_meta[i]
        print(f"\n[{i}] hint={meta['hint_text']!r} q_idx={meta['q_idx']}")
        print("    prompt tail (last 250 chars):")
        print("    " + repr(all_prompts[i][-250:]))
    print("=" * 90)
    print()

    # ----- 3) vLLM 跑首 token -----
    print(f"\n[vllm] init engine (TP={args.tensor_parallel_size}) ...")
    from vllm import LLM, SamplingParams
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
        logprobs=5,   # 返回 top-5 token 的 logprob
        seed=args.seed,
    )
    print(f"[vllm] generate {len(all_prompts)} prompts (max_tokens=1) ...")
    outs = llm.generate(all_prompts, sampling_params=sp, use_tqdm=True)

    # ----- 4) 分析: 加 hint 后首 token 是不是真的 = hint_tid + top-5 -----
    per_hint = {}  # hint_text -> {match: int, total: int, actual_top: Counter,
                   #               hint_in_top5: int, sum_hint_prob: float, sum_top1_prob: float}
    rows = []
    for meta, o in zip(all_meta, outs):
        out0 = o.outputs[0]
        if len(out0.token_ids) == 0 or out0.logprobs is None or len(out0.logprobs) == 0:
            actual_tid = -1
            actual_text = "<EMPTY>"
            top5 = []
            hint_rank = -1
            hint_prob = 0.0
            top1_prob = 0.0
        else:
            actual_tid = int(out0.token_ids[0])
            actual_text = tok.decode([actual_tid])
            # logprobs[0] 是第 1 个 token 位置的 dict {token_id: Logprob 对象}
            lp_dict = out0.logprobs[0]
            # 解析 top-5 (按 logprob 降序)
            entries = []
            for tid, lp_obj in lp_dict.items():
                lp = float(getattr(lp_obj, "logprob", lp_obj))
                entries.append((int(tid), lp))
            entries.sort(key=lambda x: -x[1])
            top5 = []
            for tid, lp in entries[:5]:
                ttext = tok.decode([tid])
                top5.append({
                    "tid": tid,
                    "text": ttext,
                    "logp": lp,
                    "prob": math.exp(lp),
                })
            top1_prob = top5[0]["prob"] if top5 else 0.0
            # hint 在 top-5 里的排名
            hint_rank = -1
            hint_prob = 0.0
            for r_idx, e in enumerate(top5):
                if e["tid"] == meta["hint_tid"]:
                    hint_rank = r_idx + 1
                    hint_prob = e["prob"]
                    break

        match = (actual_tid == meta["hint_tid"])
        rows.append({
            "hint_text": meta["hint_text"],
            "hint_tid": meta["hint_tid"],
            "q_idx": meta["q_idx"],
            "actual_tid": actual_tid,
            "actual_text": actual_text,
            "match": match,
            "top5": top5,
            "hint_rank": hint_rank,        # hint 在 top-5 排名 (1=top1, -1=不在 top5)
            "hint_prob": hint_prob,
            "top1_prob": top1_prob,
        })
        h = meta["hint_text"]
        if h not in per_hint:
            per_hint[h] = {
                "match": 0, "total": 0,
                "actual_top": Counter(),
                "hint_in_top5": 0,
                "sum_hint_prob": 0.0,
                "sum_top1_prob": 0.0,
            }
        per_hint[h]["total"] += 1
        if match:
            per_hint[h]["match"] += 1
        per_hint[h]["actual_top"][actual_text] += 1
        if hint_rank > 0:
            per_hint[h]["hint_in_top5"] += 1
        per_hint[h]["sum_hint_prob"] += hint_prob
        per_hint[h]["sum_top1_prob"] += top1_prob

    # ----- 5) 落盘 + 打印 -----
    rows_path = os.path.join(args.output_dir, "rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[write] 逐条 → {rows_path}")

    # 终端汇总
    print()
    print("=" * 116)
    print(f" Base hint 听话率 (n_questions={len(problems)}, n_hint={len(candidates)})")
    print("=" * 116)
    print(f" {'hint':<14} {'match':>6} {'total':>6} {'听话率':>8} "
          f"{'hint在top5':>10} {'avg hint_p':>11} {'avg top1_p':>11} "
          f"{'top1 实际 (非 hint)':<30}")
    print(" " + "-" * 112)

    sorted_hints = sorted(
        per_hint.items(),
        key=lambda kv: -(kv[1]["match"] / max(kv[1]["total"], 1))
    )
    overall_match = 0
    overall_total = 0
    overall_hint_in_top5 = 0
    overall_sum_hint_prob = 0.0
    overall_sum_top1_prob = 0.0
    for hint_text, stats in sorted_hints:
        m, t = stats["match"], stats["total"]
        pct = m / t * 100 if t > 0 else 0
        in_top5_pct = stats["hint_in_top5"] / max(t, 1) * 100
        avg_hint_prob = stats["sum_hint_prob"] / max(t, 1)
        avg_top1_prob = stats["sum_top1_prob"] / max(t, 1)
        overall_match += m
        overall_total += t
        overall_hint_in_top5 += stats["hint_in_top5"]
        overall_sum_hint_prob += stats["sum_hint_prob"]
        overall_sum_top1_prob += stats["sum_top1_prob"]
        non_hint = [(k, v) for k, v in stats["actual_top"].most_common() if k != hint_text]
        if non_hint:
            top_other = ", ".join(f"{k!r}={v}" for k, v in non_hint[:2])
        else:
            top_other = "(all hint)"
        print(f" {hint_text!r:<14} {m:>6} {t:>6} {pct:>7.2f}% "
              f"{in_top5_pct:>9.2f}% {avg_hint_prob:>11.4f} {avg_top1_prob:>11.4f} "
              f"{top_other:<30}")

    print(" " + "-" * 112)
    overall_pct = overall_match / overall_total * 100 if overall_total > 0 else 0
    overall_in_top5_pct = overall_hint_in_top5 / max(overall_total, 1) * 100
    overall_avg_hint_prob = overall_sum_hint_prob / max(overall_total, 1)
    overall_avg_top1_prob = overall_sum_top1_prob / max(overall_total, 1)
    print(f" {'TOTAL':<14} {overall_match:>6} {overall_total:>6} {overall_pct:>7.2f}% "
          f"{overall_in_top5_pct:>9.2f}% {overall_avg_hint_prob:>11.4f} "
          f"{overall_avg_top1_prob:>11.4f}")
    print("=" * 116)

    # 抽样打印几条 top-5 详情
    print()
    print(" 抽样 top-5 详情 (前 5 条):")
    for r in rows[:5]:
        print(f"   hint={r['hint_text']!r:<10} actual={r['actual_text']!r} "
              f"match={r['match']} hint_rank={r['hint_rank']} hint_prob={r['hint_prob']:.4f}")
        for e in r["top5"]:
            mark = " ← hint" if e["tid"] == r["hint_tid"] else ""
            print(f"     {e['text']!r:<14} prob={e['prob']:.4f} logp={e['logp']:.3f}{mark}")
    print()

    # ⭐ 完整 input + output 5 条样例 (核查 hint 是否真的喂给模型, 5 个不同 hint)
    print()
    print("=" * 90)
    print(" 5 条 完整 prompt + 输出样例 (核查 hint 是否真的喂给模型)")
    print("=" * 90)
    seen_hints = set()
    sample_indices = []
    for i, r in enumerate(rows):
        if r["hint_text"] not in seen_hints:
            sample_indices.append(i)
            seen_hints.add(r["hint_text"])
        if len(sample_indices) >= 5:
            break
    for sample_idx in sample_indices:
        r = rows[sample_idx]
        print(f"\n--- 样例 {sample_idx}: hint={r['hint_text']!r} (tid={r['hint_tid']}) ---")
        print(f"完整 prompt:")
        print(repr(all_prompts[sample_idx]))
        print(f"\n实际生成首 token: {r['actual_text']!r} (tid={r['actual_tid']})  "
              f"match={r['match']}  hint_rank={r['hint_rank']}  "
              f"hint_prob={r['hint_prob']:.4f}")
        print(f"top-5 概率分布:")
        for e in r["top5"]:
            mark = " ← hint" if e["tid"] == r["hint_tid"] else ""
            print(f"  {e['text']!r:<16} prob={e['prob']:.4f} logp={e['logp']:.3f}{mark}")
    print()
    print("=" * 90)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_questions": len(problems),
            "n_hint_tokens": len(candidates),
            "overall_match": overall_match,
            "overall_total": overall_total,
            "overall_pct": overall_pct,
            "overall_hint_in_top5_pct": overall_in_top5_pct,
            "overall_avg_hint_prob": overall_avg_hint_prob,
            "overall_avg_top1_prob": overall_avg_top1_prob,
            "per_hint": {
                h: {
                    "match": s["match"],
                    "total": s["total"],
                    "match_pct": s["match"] / s["total"] * 100 if s["total"] > 0 else 0,
                    "hint_in_top5": s["hint_in_top5"],
                    "hint_in_top5_pct": s["hint_in_top5"] / s["total"] * 100 if s["total"] > 0 else 0,
                    "avg_hint_prob": s["sum_hint_prob"] / s["total"] if s["total"] > 0 else 0,
                    "avg_top1_prob": s["sum_top1_prob"] / s["total"] if s["total"] > 0 else 0,
                    "actual_top": dict(s["actual_top"].most_common(10)),
                }
                for h, s in per_hint.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] summary → {summary_path}")


if __name__ == "__main__":
    main()

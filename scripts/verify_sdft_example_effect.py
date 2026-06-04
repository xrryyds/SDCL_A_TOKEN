"""验证标准 SDFT 的 teacher_prompt 是否能让 Base 生成 fill 开头的答案。

背景:
  标准 SDFT (Self-Distillation/distil_trainer.py) 的 teacher_prompt 形态是 main.py:30-37
  的 Template —— 原题 + 塞一段示例答案 + "现在你自己写一个"。

  本脚本把采集到的 fill_answer (Base 喂 fill 首 token 后救回的正确解) 作为示例塞进
  teacher_prompt, 看 teacher (= Base) 自己生成的答案:
    1. 首 token 是否 == 该示例的 fill_token (听话率)
    2. 完整答案是否做对 (boxed == ref_answer)
  并和裸 prompt (不塞示例) 对照, 看示例带来的增量。

  若 teacher 会照 fill 首 token 起头 → 在线 SDFT 信号方向对, 路线可行。
  若仍用默认首 token → teacher 没被示例带偏, 在线 SDFT 学不到 fill 控制, 训练前否掉。

用法 (远程 4 卡):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/verify_sdft_example_effect.py \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
    --n_questions 200 --n_cand_per_q 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from collections import Counter
from string import Template
from typing import Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.data_utils import extract_boxed_content, normalize_answer  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# main.py:30-37 原 Template (teacher 看示例作答)
TEACHER_TEMPLATE = Template("""$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


def _is_correct(generated_text: str, ref_answer: str) -> bool:
    pred = extract_boxed_content(generated_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(str(ref_answer))


def _apply_chat(tok, user_content: str, enable_thinking: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking == "true":
        chat_kwargs["enable_thinking"] = True
    elif enable_thinking == "false":
        chat_kwargs["enable_thinking"] = False
    try:
        return tok.apply_chat_template(msgs, **chat_kwargs)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def _first_token_top5(out0, tok, fill_tid: int):
    """返回 (actual_tid, actual_text, top5 list, hint_rank, hint_prob, top1_prob)。"""
    if (not out0.token_ids
            or out0.logprobs is None
            or len(out0.logprobs) == 0):
        return -1, "<EMPTY>", [], -1, 0.0, 0.0
    actual_tid = int(out0.token_ids[0])
    actual_text = tok.decode([actual_tid])
    lp_dict = out0.logprobs[0]
    entries = []
    for tid, lp_obj in lp_dict.items():
        lp = float(getattr(lp_obj, "logprob", lp_obj))
        entries.append((int(tid), lp))
    entries.sort(key=lambda x: -x[1])
    top5 = []
    for tid, lp in entries[:5]:
        top5.append({
            "tid": tid, "text": tok.decode([tid]),
            "logp": lp, "prob": math.exp(lp),
        })
    top1_prob = top5[0]["prob"] if top5 else 0.0
    hint_rank, hint_prob = -1, 0.0
    for r_idx, e in enumerate(top5):
        if e["tid"] == fill_tid:
            hint_rank, hint_prob = r_idx + 1, e["prob"]
            break
    return actual_tid, actual_text, top5, hint_rank, hint_prob, top1_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str,
                    default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--fill_pool_path", type=str,
                    default=os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json"))
    ap.add_argument("--n_questions", type=int, default=200,
                    help="抽多少题 (从 fill_multi_pool 顺序取)")
    ap.add_argument("--n_cand_per_q", type=int, default=1,
                    help="每题取几个 candidate 作为示例")
    ap.add_argument("--max_prompt_length", type=int, default=8192,
                    help="vLLM prompt 窗口 (fill_answer 塞进示例后很长, 默认调大到 8192)")
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="生成完整答案的预算")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--output_dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "output", "verify_sdft_example"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable_thinking", type=str, default="auto",
                    choices=["auto", "true", "false"])
    ap.add_argument("--skip_baseline", action="store_true",
                    help="跳过裸 prompt 对照 (省一半生成时间)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # ----- 1) 加载 fill_multi_pool -----
    print(f"[load] fill pool ← {args.fill_pool_path}", flush=True)
    with open(args.fill_pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    pool = pool[: args.n_questions]
    print(f"[load] n_questions = {len(pool)}", flush=True)

    # ----- 2) tokenizer -----
    from transformers import AutoTokenizer
    print(f"[load] tokenizer ← {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ----- 3) 构造 teacher 示例 prompt (+ 可选裸 prompt 对照) -----
    teacher_prompts: List[str] = []
    meta: List[dict] = []   # {q_idx, fill_tid, fill_text, ref_answer, kind}
    for qi, item in enumerate(pool):
        question = item["question"]
        ref_answer = item.get("ref_answer", "")
        cands = item.get("candidates", [])[: args.n_cand_per_q]
        for c in cands:
            fill_tid = int(c["token_id"])
            fill_text = c["token_text"]
            fill_answer = str(c["answer"])
            user_content = TEACHER_TEMPLATE.substitute(
                orig_content=question, output_text=fill_answer
            )
            teacher_prompts.append(_apply_chat(tok, user_content, args.enable_thinking))
            meta.append({
                "q_idx": qi, "fill_tid": fill_tid, "fill_text": fill_text,
                "ref_answer": ref_answer, "kind": "teacher",
            })

    # 裸 prompt 对照: 每题一条 (用第一个 candidate 的 fill_tid 作为"期望 token"参照)
    baseline_prompts: List[str] = []
    baseline_meta: List[dict] = []
    if not args.skip_baseline:
        for qi, item in enumerate(pool):
            cands = item.get("candidates", [])
            if not cands:
                continue
            fill_tid = int(cands[0]["token_id"])
            fill_text = cands[0]["token_text"]
            baseline_prompts.append(_apply_chat(tok, str(item["question"]), args.enable_thinking))
            baseline_meta.append({
                "q_idx": qi, "fill_tid": fill_tid, "fill_text": fill_text,
                "ref_answer": item.get("ref_answer", ""), "kind": "baseline",
            })

    # prompt 长度分布 + 超长比例
    tlens = [len(tok.encode(p, add_special_tokens=False)) for p in teacher_prompts]
    over = sum(1 for L in tlens if L > args.max_prompt_length)
    print(f"[setup] teacher prompts={len(teacher_prompts)} "
          f"token_len min={min(tlens)} max={max(tlens)} "
          f"mean={sum(tlens)//max(len(tlens),1)} "
          f">{args.max_prompt_length}: {over} ({over/max(len(tlens),1)*100:.1f}%)",
          flush=True)
    if not args.skip_baseline:
        print(f"[setup] baseline prompts={len(baseline_prompts)}", flush=True)

    # 打印前 2 条 teacher prompt 末尾, 确认示例真的拼进去
    print("\n" + "=" * 90)
    print(" 前 2 条 teacher prompt 末尾 300 字符 (验证示例 + 'now answer' 是否拼上):")
    print("=" * 90)
    for i in range(min(2, len(teacher_prompts))):
        print(f"\n[{i}] fill_text={meta[i]['fill_text']!r}")
        print("    " + repr(teacher_prompts[i][-300:]))
    print("=" * 90 + "\n", flush=True)

    # ----- 4) vLLM -----
    print(f"[vllm] init (TP={args.tensor_parallel_size}) ...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path, trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_new_tokens,
        dtype="bfloat16",
    )
    sp = SamplingParams(
        n=1, temperature=0.0, top_p=1.0,
        max_tokens=args.max_new_tokens, logprobs=5, seed=args.seed,
    )

    def _run(prompts, tag):
        if not prompts:
            return []
        print(f"[vllm] generate {tag}: {len(prompts)} prompts "
              f"(max_tokens={args.max_new_tokens}) ...", flush=True)
        return llm.generate(prompts, sampling_params=sp, use_tqdm=True)

    teacher_outs = _run(teacher_prompts, "teacher")
    baseline_outs = _run(baseline_prompts, "baseline") if not args.skip_baseline else []

    # ----- 5) 统计 -----
    def _analyze(prompts, metas, outs):
        rows = []
        for p, m, o in zip(prompts, metas, outs):
            out0 = o.outputs[0]
            gen_text = out0.text
            actual_tid, actual_text, top5, hint_rank, hint_prob, top1_prob = \
                _first_token_top5(out0, tok, m["fill_tid"])
            match = (actual_tid == m["fill_tid"])
            correct = _is_correct(gen_text, m["ref_answer"])
            rows.append({
                **m, "actual_tid": actual_tid, "actual_text": actual_text,
                "match_fill": match, "correct": correct,
                "hint_rank": hint_rank, "hint_prob": hint_prob,
                "top1_prob": top1_prob, "top5": top5,
                "gen_len": len(out0.token_ids), "gen_text": gen_text,
            })
        return rows

    teacher_rows = _analyze(teacher_prompts, meta, teacher_outs)
    baseline_rows = _analyze(baseline_prompts, baseline_meta, baseline_outs) \
        if not args.skip_baseline else []

    def _agg(rows):
        n = len(rows)
        if n == 0:
            return dict(n=0, match=0, match_pct=0.0, correct=0, correct_pct=0.0,
                        in_top5=0, in_top5_pct=0.0, avg_hint_prob=0.0)
        match = sum(r["match_fill"] for r in rows)
        correct = sum(r["correct"] for r in rows)
        in_top5 = sum(1 for r in rows if r["hint_rank"] > 0)
        return dict(
            n=n, match=match, match_pct=match / n * 100,
            correct=correct, correct_pct=correct / n * 100,
            in_top5=in_top5, in_top5_pct=in_top5 / n * 100,
            avg_hint_prob=sum(r["hint_prob"] for r in rows) / n,
        )

    t_agg = _agg(teacher_rows)
    b_agg = _agg(baseline_rows)

    # ----- 6) 落盘 -----
    rows_path = os.path.join(args.output_dir, "rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for r in teacher_rows:
            f.write(json.dumps({**r, "src": "teacher"}, ensure_ascii=False) + "\n")
        for r in baseline_rows:
            f.write(json.dumps({**r, "src": "baseline"}, ensure_ascii=False) + "\n")
    print(f"\n[write] 逐条 → {rows_path}", flush=True)

    # per-fill-token 听话率 (teacher)
    per_fill: Dict[str, dict] = {}
    for r in teacher_rows:
        h = r["fill_text"]
        d = per_fill.setdefault(h, {"match": 0, "total": 0, "correct": 0,
                                    "actual_top": Counter()})
        d["total"] += 1
        d["match"] += int(r["match_fill"])
        d["correct"] += int(r["correct"])
        d["actual_top"][r["actual_text"]] += 1

    # ----- 7) 终端汇总 -----
    print("\n" + "=" * 80)
    print(" 核心结论: teacher 看 fill_answer 示例后的表现")
    print("=" * 80)
    print(f" {'':16}{'首token听话率':>16}{'正确率':>12}{'fill在top5':>12}{'avg fill_p':>12}")
    print(f" {'teacher(看示例)':16}{t_agg['match_pct']:>14.2f}% "
          f"{t_agg['correct_pct']:>10.2f}% {t_agg['in_top5_pct']:>10.2f}% "
          f"{t_agg['avg_hint_prob']:>12.4f}")
    if not args.skip_baseline:
        print(f" {'baseline(裸prompt)':16}{b_agg['match_pct']:>14.2f}% "
              f"{b_agg['correct_pct']:>10.2f}% {b_agg['in_top5_pct']:>10.2f}% "
              f"{b_agg['avg_hint_prob']:>12.4f}")
        print(f" {'增量(Δ)':16}{t_agg['match_pct']-b_agg['match_pct']:>+14.2f}% "
              f"{t_agg['correct_pct']-b_agg['correct_pct']:>+10.2f}%")
    print("=" * 80)

    print("\n per-fill-token 听话率 (teacher, 按听话率排序):")
    print(f" {'fill':<14}{'match':>7}{'total':>7}{'听话率':>9}{'正确率':>9}  top1实际(非fill)")
    print(" " + "-" * 80)
    for h, s in sorted(per_fill.items(),
                       key=lambda kv: -(kv[1]["match"] / max(kv[1]["total"], 1))):
        t = s["total"]
        non = [(k, v) for k, v in s["actual_top"].most_common() if k != h]
        other = ", ".join(f"{k!r}={v}" for k, v in non[:2]) if non else "(all fill)"
        print(f" {h!r:<14}{s['match']:>7}{t:>7}"
              f"{s['match']/t*100:>8.2f}%{s['correct']/t*100:>8.2f}%  {other}")
    print(" " + "-" * 80)

    # ----- 8) 5 条完整样例 -----
    samples_path = os.path.join(args.output_dir, "samples_5.jsonl")
    seen, sample_idx = set(), []
    for i, r in enumerate(teacher_rows):
        if r["fill_text"] not in seen:
            sample_idx.append(i)
            seen.add(r["fill_text"])
        if len(sample_idx) >= 5:
            break
    print("\n" + "=" * 90)
    print(" 5 条完整样例 (teacher 看示例后的生成)")
    print("=" * 90)
    with open(samples_path, "w", encoding="utf-8") as f:
        for si in sample_idx:
            r = teacher_rows[si]
            print(f"\n--- 样例 q_idx={r['q_idx']} fill={r['fill_text']!r} "
                  f"(tid={r['fill_tid']}) match={r['match_fill']} correct={r['correct']} ---")
            print(f" 实际首 token: {r['actual_text']!r}  hint_rank={r['hint_rank']} "
                  f"hint_prob={r['hint_prob']:.4f}")
            print(f" top-5:")
            for e in r["top5"]:
                mark = " ← fill" if e["tid"] == r["fill_tid"] else ""
                print(f"   {e['text']!r:<14} prob={e['prob']:.4f}{mark}")
            print(f" 生成前 200 字符: {r['gen_text'][:200]!r}")
            f.write(json.dumps({
                "q_idx": r["q_idx"], "fill_text": r["fill_text"], "fill_tid": r["fill_tid"],
                "teacher_prompt": teacher_prompts[si],
                "actual_first_token_text": r["actual_text"],
                "match_fill": r["match_fill"], "correct": r["correct"],
                "hint_rank": r["hint_rank"], "hint_prob": r["hint_prob"],
                "top5": r["top5"], "gen_text": r["gen_text"],
            }, ensure_ascii=False, indent=2) + "\n")
    print("\n" + "=" * 90)
    print(f"[write] 5 条样例 → {samples_path}", flush=True)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_questions": len(pool), "n_cand_per_q": args.n_cand_per_q,
            "max_prompt_length": args.max_prompt_length,
            "prompt_token_len": {"min": min(tlens), "max": max(tlens),
                                 "mean": sum(tlens) // max(len(tlens), 1),
                                 "over_limit": over},
            "teacher": t_agg, "baseline": b_agg,
            "per_fill": {
                h: {"match": s["match"], "total": s["total"],
                    "match_pct": s["match"] / s["total"] * 100,
                    "correct_pct": s["correct"] / s["total"] * 100,
                    "actual_top": dict(s["actual_top"].most_common(10))}
                for h, s in per_fill.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise

"""统计 Qwen3-8B 在 MATH train 7500 题上的"思考首词"分布 (thinking on).

⚠ 重要: Qwen3 chat_template + enable_thinking=True 会让生成:
  - 第 1 个 token: 100% `<think>` (tid 151667)
  - 第 2 个 token: 100% `\n`
  - 第 3 个 token: 真正的"思考开始词" (跟 R1-Distill 'Okay' 位置对齐)

本脚本统计**第 3 个 token**, 跳过被模板锁死的 <think> + \n。

vLLM 4 卡 TP, 每题生成 3 个 token: [<think>, \n, real_first_word].

用法:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/qwen3_first_token_stats.py \
    --model_path /workspace/SDCL_A_TOKEN/model/Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "output", "first_token_qwen3_8b"),
    )
    ap.add_argument("--max_questions", type=int, default=None,
                    help="限制题数 (debug 用), 默认全集 ~7500")
    ap.add_argument("--max_prompt_length", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--enable_thinking", type=str, default="true",
                    choices=["true", "false"],
                    help="Qwen3 chat_template 是否 enable_thinking (默认 true)")
    args = ap.parse_args()

    enable_thinking = (args.enable_thinking == "true")
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- 1) 加载数据 -----
    print(f"[load] Math_All(train=True) ...")
    from data_math.MATH_util import Math_All
    data = Math_All(train=True, subset_name="all")
    problems = data.problems
    if args.max_questions:
        problems = problems[: args.max_questions]
    print(f"[load] n_questions = {len(problems)}")

    # ----- 2) tokenizer + 应用 chat template -----
    from transformers import AutoTokenizer
    print(f"[load] tokenizer ← {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts: List[str] = []
    for q in problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        # Qwen3 chat_template 支持 enable_thinking 参数
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # 老版 tokenizer 不支持 enable_thinking, fallback
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)

    print(f"[setup] prompt[0] tail (200 chars):")
    print(repr(prompts[0][-200:]))
    print(f"[setup] enable_thinking={enable_thinking}")

    # 抓个示例 prompt token 长度
    sample_ids = tok.encode(prompts[0], add_special_tokens=False)
    print(f"[setup] prompt[0] token len = {len(sample_ids)}")

    # ----- 3) vLLM generate, max_tokens=1 (只要首 token) -----
    print(f"\n[vllm] init engine (TP={args.tensor_parallel_size}, "
          f"gpu_mem={args.gpu_memory_utilization}) ...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + 4,  # 给点余量
        dtype="bfloat16",
    )
    # greedy, max_tokens=3: 第1=<think>, 第2=\n, 取第3 (思考首词)
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=3,
        logprobs=0,
        seed=42,
    )
    print(f"[vllm] generate {len(prompts)} prompts (max_tokens=3, 取第 3 个 token) ...")
    outs = llm.generate(prompts, sampling_params=sp, use_tqdm=True)

    # ----- 4) 收集 第 3 个 token (跳过 <think> + \n) -----
    rows = []
    cnt: Counter = Counter()
    n_skipped_first_not_think = 0
    for idx, o in enumerate(outs):
        out0 = o.outputs[0]
        if len(out0.token_ids) < 3:
            tid = -1
            ttext = "<EMPTY_OR_SHORT>"
            head = ""
            first_tid = -1
            second_tid = -1
        else:
            first_tid = int(out0.token_ids[0])
            second_tid = int(out0.token_ids[1])
            tid = int(out0.token_ids[2])  # ← 第 3 个 token
            ttext = tok.decode([tid])
            head = out0.text[:60].replace("\n", "\\n")
        # 校验: 第 1 个 token 应该 100% 是 <think>
        if first_tid != -1 and first_tid != 151667:
            n_skipped_first_not_think += 1
        cnt[tid] += 1
        rows.append({
            "question_idx": idx,
            "first_token_id": first_tid,           # 应该是 <think>
            "second_token_id": second_tid,         # 应该是 \n
            "third_token_id": tid,                 # 关注的: 思考首词
            "third_token_text_repr": repr(ttext),
            "pred_head60": head,
        })

    if n_skipped_first_not_think > 0:
        print(f"\n[warn] {n_skipped_first_not_think}/{len(rows)} 题首 token 不是 <think>")

    # ----- 5) 落盘 + 打印 -----
    rows_path = os.path.join(args.output_dir, "lora_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[write] 逐题首 token → {rows_path}")

    n_total = len(rows)
    counts_dump = []
    for tid, c in cnt.most_common():
        text = tok.decode([tid]) if tid >= 0 else "<EMPTY>"
        counts_dump.append({
            "token_id": tid,
            "token_text": text,
            "count": c,
            "pct": c / n_total * 100,
        })
    counts_path = os.path.join(args.output_dir, "counts.json")
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_total": n_total,
            "unique": len(cnt),
            "enable_thinking": enable_thinking,
            "tokens": counts_dump,
        }, f, ensure_ascii=False, indent=2)
    print(f"[write] 计数表 → {counts_path}")

    # entropy
    probs = [c / n_total for c in cnt.values()]
    entropy_bits = -sum(p * math.log2(p) for p in probs if p > 0)

    # 终端汇总
    summary_lines = []
    summary_lines.append("=" * 84)
    summary_lines.append(f" Qwen3-8B 第 3 个 token 分布 (跳过 <think> + \\n, "
                         f"MATH train, n={n_total}, enable_thinking={enable_thinking})")
    summary_lines.append("=" * 84)
    summary_lines.append(f" 第 1 个 token: 100% <think> (tid 151667), 模板锁定")
    summary_lines.append(f" 第 2 个 token: 100% \\n, 模板锁定")
    summary_lines.append(f" 第 3 个 token: 真正的'思考首词', 跟 R1-Distill 'Okay' 位置对齐")
    summary_lines.append(f"")
    summary_lines.append(f" unique tokens : {len(cnt)}")
    summary_lines.append(f" entropy (bits): {entropy_bits:.4f}")
    summary_lines.append(f" 对比 R1-Distill (旧实验): MATH-500 上 unique=9, "
                         f"'Okay' 占 95%, entropy ≈ 0.34 bits")
    summary_lines.append("")
    summary_lines.append(f" Top-30 第 3 个 token:")
    summary_lines.append(f" {'rank':>4}  {'tid':>7}  {'count':>5}  {'pct':>7}  token_text")
    summary_lines.append(" " + "-" * 80)
    for i, (tid, c) in enumerate(cnt.most_common(30), 1):
        text = tok.decode([tid]) if tid >= 0 else "<EMPTY>"
        summary_lines.append(
            f" {i:>4}  {tid:>7}  {c:>5}  {c/n_total*100:>6.2f}%  {text!r}"
        )
    if len(cnt) > 30:
        rest = sum(c for _, c in cnt.most_common()[30:])
        summary_lines.append(
            f" {'...':>4}  {'rest':>7}  {rest:>5}  {rest/n_total*100:>6.2f}%  "
            f"(其它 {len(cnt) - 30} 个 token)"
        )
    summary_lines.append("=" * 84)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\n[write] 汇总 → {summary_path}")


if __name__ == "__main__":
    main()

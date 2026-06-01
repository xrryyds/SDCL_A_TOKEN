"""验证 GRPO 真正的 IS bug: vllm logp 和 actor logp 在 sequence 级别累加后被 TRL mask 掉。

TRL 0.27.2 默认 vllm_importance_sampling_correction=True, mode='sequence_mask', cap=3.0.
逻辑:
  per_token_diff = actor_logp - vllm_logp
  seq_sum_diff = sum over completion tokens
  seq_ratio = exp(seq_sum_diff)
  if seq_ratio > 3.0: mask 整段 (reward × 0)
  if seq_ratio < 1/3:  也异常 (mean ratio 接近 0 时大量 mask)

我们要测:
  (a) per-token diff mean / std (单 token 量级)
  (b) per-sequence sum_diff 分布 (整段累加后多大)
  (c) seq_ratio = exp(seq_sum_diff): 多少比例 > cap=3 (被 mask), 多少 < 1/cap=0.33 (被 mask)
  (d) 不修复 vs 我之前的 lp×T 修复, 谁的 mask 比例更低

不修复 = 直接用 vLLM 给的 sharp_logp 当 old_logp
修复1   = sharp_logp × T (我之前的尝试, 引入 bias)
修复2   = 不动 logp, 关掉 IS correction (推荐)

用法: 跟 verify_vllm_logp_unscale.py 一样, 单卡跑
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--cap", type=float, default=3.0,
                    help="TRL 默认 vllm_importance_sampling_cap")
    args = ap.parse_args()

    # ----- 1) 准备真实 MATH train prompts (而不是 toy 例子) -----
    print("[setup] load Math_All ...")
    from data_math.MATH_util import Math_All
    data = Math_All(train=True, subset_name="all")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts_text: List[str] = []
    for i in range(args.n_prompts):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data.problems[i]},
        ]
        prompts_text.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    prompt_ids_list = [tok.encode(p, add_special_tokens=False) for p in prompts_text]
    print(f"[setup] n_prompts={len(prompts_text)}, prompt lens=", [len(p) for p in prompt_ids_list])

    # ----- 2) vLLM generate, 拿 sharp_logp -----
    print("\n[vllm] init engine ...")
    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048 + args.max_new_tokens,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        logprobs=0,
        seed=42,
    )
    inputs = [TokensPrompt(prompt_token_ids=pids) for pids in prompt_ids_list]
    print(f"[vllm] generate (T={args.temperature}, top_p={args.top_p}) ...")
    outs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)

    completions: List[List[int]] = []
    sharp_logps: List[List[float]] = []
    for o in outs:
        out0 = o.outputs[0]
        cids = list(out0.token_ids)
        lps: List[float] = []
        if out0.logprobs is not None:
            for tid, lp_dict in zip(cids, out0.logprobs):
                lp_obj = lp_dict.get(tid) if isinstance(lp_dict, dict) else None
                lps.append(
                    float(getattr(lp_obj, "logprob", lp_obj)) if lp_obj is not None else 0.0
                )
        completions.append(cids)
        sharp_logps.append(lps)
    print(f"[vllm] completion lens=", [len(c) for c in completions])

    # 释放 vLLM
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ----- 3) HF model forward, 算 actor raw_logp -----
    print("\n[hf] load actor model ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    actor_logps: List[List[float]] = []
    for prompt_ids, comp_ids in zip(prompt_ids_list, completions):
        seq = prompt_ids + comp_ids
        ids = torch.tensor([seq], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        comp_lo = len(prompt_ids) - 1
        comp_hi = len(seq) - 1
        slice_logits = logits[comp_lo:comp_hi].float()
        slice_logp = torch.log_softmax(slice_logits, dim=-1)
        comp_tensor = torch.tensor(comp_ids, device="cuda", dtype=torch.long)
        token_actor_lp = slice_logp.gather(1, comp_tensor.unsqueeze(1)).squeeze(1)
        actor_logps.append(token_actor_lp.cpu().tolist())

    # ----- 4) 对三种策略计算 IS ratio 分布 -----
    T = args.temperature
    cap = args.cap

    def calc_seq_stats(label: str, old_logps_list: List[List[float]]):
        print(f"\n{'=' * 80}")
        print(f" [{label}]")
        print(f"{'=' * 80}")

        n_seq = len(old_logps_list)
        all_token_diff: List[float] = []
        seq_sum_diffs: List[float] = []
        seq_ratios: List[float] = []
        for actor, old, comp in zip(actor_logps, old_logps_list, completions):
            # per_token_diff = actor - old (TRL 同款)
            t_diff = [a - o for a, o in zip(actor, old)]
            all_token_diff.extend(t_diff)
            sum_diff = sum(t_diff)
            seq_sum_diffs.append(sum_diff)
            # clip exp 防 overflow
            seq_ratios.append(math.exp(max(min(sum_diff, 50), -50)))

        # token level
        token_mean = sum(all_token_diff) / len(all_token_diff)
        token_std = (sum((x - token_mean) ** 2 for x in all_token_diff) / len(all_token_diff)) ** 0.5
        print(f"  Per-token diff (actor - old): n={len(all_token_diff)}")
        print(f"    mean = {token_mean:+.4f}, std = {token_std:.4f}")
        print(f"    min  = {min(all_token_diff):+.4f}, max = {max(all_token_diff):+.4f}")

        # sequence level
        print(f"  Per-sequence sum_diff (n={n_seq}):")
        for i, sd in enumerate(seq_sum_diffs):
            print(f"    seq[{i}] len={len(completions[i])}, sum_diff={sd:+.3f}, "
                  f"seq_ratio=exp({sd:+.2f})={seq_ratios[i]:.4g}")

        # mask analysis
        n_high = sum(1 for r in seq_ratios if r > cap)
        n_low = sum(1 for r in seq_ratios if r < 1 / cap)
        n_in = n_seq - n_high - n_low
        print(f"  Mask analysis (cap={cap}, mode=sequence_mask):")
        print(f"    seq_ratio > {cap}      : {n_high}/{n_seq} 被 mask (太大, ratio 爆)")
        print(f"    seq_ratio < {1/cap:.3f}  : {n_low}/{n_seq} 也算异常 (mean 数据偏)")
        print(f"    in [{1/cap:.3f}, {cap}]  : {n_in}/{n_seq} 健康未 mask")
        # token_truncate 模式: 只 clip token-level 比例
        # 这个就是 ratio 分布
        all_token_ratios = [math.exp(max(min(d, 50), -50)) for d in all_token_diff]
        n_tok_high = sum(1 for r in all_token_ratios if r > cap)
        n_tok = len(all_token_ratios)
        print(f"  Token-level (cap={cap}, mode=token_truncate): "
              f"{n_tok_high}/{n_tok} ({n_tok_high/n_tok*100:.2f}%) tokens 被 clip")

    # 策略 A: 直接用 vLLM sharp_logp 作为 old (TRL 默认行为, "不修复")
    calc_seq_stats("不修复 (old = vLLM sharp_logp)", sharp_logps)

    # 策略 B: 用 sharp_logp × T 作为 old (我之前那个错误修复)
    corrected_lp = [[sp * T for sp in lps] for lps in sharp_logps]
    calc_seq_stats(f"修复1: old = sharp_logp × T (T={T}, 我之前那个改动)", corrected_lp)

    print()
    print("=" * 80)
    print(" 结论判断")
    print("=" * 80)
    print(" - 如果 [不修复] 'seq_ratio > cap' 比例高 → TRL 默认配置会 mask 大量 seq, GRPO 学不动")
    print(" - 如果 [修复1]  'seq_ratio > cap' 也很高 → 我的修复没解决问题")
    print(" - 推荐做法: 不动 logp, 关掉 vllm_importance_sampling_correction (不再 mask)")
    print("            或换 mode='token_truncate' (只 clip 个别 token, 不整段 mask)")


if __name__ == "__main__":
    main()

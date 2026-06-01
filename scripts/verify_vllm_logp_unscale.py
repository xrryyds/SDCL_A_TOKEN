"""快速验证: vLLM logp × T 还原 = HF forward raw logp ?

不依赖 TRL, 用 vLLM + transformers 各自跑一次 forward,
对比同一 (prompt, completion) 序列上每 token 的 logp。

修复假设:
  vLLM 返回 sharp_logp = log_softmax(logits / T)_tok
  HF forward raw_logp = log_softmax(logits)_tok
  我们的修复: lp_corrected = sharp_logp * T
  期望: lp_corrected 接近 raw_logp, ratio = exp(raw - corrected) ≈ 1

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/verify_vllm_logp_unscale.py \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B
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
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n_prompts", type=int, default=2)
    args = ap.parse_args()

    # ----- 1) 准备 prompts -----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test_problems = [
        "What is 2 + 3?",
        "What is the value of 7 * 8?",
    ][: args.n_prompts]

    SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
    prompts_text: List[str] = []
    for q in test_problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts_text.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    prompt_ids_list = [
        tok.encode(p, add_special_tokens=False) for p in prompts_text
    ]
    print(f"[setup] n_prompts={len(prompts_text)}")
    for i, pids in enumerate(prompt_ids_list):
        print(f"  prompt[{i}] len={len(pids)}")

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
        print(f"  completion len={len(cids)}, first 5 ids={cids[:5]}")
        print(f"  sharp_logp first 5={[f'{x:.3f}' for x in lps[:5]]}")

    # 释放 vLLM 显存, 让 HF model 能上同一张卡
    print("\n[vllm] del engine, empty cache ...")
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ----- 3) HF model forward, 算 raw_logp -----
    print("\n[hf] load model ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    raw_logps: List[List[float]] = []
    for prompt_ids, comp_ids in zip(prompt_ids_list, completions):
        seq = prompt_ids + comp_ids
        ids = torch.tensor([seq], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids)
            logits = out.logits[0]  # [T, V]
        # 预测 seq[t] 来自 logits[t-1]; 我们要 completion 段每个 token 的 raw_logp
        # completion 起点在 seq 中 index = len(prompt_ids); logits[len(prompt_ids)-1] 预测它
        comp_lo = len(prompt_ids) - 1
        comp_hi = len(seq) - 1  # 不含
        # raw logp via log_softmax (T=1)
        slice_logits = logits[comp_lo:comp_hi].float()
        slice_logp = torch.log_softmax(slice_logits, dim=-1)
        # gather 对应 token id
        comp_tensor = torch.tensor(comp_ids, device="cuda", dtype=torch.long)
        token_raw_lp = slice_logp.gather(1, comp_tensor.unsqueeze(1)).squeeze(1)
        raw_logps.append(token_raw_lp.cpu().tolist())

    # ----- 4) 对比统计 -----
    T = args.temperature
    print(f"\n{'=' * 80}")
    print(f" 验证: lp_corrected = sharp_logp * T (T={T})")
    print(f"{'=' * 80}")
    print(f"{'prompt':>6} {'tok_idx':>7} {'tid':>7} {'sharp_lp':>10} "
          f"{'corrected':>10} {'raw_lp':>10} {'diff':>10} {'ratio':>10}")
    print("-" * 80)

    all_diff: List[float] = []
    all_ratio: List[float] = []

    for pi, (sharps, raws, cids) in enumerate(zip(sharp_logps, raw_logps, completions)):
        n_show = min(8, len(sharps))
        for ti in range(n_show):
            sharp = sharps[ti]
            corrected = sharp * T
            raw = raws[ti]
            diff = raw - corrected   # 应该 ~0
            ratio = math.exp(min(diff, 50))   # exp(diff), 防 overflow
            all_diff.append(diff)
            all_ratio.append(ratio)
            print(f"{pi:>6} {ti:>7} {cids[ti]:>7} {sharp:>10.3f} "
                  f"{corrected:>10.3f} {raw:>10.3f} {diff:>+10.3f} {ratio:>10.4f}")
        # 整段统计
        diffs = [r - s * T for r, s in zip(raws, sharps)]
        ratios = [math.exp(min(d, 50)) for d in diffs]
        print(f"  prompt[{pi}] over {len(diffs)} tokens: "
              f"diff mean={sum(diffs)/len(diffs):+.4f}, "
              f"ratio mean={sum(ratios)/len(ratios):.4f}, "
              f"ratio min={min(ratios):.4f}, max={max(ratios):.4f}")
        # 全部累加
        all_diff.extend(diffs[n_show:])
        all_ratio.extend(ratios[n_show:])
        print()

    # 全局
    print("=" * 80)
    print(f" 全局 (n={len(all_diff)} tokens):")
    print(f"   修复后 diff (raw - corrected) mean={sum(all_diff)/len(all_diff):+.4f}")
    print(f"   修复后 ratio = exp(diff): "
          f"mean={sum(all_ratio)/len(all_ratio):.4f}, "
          f"min={min(all_ratio):.4f}, max={max(all_ratio):.4f}")
    print(f"   理想: ratio mean ≈ 1.0 (PPO 起步 on-policy)")
    print()

    # 对比: 不修复时 ratio 长啥样
    bad_diff = [r - s for r, s in zip(
        [x for ls in raw_logps for x in ls],
        [x for ls in sharp_logps for x in ls],
    )]
    bad_ratio = [math.exp(min(d, 50)) for d in bad_diff]
    print(f" 对比: 不修复 (vLLM 原 sharp_lp 当 old):")
    print(f"   diff mean={sum(bad_diff)/len(bad_diff):+.4f}")
    print(f"   ratio mean={sum(bad_ratio)/len(bad_ratio):.4f}, "
          f"min={min(bad_ratio):.4f}, max={max(bad_ratio):.4f}")
    print(f"   (训练 log 实测 ratio mean=0.0005, 应该跟这个 bad ratio 一致)")
    print("=" * 80)


if __name__ == "__main__":
    main()

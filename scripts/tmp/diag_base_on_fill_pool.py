"""验证: take_exam 直接跑 Base on fill_multi_pool.json 1181 题。

背景:
  fill_multi_pool.json 是从 mistake 池 1379 题里 fill 救回的 1181 题子集,
  这些题 Base 在重建池时全部做错。用户已多次评测 1379 题 Base 稳定 0%。
  但 eval_v3.py 评出 Base on fill_multi_pool = 30.57% (361/1181), 不合理。

目标:
  复现/排除 take_exam 本身的 bug, 与 eval_v3.py 调度无关。
  - 若这里 ≈ 0%   → eval_v3 调度 bug (Base pass 误用了 LoRA / 数据集错乱 / ...)
  - 若这里 ≈ 30% → take_exam 本身有 bug (与池构造时跑出的 Base 答案不一致)

口径 (与 eval_v3 + 池构造对齐):
  max_prompt_length=10240, max_new_tokens=8192
  T=0, top_p=1, sample_n=1, greedy
  模型: Base, 无 LoRA

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_base_on_fill_pool.py
"""
import json, gc, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam

MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
POOL = os.path.join(_ROOT, "datasets", "exam", "fill_multi_pool.json")


def judge(ans, ref):
    b = extract_boxed_content(ans or "")
    if b is None:
        return None
    return normalize_answer(b) == normalize_answer(str(ref))


def main():
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    d = json.load(open(POOL))
    q = [it["question"] for it in d]
    ref = [str(it["ref_answer"]) for it in d]
    sol = [it.get("ref_solution", "") for it in d]
    idx = list(range(len(d)))
    print(f"[load] fill_multi_pool: {len(d)} 题", flush=True)
    print(f"[cfg]  max_prompt=10240 max_new=8192 greedy(T=0) device_ids={device_ids}", flush=True)
    print(f"[cfg]  use_lora=False (纯 Base, 不传 adapter_path)", flush=True)

    te = TakeExam(model_path=MODEL, max_prompt_length=10240, max_new_tokens=8192)
    try:
        res = te.exam_multi_gpu(
            q, sol, ref, idx,
            device_ids=device_ids, write_output=False,
            sample_n=1, temperature=0.0, top_p=1.0,
        )
    finally:
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_ok = 0
    n_none = 0
    for r in res:
        i = r["question_idx"]
        j = judge(r["answer"], ref[i])
        if j is None:
            n_none += 1
        elif j:
            n_ok += 1
    N = len(res)

    print("\n" + "=" * 70)
    print(f"Base on fill_multi_pool.json (1181 题, mistake 子集 fill 救回)")
    print("=" * 70)
    print(f"  做对          : {n_ok}/{N} = {n_ok/N*100:.2f}%  ← 期望 ≈ 0%")
    print(f"  没 boxed      : {n_none}/{N} = {n_none/N*100:.2f}%")
    print(f"  做错 (有boxed): {N-n_ok-n_none}/{N} = {(N-n_ok-n_none)/N*100:.2f}%")
    print("=" * 70)
    print("解读:")
    print("  ≈0%   → eval_v3 调度 bug (Base pass 误用 LoRA / 路径错 / ...)")
    print("  ≈30%  → take_exam 本身漂移, 池构造和现在的 Base 答案不同")


if __name__ == "__main__":
    main()

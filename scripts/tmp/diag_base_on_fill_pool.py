"""验证: take_exam 直接跑 Base on mistake/fill 池, 看与 eval_v3 评出的数字是否一致。

背景:
  用户多次评测 mistake_DS_MATH_pool.json 1379 题 Base 稳定 0%。
  但 eval_v3.py 评出 Base on fill_multi_pool.json (1181 题) = 30.57% (361/1181), 不合理。
  fill_multi_pool 是 mistake 池子集 (Base 都做错), Base 该 ≈0%。

目标:
  用 take_exam 直接跑 Base, 复现/排除 take_exam 本身的 bug, 与 eval_v3 调度无关。
  - 若这里 ≈ 0%   → eval_v3 调度 bug
  - 若这里 ≈ 30% → take_exam 本身漂移 (与池构造时跑出的 Base 答案不一致)

口径 (与 eval_v3 + 池构造对齐):
  max_prompt_length=10240, max_new_tokens=8192
  T=0, top_p=1, sample_n=1, greedy
  模型: Base, 无 LoRA

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  # 默认测 mistake 池 1379 题 (用户已多次确认 0% 的基准)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_base_on_fill_pool.py

  # 测 fill 池 1181 题 (eval_v3 评出 30.57% 的那个)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_base_on_fill_pool.py --pool fill
"""
import argparse, json, gc, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam

MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
MISTAKE = os.path.join(_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")
FILL = os.path.join(_ROOT, "datasets", "exam", "fill_multi_pool.json")


def judge(ans, ref):
    b = extract_boxed_content(ans or "")
    if b is None:
        return None
    return normalize_answer(b) == normalize_answer(str(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["mistake", "fill"], default="mistake",
                    help="mistake=主 mistake 池 / fill=主 fill_multi_pool")
    ap.add_argument("--path", type=str, default=None,
                    help="任意 mistake 池路径 (覆盖 --pool, 用于跑 .bak.* 旧池)")
    args = ap.parse_args()

    if args.path:
        pool_path = args.path
        label = os.path.basename(pool_path)
    else:
        pool_path = MISTAKE if args.pool == "mistake" else FILL
        label = args.pool

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    d = json.load(open(pool_path))
    q = [it["question"] for it in d]
    ref = [str(it["ref_answer"]) for it in d]
    sol = [it.get("ref_solution", "") for it in d]
    idx = list(range(len(d)))
    print(f"[load] {label} 池: {len(d)} 题  ← {pool_path}", flush=True)
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
    print(f"Base on {label} 池 ({N} 题)")
    print("=" * 70)
    print(f"  做对          : {n_ok}/{N} = {n_ok/N*100:.2f}%  ← 期望 ≈ 0%")
    print(f"  没 boxed      : {n_none}/{N} = {n_none/N*100:.2f}%")
    print(f"  做错 (有boxed): {N-n_ok-n_none}/{N} = {(N-n_ok-n_none)/N*100:.2f}%")
    print("=" * 70)
    print("解读:")
    print("  ≈0%   → take_exam 没问题, eval_v3 调度有 bug")
    print("  ≈30%  → take_exam 本身漂移, 池构造和现在的 Base 答案不同")


if __name__ == "__main__":
    main()

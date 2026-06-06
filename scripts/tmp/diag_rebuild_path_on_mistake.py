"""复刻 rebuild_math_pool_8k 的算法路径, 但只跑当前 1379 题 mistake 池。

背景:
  - 2026-06-04 第一次 rebuild: corr=6077 mistake=1419 acc=81.07%
  - 2026-06-05 (本会话) 重跑 rebuild: corr=6117 mistake=1379 acc=81.60%
  - diag_base_on_fill_pool 重评当前 1379 池: 344/1379=24.95% 做对 (异常)
  - 用户记忆里 1419 池稳定 0%

矛盾: rebuild 路径生成 mistake 池时这些题都判错 (定义上), 但 diag 路径重评 25% 做对。
两条路径 vLLM 调用相同 (都是 TakeExam.exam_multi_gpu), 但题数不同 (rebuild=7500, diag=1379)。

本脚本: 用 rebuild 那条 take_exam → boxed match 判分路径, 只跑 1379 题, 看 acc。

预期分支:
  ≈0%   → rebuild 路径稳, diag 有 bug (但代码看两边一样, 不该差)
  ≈25%  → rebuild 路径自己跑 1379 题也 25% 对, 说明 vLLM 跑不同 batch/题集出不同输出
            → 这就是根因: vLLM 长 greedy 跨题集不可重现
  其他  → 再分析

口径 (与 rebuild_math_pool_8k 完全一致):
  TakeExam(model_path, max_prompt_length=10240, max_new_tokens=8192)
  exam_multi_gpu(... sample_n=1, temperature=0.0, top_p=1.0, write_output=False)
  judge: extract_boxed_content + normalize_answer (与 teacher_mark_paper 一致)

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_rebuild_path_on_mistake.py
"""
import argparse, gc, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam

MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
MIS = os.path.join(_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")


def judge(ans, ref):
    """与 teacher_mark_paper 一致: extract_boxed_content + normalize_answer。"""
    b = extract_boxed_content(ans or "")
    if b is None:
        return None
    return normalize_answer(b) == normalize_answer(str(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mistake_path", type=str, default=MIS,
                    help="mistake 池路径 (默认主池, 可改 .bak.* 测旧 1419 池)")
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    d = json.load(open(args.mistake_path))
    # 从池里直接拿 question / ref / question_idx, 不重置 idx (保持原始映射, 跟 rebuild 一致)
    q = [it["question"] for it in d]
    ref = [str(it["ref_answer"]) for it in d]
    sol = [it.get("ref_solution", "") for it in d]
    qidx = [it.get("question_idx", i) for i, it in enumerate(d)]

    print(f"[load] {args.mistake_path}", flush=True)
    print(f"       {len(d)} 题, question_idx 范围: {min(qidx)}..{max(qidx)} (unique={len(set(qidx))})", flush=True)
    print(f"[cfg]  TakeExam max_prompt=10240 max_new=8192 (与 rebuild 一致)", flush=True)
    print(f"[cfg]  greedy: sample_n=1 T=0 top_p=1, device_ids={device_ids}", flush=True)
    print(f"[cfg]  Base, no LoRA", flush=True)

    # 跟 student_take_exam_Math_sub 同款构造 (main.py:617-621)
    te = TakeExam(MODEL, max_prompt_length=10240, max_new_tokens=8192)
    try:
        # 跟 student_take_exam_Math_sub 同款调用 (main.py:626-635)
        # 但 write_output 不传 (默认走 take_exam 内部, 我们用返回值)
        res = te.exam_multi_gpu(
            q, sol, ref, qidx,
            sample_n=1, temperature=0.0, top_p=1.0,
            device_ids=device_ids,
            write_output=False,
        )
    finally:
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 判分 (与 teacher_mark_paper 一致)
    n_ok = 0
    n_none = 0
    n_real_wrong = 0
    by_qi = {r["question_idx"]: r["answer"] for r in res}
    for it in d:
        qi = it.get("question_idx")
        ans = by_qi.get(qi)
        if ans is None:
            # 这题没结果 (异常, 不该发生)
            continue
        j = judge(ans, str(it["ref_answer"]))
        if j is None:
            n_none += 1
        elif j:
            n_ok += 1
        else:
            n_real_wrong += 1
    N = n_ok + n_none + n_real_wrong

    print("\n" + "=" * 70)
    print(f"rebuild 算法路径重评 1379 mistake 池 Base")
    print("=" * 70)
    print(f"  做对 (boxed==ref)        : {n_ok}/{N} = {n_ok/N*100:.2f}%  ← 期望≈0% (rebuild 池定义)")
    print(f"  没 boxed (截断)          : {n_none}/{N} = {n_none/N*100:.2f}%")
    print(f"  真错 (有 boxed 但 != ref): {n_real_wrong}/{N} = {n_real_wrong/N*100:.2f}%")
    print("=" * 70)
    print(f"对照 diag_base_on_fill_pool: 344/1379 = 24.95%")
    print("解读:")
    print("  本次 ≈0%   → rebuild 路径稳定, diag 路径有 bug (但代码看两边一致)")
    print("  本次 ≈25%  → rebuild 路径自己跑 1379 题也 ~25% 对; vLLM 跨题集不可重现 (根因)")
    print("  本次其他   → 再分析")


if __name__ == "__main__":
    main()

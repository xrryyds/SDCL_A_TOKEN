"""验证 vLLM greedy 两次生成的抖动: 同一批 MATH train 题, 完全相同参数跑两次,
对比 boxed 判对结果有多少题翻转。

回答: Base 在 mistake 池评出 27% 到底是不是 vLLM greedy 非确定性导致的。
若两次翻转率 << 27%, 说明 27% 不是抖动, 另有原因 (截断/池构造)。

用法 (远程 4 卡):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_greedy_jitter.py --n 300 --max_new 8192
"""
import argparse, gc, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam
from data_math.MATH_util import Math_All


def judge(ans, ref):
    b = extract_boxed_content(ans or "")
    if b is None:
        return None  # 没 boxed (截断/没答完)
    return normalize_answer(b) == normalize_answer(str(ref))


def run_once(questions, ref_solutions, ref_answers, indices, device_ids,
             max_prompt_length, max_new):
    te = TakeExam(model_path=MODEL, max_prompt_length=max_prompt_length, max_new_tokens=max_new)
    try:
        res = te.exam_multi_gpu(
            questions, ref_solutions, ref_answers, indices,
            device_ids=device_ids, write_output=False,
            sample_n=1, temperature=0.0, top_p=1.0,
        )
    finally:
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {r["question_idx"]: r["answer"] for r in res}


MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="抽多少题")
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--max_prompt_length", type=int, default=10240, help="vLLM 总窗口=2048+max_new")
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = [int(x) for x in cuda.split(",") if x.strip()] if cuda else [0]
    # take_exam 子进程会重设 CUDA_VISIBLE_DEVICES, 用逻辑 id
    device_ids = list(range(len(device_ids)))

    data = Math_All(train=True, subset_name="all")
    q = data.problems[:args.n]
    sol = data.solutions[:args.n]
    ans = data.answers[:args.n]
    idx = list(range(len(q)))
    print(f"[load] {len(q)} 题 MATH train, max_new={args.max_new} 总窗口={args.max_prompt_length}", flush=True)

    print("[run 1/2] 第一次 greedy ...", flush=True)
    r1 = run_once(q, sol, ans, idx, device_ids, args.max_prompt_length, args.max_new)
    print("[run 2/2] 第二次 greedy (相同参数) ...", flush=True)
    r2 = run_once(q, sol, ans, idx, device_ids, args.max_prompt_length, args.max_new)

    # 对齐统计
    same_text = 0          # 两次生成文本完全相同
    judge_flip = 0         # 判对结果翻转 (对<->错, 含 None)
    both_correct = 0
    both_wrong = 0
    flip_w2c = 0           # 第一次错/None, 第二次对
    flip_c2w = 0           # 第一次对, 第二次错/None
    n_none1 = 0
    n_none2 = 0
    n = 0
    for i in idx:
        a1, a2 = r1.get(i), r2.get(i)
        if a1 is None or a2 is None:
            continue
        n += 1
        if a1 == a2:
            same_text += 1
        j1 = judge(a1, ans[i])
        j2 = judge(a2, ans[i])
        if j1 is None:
            n_none1 += 1
        if j2 is None:
            n_none2 += 1
        c1 = (j1 is True)
        c2 = (j2 is True)
        if c1 and c2:
            both_correct += 1
        elif (not c1) and (not c2):
            both_wrong += 1
        if c1 != c2:
            judge_flip += 1
            if (not c1) and c2:
                flip_w2c += 1
            else:
                flip_c2w += 1

    print("\n" + "=" * 60)
    print(f"vLLM greedy 两次生成抖动 (n={n} 题, T=0 同参数)")
    print("=" * 60)
    print(f"  两次文本完全相同      : {same_text}/{n} = {same_text/n*100:.1f}%")
    print(f"  判对结果翻转 (对<->错): {judge_flip}/{n} = {judge_flip/n*100:.1f}%  ← 这就是'抖动率'")
    print(f"    其中 错→对          : {flip_w2c}")
    print(f"    其中 对→错          : {flip_c2w}")
    print(f"  两次都对              : {both_correct}")
    print(f"  两次都错              : {both_wrong}")
    print(f"  第一次没boxed(截断)   : {n_none1}/{n} = {n_none1/n*100:.1f}%")
    print(f"  第二次没boxed(截断)   : {n_none2}/{n} = {n_none2/n*100:.1f}%")
    print("=" * 60)
    print("解读:")
    print("  若'判对翻转率' << 27% → mistake池27%不是抖动, 是截断/池构造问题")
    print("  若'没boxed率'高 → 8192不够, 长链截断是主因 (翻转多来自截断边界题)")


if __name__ == "__main__":
    main()

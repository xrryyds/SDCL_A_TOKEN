"""验证新 mistake 池在当前 take_exam 下的抖动率: 两次独立 greedy run, 对比文本/判分。

口径 (与 rebuild_math_pool_8k 一致):
  max_prompt_length=10240 (vLLM 总窗口 = 2048 prompt + 8192 gen)
  max_new_tokens=8192
  temperature=0.0, top_p=1.0, sample_n=1
  模型: Base (无 LoRA)

流程:
  轮1: 启 vLLM → 对全部 mistake 池题 greedy → answers_1 → 卸载
  轮2: 启 vLLM → 同样题 greedy → answers_2 → 卸载
  对比: answers_1 vs answers_2 (文本相同率 / 判对翻转率)

  期望: 若新池稳定, 文本完全相同率应 ≈100%, 判对翻转率 ≈0%。
        若仍有大量翻转 → vLLM 在长生成上有非确定性 (即使 greedy)。

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_pool_jitter.py
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
    b = extract_boxed_content(ans or "")
    if b is None:
        return None
    return normalize_answer(b) == normalize_answer(str(ref))


def run_once(questions, ref_solutions, ref_answers, indices, device_ids,
             max_prompt_length, max_new):
    te = TakeExam(model_path=MODEL,
                  max_prompt_length=max_prompt_length,
                  max_new_tokens=max_new)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="抽多少题, 0=全部 mistake 池")
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--max_prompt_length", type=int, default=10240,
                    help="vLLM 总窗口 = 2048 prompt + 8192 gen")
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    mis_all = json.load(open(MIS))
    mis = mis_all if args.n <= 0 else mis_all[:args.n]
    q = [it["question"] for it in mis]
    ans_ref = [str(it["ref_answer"]) for it in mis]
    sol = [it.get("ref_solution", "") for it in mis]
    # 用列表下标做 idx, 不依赖原 question_idx
    idx = list(range(len(q)))

    print(f"[load] mistake 池 {len(mis_all)} 题, 本次跑 {len(q)} 题",
          flush=True)
    print(f"[cfg]  max_prompt={args.max_prompt_length} max_new={args.max_new} "
          f"greedy(T=0,top_p=1) device_ids={device_ids}", flush=True)

    print("\n[run 1/2] 第一次 greedy ...", flush=True)
    r1 = run_once(q, sol, ans_ref, idx, device_ids,
                  args.max_prompt_length, args.max_new)
    print("[run 2/2] 第二次 greedy (相同参数) ...", flush=True)
    r2 = run_once(q, sol, ans_ref, idx, device_ids,
                  args.max_prompt_length, args.max_new)

    same_text = 0
    judge_flip = 0
    both_correct = 0
    both_wrong = 0
    flip_w2c = 0
    flip_c2w = 0
    n_none1 = 0
    n_none2 = 0
    n = 0
    flip_examples = []   # 抽几个翻转例子看
    for i in idx:
        a1, a2 = r1.get(i), r2.get(i)
        if a1 is None or a2 is None:
            continue
        n += 1
        if a1 == a2:
            same_text += 1
        j1 = judge(a1, ans_ref[i])
        j2 = judge(a2, ans_ref[i])
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
            if len(flip_examples) < 3:
                flip_examples.append((i, ans_ref[i], j1, j2, a1[-80:], a2[-80:]))

    print("\n" + "=" * 70)
    print(f"新 mistake 池 vLLM greedy 两次抖动 (n={n} 题)")
    print("=" * 70)
    print(f"  两次文本完全相同      : {same_text}/{n} = {same_text/n*100:.2f}%")
    print(f"  判对结果翻转 (对<->错): {judge_flip}/{n} = {judge_flip/n*100:.2f}%  ← 抖动率")
    print(f"    错→对              : {flip_w2c}")
    print(f"    对→错              : {flip_c2w}")
    print(f"  两次都对              : {both_correct}/{n} = {both_correct/n*100:.2f}%")
    print(f"  两次都错              : {both_wrong}/{n} = {both_wrong/n*100:.2f}%")
    print(f"  第一次没boxed(截断)   : {n_none1}/{n} = {n_none1/n*100:.2f}%")
    print(f"  第二次没boxed(截断)   : {n_none2}/{n} = {n_none2/n*100:.2f}%")
    print("=" * 70)
    print("解读:")
    print("  文本相同≈100%, 翻转≈0% → 新池 + 当前 take_exam 完全稳定, 27% 已被根治")
    print("  文本相同低 → vLLM greedy 仍非确定 (长生成累积浮点漂移, 池构造本身不可重复)")
    print("  翻转主要在'没boxed↔有boxed'边界 → 长链尾部漂移导致截断/收尾不一致")

    if flip_examples:
        print("\n--- 翻转样例 (前3) ---")
        for i, ref, j1, j2, t1, t2 in flip_examples:
            print(f"  i={i} ref={ref!r} j1={j1} j2={j2}")
            print(f"    r1 尾80: {t1!r}")
            print(f"    r2 尾80: {t2!r}")


if __name__ == "__main__":
    main()

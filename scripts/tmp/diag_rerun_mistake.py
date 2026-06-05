"""直接拿 mistake 池的题, 用当前 take_exam 重评 Base, 看正确率。

矛盾: MATH train 前300题 Base 现在做对 82.7%, 且 greedy 零抖动零截断。
那 mistake 池(应是 Base 做错的题)用当前流程重评, Base 该做对多少?
  - 若接近 0% → 池和评测口径一致, 之前 27% 另有原因 (eval_v3 调用差异)
  - 若 ~27% 或更高 → 池构造口径和当前 take_exam 不一致 (prompt/参数变了)

并打印: mistake 池里存的 answer vs 现在重新生成的 answer 是否相同 (看口径是否一致)。

用法 (远程 4 卡):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_rerun_mistake.py --n 300 --max_new 8192
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--max_prompt_length", type=int, default=10240)
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    mis = json.load(open(MIS))[:args.n]
    q = [it["question"] for it in mis]
    ans_ref = [str(it["ref_answer"]) for it in mis]
    sol = [it.get("ref_solution", "") for it in mis]
    idx = list(range(len(q)))
    stored = [it.get("answer", "") for it in mis]  # 池里构造时存的 Base 输出
    print(f"[load] mistake 池前 {len(q)} 题, max_new={args.max_new}", flush=True)

    te = TakeExam(model_path=MODEL, max_prompt_length=args.max_prompt_length, max_new_tokens=args.max_new)
    try:
        res = te.exam_multi_gpu(q, sol, ans_ref, idx, device_ids=device_ids,
                                write_output=False, sample_n=1, temperature=0.0, top_p=1.0)
    finally:
        del te; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_idx = {r["question_idx"]: r["answer"] for r in res}

    n_correct = 0      # 现在重评 Base 做对 (本应=0)
    n_none = 0
    n_same_as_stored = 0   # 现在生成 == 池里存的
    for i in idx:
        new_ans = by_idx.get(i)
        if new_ans is None:
            continue
        j = judge(new_ans, ans_ref[i])
        if j is None:
            n_none += 1
        elif j:
            n_correct += 1
        if new_ans.strip() == stored[i].strip():
            n_same_as_stored += 1

    N = len(q)
    print("\n" + "=" * 60)
    print(f"mistake 池前 {N} 题, 用当前 take_exam 重评 Base")
    print("=" * 60)
    print(f"  现在做对          : {n_correct}/{N} = {n_correct/N*100:.1f}%  ← 本应≈0%")
    print(f"  现在没boxed(截断) : {n_none}/{N} = {n_none/N*100:.1f}%")
    print(f"  生成==池里存的    : {n_same_as_stored}/{N} = {n_same_as_stored/N*100:.1f}%  ← 看口径是否一致")
    print("=" * 60)
    print("解读:")
    print("  做对≈0% + 生成==存的≈100% → 口径一致, 池没问题 (eval_v3 的 27% 来自别处)")
    print("  做对高 + 生成≠存的 → 当前 take_exam 口径和构造池时不一致 (prompt/参数漂移)")

    # 抽1题看构造存的 vs 现在生成的差异
    print("\n--- 抽样 1 题: 池里存的 vs 现在生成 (头200字符) ---")
    i0 = idx[0]
    print(f"q_idx={i0} ref={ans_ref[i0]!r}")
    print(f"  池里存的 头200: {stored[i0][:200]!r}")
    print(f"  现在生成 头200: {(by_idx.get(i0) or '')[:200]!r}")


if __name__ == "__main__":
    main()

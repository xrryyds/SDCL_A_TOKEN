"""验证 mistake 池现状: 比池里存的 Base 答案 vs 当前重新跑的 Base 答案。

如果池里存的 answer 是 Base 的输出 (按 teacher_mark_paper 的逻辑, 是),
且这些题 Base 当时判错 → boxed 应该都 != ref_answer。

本脚本静态分析 (不跑模型):
  1. 池里存的 answer 现在 judge 一遍, 看判对率 (应≈0% 如果池干净)
  2. 抽 5 题打印: 池里存的 answer 头/尾 + boxed
  3. 看池的题数、ref_answer 分布、有无重复 question_idx

如果池里"存的 answer" 自己 judge 都有 ~25% 做对 → 池脏 (构造时判分和现在判分不一致)
如果池里"存的 answer" judge 都 ~0% → 池干净, 现在 24.95% 是 vLLM 重新跑的漂移 (take_exam 漂移)

用法:
  python scripts/tmp/diag_pool_self_judge.py
"""
import os, sys, json

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.data_utils import extract_boxed_content, normalize_answer

MIS = os.path.join(_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")


def main():
    d = json.load(open(MIS))
    N = len(d)
    print(f"[load] mistake 池: {N} 题  ← {MIS}", flush=True)
    print(f"[keys] sample: {list(d[0].keys())}", flush=True)

    # 1) 自评: 池里存的 answer judge 一遍
    n_ok, n_none, n_real_wrong = 0, 0, 0
    for it in d:
        ans = it.get("answer", "")
        ref = str(it.get("ref_answer", ""))
        b = extract_boxed_content(ans or "")
        if b is None:
            n_none += 1
        elif normalize_answer(b) == normalize_answer(ref):
            n_ok += 1
        else:
            n_real_wrong += 1

    print("\n" + "=" * 70)
    print(f"池里存的 answer 自评 (静态, 不跑模型)")
    print("=" * 70)
    print(f"  判对 (boxed==ref)        : {n_ok}/{N} = {n_ok/N*100:.2f}%  ← 应=0%, 否则池脏")
    print(f"  没 boxed (截断/没收尾)   : {n_none}/{N} = {n_none/N*100:.2f}%")
    print(f"  真错 (有 boxed 但 != ref): {n_real_wrong}/{N} = {n_real_wrong/N*100:.2f}%")

    # 2) question_idx 唯一性
    qidx = [it.get("question_idx") for it in d]
    print(f"\n  question_idx unique: {len(set(qidx))}/{N}  ← 应=100% 无重复")

    # 3) 抽样 3 题
    print("\n--- 抽样 3 题 ---")
    for i, it in enumerate(d[:3]):
        ans = it.get("answer", "")
        ref = it.get("ref_answer", "")
        b = extract_boxed_content(ans or "")
        print(f"\n  题 {i}: q_idx={it.get('question_idx')} ref={ref!r}")
        print(f"    存的 answer boxed = {b!r}  len = {len(ans)} 字符")
        print(f"    存的 answer 尾100: {ans[-100:]!r}")

    print("\n" + "=" * 70)
    print("解读:")
    print("  自评 ≈0% → 池干净, 24.95% 是 take_exam 现在跑出来漂了 (与构造时不一致)")
    print("  自评 ≈25% → 池脏, 池里就混了 25% 'boxed==ref 但被收进 mistake' 的题")
    print("           → 阶段 1 重建池时 teacher_mark_paper 的判分逻辑有 bug")


if __name__ == "__main__":
    main()

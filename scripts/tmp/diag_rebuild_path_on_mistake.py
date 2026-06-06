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

    # 判分 (与 teacher_mark_paper 一致), 同时抽样落盘 (池里存的 vs 现在跑的)
    n_ok = 0
    n_none = 0
    n_real_wrong = 0
    by_qi = {r["question_idx"]: r["answer"] for r in res}

    # 每题 4 类: 池里没boxed-现在没boxed / 池里没boxed-现在有 / 池里有boxed-现在没 / 池里有boxed-现在有
    # 重点关注 "池里没boxed-现在有" (这就是 24.95% 的来源, 截断边界题)
    cat_NN, cat_NY, cat_YN, cat_YY = [], [], [], []
    flip_w2c, flip_c2w = [], []   # 池里 boxed!=ref / 现在 boxed==ref ; 反之

    for it in d:
        qi = it.get("question_idx")
        ans_now = by_qi.get(qi)
        if ans_now is None:
            continue
        ref = str(it["ref_answer"])
        ans_pool = it.get("answer", "")

        b_pool = extract_boxed_content(ans_pool or "")
        b_now = extract_boxed_content(ans_now or "")
        # 注意 extract_boxed_content 找不到时返回 "" (不是 None)
        has_pool = bool(b_pool)
        has_now = bool(b_now)

        # 判分 (现在跑出来的)
        j = judge(ans_now, ref)
        if j is None:   # 这分支基本走不到 (因为 "" 不是 None)
            n_none += 1
        elif j:
            n_ok += 1
        else:
            n_real_wrong += 1

        # 分类记录
        rec = {
            "qi": qi,
            "ref": ref,
            "len_pool": len(ans_pool),
            "len_now": len(ans_now),
            "boxed_pool": b_pool,
            "boxed_now": b_now,
        }
        if not has_pool and not has_now:
            cat_NN.append(rec)
        elif not has_pool and has_now:
            cat_NY.append(rec)
        elif has_pool and not has_now:
            cat_YN.append(rec)
        else:
            cat_YY.append(rec)

        # 翻转: 池里没做对 (boxed!=ref 或 没boxed) / 现在做对
        pool_correct = has_pool and normalize_answer(b_pool) == normalize_answer(ref)
        now_correct = has_now and normalize_answer(b_now) == normalize_answer(ref)
        if (not pool_correct) and now_correct:
            flip_w2c.append(rec)
        elif pool_correct and (not now_correct):
            flip_c2w.append(rec)

    N = n_ok + n_none + n_real_wrong

    print("\n" + "=" * 70)
    print(f"rebuild 算法路径重评 {N} mistake 池 Base")
    print("=" * 70)
    print(f"  做对 (boxed==ref)        : {n_ok}/{N} = {n_ok/N*100:.2f}%  ← 期望≈0% (rebuild 池定义)")
    print(f"  没 boxed (截断)          : {n_none}/{N} = {n_none/N*100:.2f}%")
    print(f"  真错 (有 boxed 但 != ref): {n_real_wrong}/{N} = {n_real_wrong/N*100:.2f}%")
    print("=" * 70)
    print(f"对照 diag_base_on_fill_pool: 344/1379 = 24.95%")

    # boxed 状态分类 (池里 vs 现在)
    print("\n池里 boxed vs 现在 boxed 状态分类")
    print("  N=没boxed, Y=有boxed   (池里, 现在)")
    print(f"  N→N (都没写完)  : {len(cat_NN)}")
    print(f"  N→Y (现在写完了): {len(cat_NY)}  ← 截断边界题, vLLM 多写了几 token")
    print(f"  Y→N (现在反而没): {len(cat_YN)}")
    print(f"  Y→Y (都有 boxed): {len(cat_YY)}")
    print(f"\n翻转: 池里错→现在对 = {len(flip_w2c)}; 池里对→现在错 = {len(flip_c2w)}")

    # 抽样: 从 N→Y 类 (做对那批) 抽 3 题打印, 重点看长度差
    print("\n--- N→Y 类抽样 3 题 (池里没 boxed, 现在有 boxed) ---")
    for r in cat_NY[:3]:
        # 找原题 + 现在跑出来的 answer
        ans_now = by_qi.get(r["qi"], "")
        ans_pool = next((it["answer"] for it in d if it.get("question_idx") == r["qi"]), "")
        print(f"\n  qi={r['qi']} ref={r['ref']!r}")
        print(f"    池里 answer len={r['len_pool']} 字符  boxed={r['boxed_pool']!r}")
        print(f"    现在 answer len={r['len_now']} 字符  boxed={r['boxed_now']!r}")
        print(f"    池里头200: {ans_pool[:200]!r}")
        print(f"    现在头200: {ans_now[:200]!r}")
        print(f"    头200相同?: {ans_pool[:200] == ans_now[:200]}")
        print(f"    池里尾100: {ans_pool[-100:]!r}")
        print(f"    现在尾100: {ans_now[-100:]!r}")
        # 找首次分叉的字符位置
        lim = min(len(ans_pool), len(ans_now))
        diff = next((i for i in range(lim) if ans_pool[i] != ans_now[i]), lim)
        print(f"    首次分叉在第 {diff} 字符 (池里 {len(ans_pool)} / 现在 {len(ans_now)})")

    print("\n" + "=" * 70)
    print("解读:")
    print("  本次 ≈0%   → rebuild 路径稳定, diag 路径有 bug")
    print("  本次 ≈25%  → 截断边界题假说: vLLM batch 间微小 token 漂移导致截断/不截断翻面")
    print("              N→Y 类多 → 现在多写了几 token 把 boxed 写完")
    print("  本次其他   → 再分析")


if __name__ == "__main__":
    main()

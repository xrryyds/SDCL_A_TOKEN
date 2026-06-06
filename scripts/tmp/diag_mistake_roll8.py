"""mistake 池 roll-8 评测 + 收集 unsolve_pool。

口径 (论文 S-GRPO baseline = pass@1 averaged over 8 trials):
  T=0.6, top_p=0.95, sample_n=8
  max_prompt_length=6144, max_new_tokens=4096 (与 4k 口径池构造对齐)

输入:
  datasets/exam/mistake_DS_MATH_pool.json (4k 口径池)

指标:
  pass@1 = 平均做对率 (sum(对次数 / 8) / N), 论文口径
  pass@8 = any@8 (8 次里任意一次做对的题占比)
  每题做对次数分布 (0/1/2/.../8)

输出:
  scripts/tmp/roll8_base_<TS>.jsonl                  (raw answers)
  datasets/exam/unsolve_pool.json                    (8 次全错的题, --collect_unsolved)
  stdout 表格

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_roll8.py --collect_unsolved
"""
import argparse, gc, json, os, sys, time
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam

MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
MIS = os.path.join(_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")
TMP_DIR = os.path.join(_ROOT, "scripts", "tmp")


def is_correct(ans: str, ref: str) -> bool:
    """boxed 字符串 normalize 比较 (与 teacher_mark_paper / eval_v3 一致)。"""
    b = extract_boxed_content(ans or "")
    if not b:
        return False
    return normalize_answer(b) == normalize_answer(str(ref))


def run_roll8(label, lora_path, mistake_path, max_prompt, max_new, n, T, top_p,
              device_ids, out_jsonl):
    print(f"\n{'='*70}\n[{label}] start  lora={lora_path}\n{'='*70}", flush=True)
    t0 = time.time()

    d = json.load(open(mistake_path))
    q = [it["question"] for it in d]
    ref = [str(it["ref_answer"]) for it in d]
    sol = [it.get("ref_solution", "") for it in d]
    qidx = [it.get("question_idx", i) for i, it in enumerate(d)]
    N = len(d)
    print(f"[{label}] 题数: {N}", flush=True)
    print(f"[{label}] cfg : max_prompt={max_prompt} max_new={max_new} n={n} T={T} top_p={top_p}", flush=True)

    kwargs = dict(model_path=MODEL, max_prompt_length=max_prompt, max_new_tokens=max_new)
    if lora_path:
        kwargs["use_lora"] = True
        kwargs["adapter_path"] = lora_path

    te = TakeExam(**kwargs)
    try:
        res = te.exam_multi_gpu(
            q, sol, ref, qidx,
            device_ids=device_ids, write_output=False,
            sample_n=n, temperature=T, top_p=top_p,
        )
    finally:
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 判分: 每题 8 次 → 对的次数
    n_correct_per_q = []
    rows = []
    for r in res:
        i = r["question_idx"]
        # 从原 d 里找 ref (res 里的 ref_answer 已经 strip 过, 但保险起见用 d)
        # res 与 d 顺序应一致, 用 by_qi 查
        samples = r.get("samples", [])
        ref_i = r["ref_answer"]
        n_ok = sum(1 for s in samples if is_correct(s, ref_i))
        n_correct_per_q.append(n_ok)
        rows.append({
            "question_idx": i,
            "ref_answer": ref_i,
            "n_correct_of_8": n_ok,
            "samples": samples,   # 留原文方便后查
        })

    # 指标
    pass1 = sum(c / n for c in n_correct_per_q) / N * 100
    pass_any = sum(1 for c in n_correct_per_q if c > 0) / N * 100
    dist = Counter(n_correct_per_q)

    # 落盘
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\n[{label}] 完成 ({elapsed:.0f}s = {elapsed/60:.1f} min)", flush=True)
    print(f"  pass@1 (avg of {n})   : {pass1:.2f}%   ← 论文口径")
    print(f"  pass@{n} (any)         : {pass_any:.2f}%")
    print(f"  做对次数分布:")
    for k in range(n + 1):
        cnt = dist.get(k, 0)
        print(f"    对 {k}/{n} 次 : {cnt:5d} ({cnt/N*100:.2f}%)")
    print(f"  raw → {out_jsonl}", flush=True)

    return {
        "label": label,
        "N": N,
        "pass1": pass1,
        "pass_any": pass_any,
        "dist": dict(dist),
        "elapsed_sec": elapsed,
        "out_jsonl": out_jsonl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mistake_path", type=str, default=MIS)
    ap.add_argument("--max_prompt_length", type=int, default=6144,
                    help="vLLM 总窗口 = 2048 prompt + 4096 gen")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--collect_unsolved", action="store_true",
                    help="8 次全错的题写到 datasets/exam/unsolve_pool.json")
    ap.add_argument("--unsolve_path", type=str,
                    default=os.path.join(_ROOT, "datasets", "exam", "unsolve_pool.json"))
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_jsonl = os.path.join(TMP_DIR, f"roll8_base_{ts}.jsonl")

    # 跑 Base
    s = run_roll8("Base", None, args.mistake_path, args.max_prompt_length,
                  args.max_new_tokens, args.n, args.temperature, args.top_p,
                  device_ids, base_jsonl)

    # 收集 unsolve_pool: 8 次全错的题
    if args.collect_unsolved:
        d_orig = json.load(open(args.mistake_path))
        # 从 base_jsonl 读 n_correct_of_8 == 0 的题 idx
        unsolved_qidx = set()
        with open(base_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("n_correct_of_8", 0) == 0:
                    unsolved_qidx.add(r["question_idx"])

        # 从原 mistake 池里捞出对应题 (含 ref_solution), 不带 samples
        unsolve_items = []
        for it in d_orig:
            qi = it.get("question_idx")
            if qi in unsolved_qidx:
                unsolve_items.append({
                    "question_idx": qi,
                    "question": it.get("question", ""),
                    "ref_answer": str(it.get("ref_answer", "")),
                    "ref_solution": it.get("ref_solution", ""),
                })

        os.makedirs(os.path.dirname(args.unsolve_path), exist_ok=True)
        with open(args.unsolve_path, "w", encoding="utf-8") as f:
            json.dump(unsolve_items, f, ensure_ascii=False, indent=2)
        print(f"\n[unsolve_pool] {len(unsolve_items)} 题 → {args.unsolve_path}", flush=True)

    # 汇总
    print("\n" + "=" * 78)
    print(f" mistake 池 roll-8 评测 (n={args.n}, T={args.temperature}, top_p={args.top_p})")
    print("=" * 78)
    print(f"  pass@1 (avg of {args.n}) : {s['pass1']:.2f}%   ← 论文口径")
    print(f"  pass@{args.n} (any)       : {s['pass_any']:.2f}%")
    print(f"  unsolved (0/{args.n} 对) : {s['dist'].get(0, 0)} 题")
    print("=" * 78)


if __name__ == "__main__":
    main()

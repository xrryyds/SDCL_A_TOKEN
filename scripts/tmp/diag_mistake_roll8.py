"""mistake 池 roll-8 评测 Base + LoRA。

口径 (论文 S-GRPO baseline = pass@1 averaged over 8 trials):
  T=0.6, top_p=0.95, sample_n=8
  max_prompt_length=10240, max_new_tokens=8192 (与池构造对齐)
  模型: DS R1-Distill-Qwen-7B Base + fillonly LoRA

输入:
  datasets/exam/mistake_DS_MATH_pool.json (当前 1394 题)

指标:
  pass@1 = 平均做对率 (sum(对次数 / 8) / N), 论文口径
  pass@8 = any@8 (8 次里任意一次做对的题占比)
  每题做对次数分布 (0/1/2/.../8)

输出:
  scripts/tmp/roll8_base_<TS>.jsonl
  scripts/tmp/roll8_lora_<TS>.jsonl
  stdout 表格 (Base / LoRA / Δ)

用法 (4 卡 H800):
  cd /workspace/SDCL_A_TOKEN
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_roll8.py \\
      --lora_path output/fillonly_4card_20260606_073155/checkpoint_epoch_2
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
    ap.add_argument("--lora_path", type=str, required=True,
                    help="fillonly LoRA checkpoint, e.g. output/fillonly_4card_20260606_073155/checkpoint_epoch_2")
    ap.add_argument("--mistake_path", type=str, default=MIS)
    ap.add_argument("--max_prompt_length", type=int, default=10240)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--skip_lora", action="store_true")
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    ts = time.strftime("%Y%m%d_%H%M%S")
    base_jsonl = os.path.join(TMP_DIR, f"roll8_base_{ts}.jsonl")
    lora_jsonl = os.path.join(TMP_DIR, f"roll8_lora_{ts}.jsonl")

    summaries = []
    if not args.skip_base:
        s = run_roll8("Base", None, args.mistake_path, args.max_prompt_length,
                      args.max_new_tokens, args.n, args.temperature, args.top_p,
                      device_ids, base_jsonl)
        summaries.append(s)
    if not args.skip_lora:
        s = run_roll8("LoRA", args.lora_path, args.mistake_path, args.max_prompt_length,
                      args.max_new_tokens, args.n, args.temperature, args.top_p,
                      device_ids, lora_jsonl)
        summaries.append(s)

    # 汇总表
    print("\n" + "=" * 78)
    print(" mistake 池 roll-8 评测汇总")
    print("=" * 78)
    print(f" {'Metric':<20} {'Base':>15} {'LoRA':>15} {'Δ':>15}")
    print(" " + "-" * 76)
    by = {s["label"]: s for s in summaries}
    base_s = by.get("Base")
    lora_s = by.get("LoRA")
    for metric, label in [("pass1", f"pass@1 avg of {args.n}"), ("pass_any", f"pass@{args.n} (any)")]:
        b = base_s[metric] if base_s else None
        l = lora_s[metric] if lora_s else None
        b_str = f"{b:.2f}%" if b is not None else "-"
        l_str = f"{l:.2f}%" if l is not None else "-"
        d_str = f"{l - b:+.2f}%" if (b is not None and l is not None) else "-"
        print(f" {label:<20} {b_str:>15} {l_str:>15} {d_str:>15}")
    print("=" * 78)


if __name__ == "__main__":
    main()

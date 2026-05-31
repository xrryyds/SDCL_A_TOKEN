"""MATH-500 greedy 前后对比脚本。

读 `output/eval_v3_<ts>/math500_base_greedy.jsonl` 与 `math500_lora_greedy.jsonl`,
按 question_idx 对齐, 分四类:
  - 变好  : Base 错 → LoRA 对
  - 变差  : Base 对 → LoRA 错
  - 都对  : 两个都对
  - 都错  : 两个都错 (再细分为 same_pred / diff_pred)

每条 jsonl 落盘:
  question_idx, question (从 Math_500 反查), ref_answer,
  base_pred / base_extracted / base_correct / base_len_chars
  lora_pred / lora_extracted / lora_correct / lora_len_chars
  category (improve/regress/both_correct/both_wrong_same/both_wrong_diff)

终端打印分类计数 + 平均生成长度对比。

用法:
  python scripts/compare_math500_base_vs_lora.py \
    --eval_dir output/eval_v3_<ts> \
    --output_dir output/eval_v3_<ts>/diff_math500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_math.MATH_500_data_util import Math_500


def load_jsonl(path: str) -> Dict[int, dict]:
    """按 question_idx 索引 jsonl。"""
    out: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            out[it["question_idx"]] = it
    return out


def classify(base_correct: bool, lora_correct: bool, base_pred: str, lora_pred: str) -> str:
    if (not base_correct) and lora_correct:
        return "improve"
    if base_correct and (not lora_correct):
        return "regress"
    if base_correct and lora_correct:
        return "both_correct"
    # both wrong
    if base_pred == lora_pred:
        return "both_wrong_same"
    return "both_wrong_diff"


def main():
    parser = argparse.ArgumentParser(description="MATH-500 base vs LoRA greedy 对比")
    parser.add_argument(
        "--eval_dir", type=str, required=True,
        help="eval_v3.py 的输出目录, 含 math500_base_greedy.jsonl / math500_lora_greedy.jsonl",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="对比结果落盘目录, 默认 <eval_dir>/diff_math500",
    )
    args = parser.parse_args()

    base_path = os.path.join(args.eval_dir, "math500_base_greedy.jsonl")
    lora_path = os.path.join(args.eval_dir, "math500_lora_greedy.jsonl")
    for p in [base_path, lora_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到文件: {p}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.eval_dir, "diff_math500")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] base = {base_path}")
    print(f"[load] lora = {lora_path}")
    base_items = load_jsonl(base_path)
    lora_items = load_jsonl(lora_path)
    print(f"[load] base n={len(base_items)}  lora n={len(lora_items)}")

    # 反查 MATH-500 question 原文
    print("[load] Math_500 ...")
    m500 = Math_500()
    questions: List[str] = m500.problems
    print(f"[load] Math_500 n={len(questions)}")

    # 对齐
    common_idx = sorted(set(base_items.keys()) & set(lora_items.keys()))
    only_base = sorted(set(base_items.keys()) - set(lora_items.keys()))
    only_lora = sorted(set(lora_items.keys()) - set(base_items.keys()))
    if only_base or only_lora:
        print(f"[warn] only-base idx count = {len(only_base)}, only-lora idx count = {len(only_lora)}")

    by_cat: Dict[str, List[dict]] = {
        "improve": [], "regress": [], "both_correct": [],
        "both_wrong_same": [], "both_wrong_diff": [],
    }
    base_len_sum, lora_len_sum = 0, 0

    for idx in common_idx:
        b = base_items[idx]
        l = lora_items[idx]
        base_pred = b["pred_extracted"]
        lora_pred = l["pred_extracted"]
        base_correct = bool(b["correct"])
        lora_correct = bool(l["correct"])
        base_raw = b["pred_raw"]
        lora_raw = l["pred_raw"]
        base_len = len(base_raw)
        lora_len = len(lora_raw)
        base_len_sum += base_len
        lora_len_sum += lora_len

        cat = classify(base_correct, lora_correct, base_pred, lora_pred)
        q = questions[idx] if idx < len(questions) else ""

        rec = {
            "question_idx": idx,
            "question": q,
            "ref_answer": b["ref_answer"],
            "category": cat,
            "base_extracted": base_pred,
            "lora_extracted": lora_pred,
            "base_correct": base_correct,
            "lora_correct": lora_correct,
            "base_len_chars": base_len,
            "lora_len_chars": lora_len,
            "len_delta": lora_len - base_len,
            "base_pred_raw": base_raw,
            "lora_pred_raw": lora_raw,
        }
        by_cat[cat].append(rec)

    # ============ 落盘 ============
    for cat, items in by_cat.items():
        path = os.path.join(args.output_dir, f"{cat}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[write] {cat:<18}  n={len(items):>4}  →  {path}")

    # 落盘 summary.json
    n_total = len(common_idx)
    summary = {
        "eval_dir": args.eval_dir,
        "output_dir": args.output_dir,
        "n_common": n_total,
        "n_only_base": len(only_base),
        "n_only_lora": len(only_lora),
        "counts": {k: len(v) for k, v in by_cat.items()},
        "base_acc": sum(1 for i in common_idx if base_items[i]["correct"]) / n_total if n_total else 0.0,
        "lora_acc": sum(1 for i in common_idx if lora_items[i]["correct"]) / n_total if n_total else 0.0,
        "avg_len_base": base_len_sum / n_total if n_total else 0.0,
        "avg_len_lora": lora_len_sum / n_total if n_total else 0.0,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ============ 终端打印 ============
    print()
    print("=" * 72)
    print(f" MATH-500 Base vs LoRA  对比汇总  (n_common = {n_total})")
    print("=" * 72)
    print(f"  Base 准确率 : {summary['base_acc']*100:.2f}%")
    print(f"  LoRA 准确率 : {summary['lora_acc']*100:.2f}%")
    print(f"  Δ          : {(summary['lora_acc']-summary['base_acc'])*100:+.2f}%")
    print()
    print(" 分类计数:")
    for cat in ["improve", "regress", "both_correct", "both_wrong_same", "both_wrong_diff"]:
        n = len(by_cat[cat])
        pct = n / n_total * 100 if n_total else 0.0
        label_map = {
            "improve": "Base错→LoRA对 (变好)",
            "regress": "Base对→LoRA错 (变差)",
            "both_correct": "都对",
            "both_wrong_same": "都错·同pred",
            "both_wrong_diff": "都错·不同pred",
        }
        print(f"  {label_map[cat]:<24}  {n:>4}  ({pct:5.2f}%)")
    print()
    print(" 生成长度 (chars):")
    print(f"  avg Base  = {summary['avg_len_base']:.1f}")
    print(f"  avg LoRA  = {summary['avg_len_lora']:.1f}")
    print(f"  Δ         = {summary['avg_len_lora']-summary['avg_len_base']:+.1f}")
    print()
    print(f" 详细 jsonl + summary.json 见: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()

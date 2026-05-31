"""把 LoRA 首 token 稀有的那几题 (count==1 那批) 映射到 5 分类。

输入:
  - <eval_dir>/first_token/lora_rows.jsonl      (来自 first_token_stats.py)
  - <eval_dir>/diff_math500/{improve,regress,both_correct,both_wrong_same,both_wrong_diff}.jsonl
    (来自 compare_math500_base_vs_lora.py)

逻辑:
  - 从 lora_rows.jsonl 找出首 token count <= --max_count 的所有题 (默认 1, 即唯一题)
  - 对每个 question_idx, 在 5 个分类 jsonl 里找出它落在哪一类
  - 终端打印 (idx, first_token_text, category, base_extracted, lora_extracted, ref_answer)

用法:
  python scripts/locate_rare_first_token.py --eval_dir output/eval_v3_<ts>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser(description="把 LoRA 稀有首 token 的题映射到 5 分类")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument(
        "--max_count", type=int, default=1,
        help="只看首 token 出现次数 <= max_count 的题 (默认 1, 即唯一题)",
    )
    args = parser.parse_args()

    ft_path = os.path.join(args.eval_dir, "first_token", "lora_rows.jsonl")
    diff_dir = os.path.join(args.eval_dir, "diff_math500")
    if not os.path.isfile(ft_path):
        raise FileNotFoundError(f"先跑 first_token_stats.py 生成 {ft_path}")
    if not os.path.isdir(diff_dir):
        raise FileNotFoundError(f"先跑 compare_math500_base_vs_lora.py 生成 {diff_dir}")

    # 1) 加载 lora_rows, 统计每个 first_token 的题数
    ft_rows = load_jsonl(ft_path)
    tid_count: Counter = Counter(r["first_token_id"] for r in ft_rows)
    rare_tids = {tid for tid, c in tid_count.items() if c <= args.max_count}
    rare_idx_to_meta: Dict[int, dict] = {}
    for r in ft_rows:
        if r["first_token_id"] in rare_tids:
            rare_idx_to_meta[r["question_idx"]] = r

    print(f"[load] lora_rows n={len(ft_rows)}, 稀有 token (count<={args.max_count}) "
          f"共 {len(rare_tids)} 种 → {len(rare_idx_to_meta)} 题")

    # 2) 加载 5 个分类, 建 idx → category 映射, 同时存详情
    cat_files = {
        "improve": "improve.jsonl",
        "regress": "regress.jsonl",
        "both_correct": "both_correct.jsonl",
        "both_wrong_same": "both_wrong_same.jsonl",
        "both_wrong_diff": "both_wrong_diff.jsonl",
    }
    idx_to_cat: Dict[int, str] = {}
    idx_to_diff: Dict[int, dict] = {}
    for cat, fname in cat_files.items():
        items = load_jsonl(os.path.join(diff_dir, fname))
        for it in items:
            idx_to_cat[it["question_idx"]] = cat
            idx_to_diff[it["question_idx"]] = it

    # 3) 逐题打印
    by_cat: Dict[str, List[int]] = defaultdict(list)
    print()
    print("=" * 110)
    print(" 稀有首 token 题  → 5 分类映射")
    print("=" * 110)
    print(f" {'idx':>4}  {'first_tok':<14}  {'category':<18}  {'ref':<14}  {'base_pred':<14}  {'lora_pred':<14}")
    print(" " + "-" * 106)
    for idx in sorted(rare_idx_to_meta.keys()):
        meta = rare_idx_to_meta[idx]
        ft_text = meta["first_token_text_repr"]
        cat = idx_to_cat.get(idx, "?")
        diff = idx_to_diff.get(idx, {})
        ref = (diff.get("ref_answer") or "")[:14]
        bp = (diff.get("base_extracted") or "")[:14]
        lp = (diff.get("lora_extracted") or "")[:14]
        print(f" {idx:>4}  {ft_text:<14}  {cat:<18}  {ref:<14}  {bp:<14}  {lp:<14}")
        by_cat[cat].append(idx)

    # 4) 分类汇总
    print()
    print(" 分类汇总:")
    for cat in ["improve", "regress", "both_correct", "both_wrong_same", "both_wrong_diff"]:
        idxs = by_cat.get(cat, [])
        print(f"  {cat:<18}  n={len(idxs):<3}  idx={idxs}")
    print("=" * 110)


if __name__ == "__main__":
    main()

"""统计 LoRA 在 MATH-500 上 greedy 预测的首 token 分布。

口径与训练对齐 (a_token_sdcl_train.py 第 217-225 行):
  - 用 tokenizer encode pred_raw (add_special_tokens=False), 取第 1 个 token id
  - 同时输出 token text (tokenizer.decode([tid])) + 该 token 的题数与占比

可选: 加 --include_base 同时统计 Base, 输出对比。

用法:
  python scripts/first_token_stats.py \
    --eval_dir output/eval_v3_<ts> \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def stats_first_tokens(
    items: List[dict],
    tokenizer,
) -> Tuple[Counter, List[Tuple[int, int, str, str, int]]]:
    """返回 (Counter[token_id], rows: list of (idx, tid, ttext_repr, pred_head, pred_len))。"""
    cnt: Counter = Counter()
    rows = []
    for it in items:
        idx = it["question_idx"]
        pred = it.get("pred_raw", "")
        if not pred:
            tid = -1
            ttext = "<EMPTY>"
        else:
            ids = tokenizer.encode(pred, add_special_tokens=False)
            if len(ids) == 0:
                tid = -1
                ttext = "<EMPTY_AFTER_TOKENIZE>"
            else:
                tid = int(ids[0])
                ttext = tokenizer.decode([tid])
        cnt[tid] += 1
        rows.append((idx, tid, repr(ttext), pred[:40].replace("\n", "\\n"), len(pred)))
    return cnt, rows


def print_top_table(name: str, cnt: Counter, tokenizer, total: int, top_n: int = 30):
    print()
    print("=" * 84)
    print(f" {name}  首 token 分布  (n_total={total}, unique={len(cnt)})")
    print("=" * 84)
    print(f" {'rank':>4}  {'tid':>7}  {'count':>5}  {'pct':>7}  token_text")
    print(" " + "-" * 80)
    for i, (tid, c) in enumerate(cnt.most_common(top_n), 1):
        if tid == -1:
            ttext = "<EMPTY>"
        else:
            ttext = repr(tokenizer.decode([tid]))
        pct = c / total * 100
        print(f" {i:>4}  {tid:>7}  {c:>5}  {pct:>6.2f}%  {ttext}")
    if len(cnt) > top_n:
        rest = sum(c for _, c in cnt.most_common()[top_n:])
        print(f" {'...':>4}  {'rest':>7}  {rest:>5}  {rest/total*100:>6.2f}%  (其它 {len(cnt)-top_n} 个 token)")


def print_compare(base_cnt: Counter, lora_cnt: Counter, tokenizer, total: int, top_n: int = 30):
    """Base vs LoRA 同表对比, 取并集 top_n。"""
    union = set(base_cnt) | set(lora_cnt)
    rows = []
    for tid in union:
        b = base_cnt.get(tid, 0)
        l = lora_cnt.get(tid, 0)
        rows.append((tid, b, l, l - b))
    rows.sort(key=lambda r: max(r[1], r[2]), reverse=True)

    print()
    print("=" * 96)
    print(f" Base vs LoRA  首 token 对比  (n_total={total}, union_unique={len(union)})")
    print("=" * 96)
    print(f" {'tid':>7}  {'base':>5}  {'lora':>5}  {'Δ':>6}  {'base%':>6}  {'lora%':>6}  token_text")
    print(" " + "-" * 92)
    for tid, b, l, d in rows[:top_n]:
        if tid == -1:
            ttext = "<EMPTY>"
        else:
            ttext = repr(tokenizer.decode([tid]))
        print(f" {tid:>7}  {b:>5}  {l:>5}  {d:+6d}  {b/total*100:>5.2f}%  {l/total*100:>5.2f}%  {ttext}")


def main():
    parser = argparse.ArgumentParser(description="MATH-500 首 token 分布统计")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str,
        default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument(
        "--include_base", action="store_true",
        help="同时统计 Base 并打印 Base vs LoRA 对比表",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="详细 jsonl 落盘目录, 默认 <eval_dir>/first_token",
    )
    args = parser.parse_args()

    lora_path = os.path.join(args.eval_dir, "math500_lora_greedy.jsonl")
    base_path = os.path.join(args.eval_dir, "math500_base_greedy.jsonl")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(lora_path)
    if args.include_base and not os.path.isfile(base_path):
        raise FileNotFoundError(base_path)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.eval_dir, "first_token")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] tokenizer ← {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[load] {lora_path}")
    lora_items = load_jsonl(lora_path)
    print(f"[load] lora n={len(lora_items)}")

    lora_cnt, lora_rows = stats_first_tokens(lora_items, tokenizer)

    # 落盘 LoRA 详情
    rows_path = os.path.join(args.output_dir, "lora_rows.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for idx, tid, ttext_repr, head, plen in lora_rows:
            f.write(json.dumps({
                "question_idx": idx,
                "first_token_id": tid,
                "first_token_text_repr": ttext_repr,
                "pred_head40": head,
                "pred_len_chars": plen,
            }, ensure_ascii=False) + "\n")
    print(f"[write] LoRA 逐题首 token → {rows_path}")

    # 落盘 LoRA 计数表
    cnt_path = os.path.join(args.output_dir, "lora_counts.json")
    with open(cnt_path, "w", encoding="utf-8") as f:
        cnt_dump = []
        for tid, c in lora_cnt.most_common():
            cnt_dump.append({
                "token_id": tid,
                "token_text": tokenizer.decode([tid]) if tid != -1 else "<EMPTY>",
                "count": c,
                "pct": c / len(lora_items) * 100,
            })
        json.dump({"n_total": len(lora_items), "unique": len(lora_cnt), "tokens": cnt_dump},
                  f, ensure_ascii=False, indent=2)
    print(f"[write] LoRA 计数表 → {cnt_path}")

    print_top_table("LoRA", lora_cnt, tokenizer, len(lora_items), args.top_n)

    if args.include_base:
        print(f"[load] {base_path}")
        base_items = load_jsonl(base_path)
        print(f"[load] base n={len(base_items)}")
        base_cnt, _ = stats_first_tokens(base_items, tokenizer)

        # 落盘 Base 计数表
        cnt_path_b = os.path.join(args.output_dir, "base_counts.json")
        with open(cnt_path_b, "w", encoding="utf-8") as f:
            cnt_dump = []
            for tid, c in base_cnt.most_common():
                cnt_dump.append({
                    "token_id": tid,
                    "token_text": tokenizer.decode([tid]) if tid != -1 else "<EMPTY>",
                    "count": c,
                    "pct": c / len(base_items) * 100,
                })
            json.dump({"n_total": len(base_items), "unique": len(base_cnt), "tokens": cnt_dump},
                      f, ensure_ascii=False, indent=2)
        print(f"[write] Base 计数表 → {cnt_path_b}")

        print_top_table("Base", base_cnt, tokenizer, len(base_items), args.top_n)
        print_compare(base_cnt, lora_cnt, tokenizer, len(lora_items), args.top_n)

    print()


if __name__ == "__main__":
    main()

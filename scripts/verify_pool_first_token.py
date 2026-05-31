"""验证: pool 数据集 LoRA 答对的 64 题, 首 token 是否在 pool 候选里。

数据:
  - <eval_dir>/pool_lora_greedy.jsonl     (来自 eval_v3.py)
  - <eval_dir>/pool_base_greedy.jsonl
  - datasets/first_tokens_test.json       (pool 候选 376 token)
  - datasets/exam/fill_multi_pool.json    (训练数据, 含每题对应的 fill_token_id)

输出:
  对 LoRA 答对的题, 统计:
    1) LoRA 首 token 是否∈ pool 候选 376
    2) LoRA 首 token 是否 == 训练时该题的 fill_token_id
    3) LoRA 首 token 是不是 'Okay'/'Alright' (R1-Distill 默认起手词)
  按 4 类拆: only_lora_correct (improve), only_base_correct, both_correct, both_wrong

用法:
  python scripts/verify_pool_first_token.py \
    --eval_dir output/eval_v3_<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set

from transformers import AutoTokenizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str,
        default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--first_token_pool_path", type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "first_tokens_test.json"),
    )
    parser.add_argument(
        "--fill_multi_pool_path", type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json"),
    )
    args = parser.parse_args()

    pool_lora_path = os.path.join(args.eval_dir, "pool_lora_greedy.jsonl")
    pool_base_path = os.path.join(args.eval_dir, "pool_base_greedy.jsonl")
    for p in [pool_lora_path, pool_base_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    print(f"[load] tokenizer ← {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 1) pool 候选 token 集合
    print(f"[load] first_token_pool ← {args.first_token_pool_path}")
    with open(args.first_token_pool_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cand_list = d["tokens"] if isinstance(d, dict) else d
    cand_tids: Set[int] = set(int(c["token_id"]) for c in cand_list)
    cand_tid_to_text = {int(c["token_id"]): c["token_text"] for c in cand_list}
    print(f"[load] pool 候选 token: {len(cand_tids)} 种")

    # 2) 训练数据 fill_multi_pool: 该 idx 训练时用的 fill_token_id (一题可能多条)
    print(f"[load] fill_multi_pool ← {args.fill_multi_pool_path}")
    with open(args.fill_multi_pool_path, "r", encoding="utf-8") as f:
        fill_pool = json.load(f)
    # fill_multi_pool.json 里 question 顺序应与 pool eval 的 question_idx 对齐
    # 但每题可能有多条 fill 救回; 这里收集 q_text -> set(fill_tid)
    train_q_to_fills: Dict[str, Set[int]] = defaultdict(set)
    for it in fill_pool:
        q = it.get("question", "")
        ftid = it.get("fill_token_id")
        if q and ftid is not None:
            train_q_to_fills[q].add(int(ftid))
    print(f"[load] 训练 pool: {len(fill_pool)} 条, 覆盖 {len(train_q_to_fills)} 题")

    # 3) 加载 eval 输出 (pool 数据集 500 题)
    print(f"[load] {pool_lora_path}")
    lora_items = load_jsonl(pool_lora_path)
    base_items = load_jsonl(pool_base_path)
    print(f"[load] pool eval n: lora={len(lora_items)} base={len(base_items)}")

    base_by_idx = {it["question_idx"]: it for it in base_items}

    # eval pool 数据是按 fill_multi_pool 顺序 dedup 后取 question 一次, 但 question_idx
    # 是 pool 数据集里的 idx (load_pool: list(range(len(data)))), 而不是 fill_multi_pool 索引。
    # 因此用 question_idx 反查 question 文本: 重新读 fill_multi_pool 顺序里 dedup 后的 idx → q
    # 注: load_pool() 里读的是 datasets/exam/fill_multi_pool.json 同一个文件
    #     idx i 对应 fill_pool[i]["question"], 但 fill_pool 一题多条; 取每条的 q 即可
    pool_idx_to_q: Dict[int, str] = {}
    for i, it in enumerate(fill_pool):
        pool_idx_to_q[i] = it.get("question", "")

    # 4) 对 LoRA 每题, 计算首 token, 分类
    rows = []
    for it in lora_items:
        idx = it["question_idx"]
        pred = it.get("pred_raw", "") or ""
        if not pred:
            tid = -1
        else:
            ids = tokenizer.encode(pred, add_special_tokens=False)
            tid = int(ids[0]) if ids else -1
        ttext = tokenizer.decode([tid]) if tid >= 0 else "<EMPTY>"
        in_cand = (tid in cand_tids)
        # 该题训练时是否被 fill_multi_pool 覆盖 + 是否命中训练用过的 fill_token_id
        q = pool_idx_to_q.get(idx, "")
        train_fills = train_q_to_fills.get(q, set())
        match_train = (tid in train_fills) if train_fills else False
        is_okay_alright = (ttext.strip() in ("Okay", "Alright"))
        rows.append({
            "idx": idx,
            "lora_correct": bool(it["correct"]),
            "base_correct": bool(base_by_idx[idx]["correct"]) if idx in base_by_idx else False,
            "tid": tid,
            "ttext": ttext,
            "in_cand_pool": in_cand,
            "n_train_fills_for_q": len(train_fills),
            "match_train_fill": match_train,
            "is_okay_alright": is_okay_alright,
        })

    # 5) 4 分类汇总
    def bucket(r):
        if r["lora_correct"] and not r["base_correct"]:
            return "improve"
        if r["base_correct"] and not r["lora_correct"]:
            return "regress"
        if r["lora_correct"] and r["base_correct"]:
            return "both_correct"
        return "both_wrong"

    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[bucket(r)].append(r)

    print()
    print("=" * 92)
    print(" pool 数据集 LoRA 首 token 验证  (n_total={})".format(len(rows)))
    print("=" * 92)
    print(f" {'bucket':<14} {'n':>4} {'in_cand_pool':>14} {'match_train':>14} {'OKAY/Alright':>14}")
    print(" " + "-" * 88)
    for b in ["improve", "regress", "both_correct", "both_wrong"]:
        items = by_bucket.get(b, [])
        n = len(items)
        if n == 0:
            print(f" {b:<14} {n:>4} {'-':>14} {'-':>14} {'-':>14}")
            continue
        n_in = sum(1 for r in items if r["in_cand_pool"])
        n_match = sum(1 for r in items if r["match_train_fill"])
        n_okay = sum(1 for r in items if r["is_okay_alright"])
        print(
            f" {b:<14} {n:>4} "
            f"{n_in:>4} ({n_in/n*100:5.1f}%) "
            f"{n_match:>4} ({n_match/n*100:5.1f}%) "
            f"{n_okay:>4} ({n_okay/n*100:5.1f}%)"
        )

    # improve 桶逐题首 token Top
    if by_bucket["improve"]:
        print()
        print(" [improve 桶] LoRA 首 token 频次 Top 10:")
        cnt = Counter((r["ttext"], r["tid"]) for r in by_bucket["improve"])
        for (ttext, tid), c in cnt.most_common(10):
            mark_cand = "✓pool" if tid in cand_tids else " not"
            print(f"   {repr(ttext):<14} tid={tid:<7} n={c:<4} {mark_cand}")

    print()
    print(" [improve 桶] 全样本 first token + 是否在 pool 候选里:")
    for r in by_bucket["improve"][:50]:
        print(f"   idx={r['idx']:<4} tid={r['tid']:<7} {repr(r['ttext']):<14} "
              f"in_pool={r['in_cand_pool']!s:<5} match_train={r['match_train_fill']!s}")
    if len(by_bucket["improve"]) > 50:
        print(f"   ... 还有 {len(by_bucket['improve'])-50} 条")
    print()


if __name__ == "__main__":
    main()

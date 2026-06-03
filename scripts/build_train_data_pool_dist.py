"""build_train_data_pool_dist.py — 展开 fill_pool + roll → pool_dist 训练数据

数据来源:
  pool : datasets/exam/fill_multi_pool.json       (500 题 × avg 48.6 candidates)
  roll : datasets/exam/fill_multi_pool_roll.json  (1296 题 × avg 1.65 candidates)

每题做对的 N 个 candidate 全部展开为 N 条独立训练样本。
**关键新字段** target_token_ids: 该题所有对的 candidate 的 token_id 列表
(同一题内所有样本共享,首 token 目标分布 = 在 target_token_ids 上 uniform 1/N)。

Schema(对齐新 trainer a_token_pool_dist_train.py:_load_train_data / _encode_sample):
  {
    "source": "pool_dist",
    "question": str,
    "answer":   str,                # 整段对答案(含首 token)
    "fill_token_id":   int,         # 该样本的首 token id (∈ target_token_ids)
    "fill_token_text": str,
    "target_token_ids": List[int],  # 该题所有对的 candidate 的 token_id (uniform 目标分布支撑集)
    "question_idx":    int,
    "ref_answer":      str,
    "origin_pool":     "pool" | "roll",   # 来源池(诊断用)
  }

Loss 形式(由 trainer 实现, 本脚本只准备数据):
  首 token 位置: KL(student || target_dist)
                target_dist[v] = 1/N if v in target_token_ids else 0
  后续 answer:   KL(student || teacher)   (teacher 现跑, base 不挂 LoRA)
"""

import argparse
import json
import logging
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_POOL_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json"
)
DEFAULT_ROLL_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_pool_roll.json"
)
DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "train", "train_data_pool_dist.json"
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expand(items, origin_pool):
    """把一个池(pool 或 roll)展开为 pool_dist 样本列表。

    一题内所有对的 candidate token_id 收集为 target_token_ids,该题展开成 N 条样本,
    每条样本带相同的 target_token_ids,但 fill_token_id 不同(自己的首 token)。
    """
    out = []
    n_q_total = 0
    n_q_kept = 0
    n_skipped_no_ttxt = 0
    n_skipped_no_ans = 0
    for item in items:
        n_q_total += 1
        q = item.get("question", "")
        ref = str(item.get("ref_answer", ""))
        qidx = item.get("question_idx", -1)
        cands = item.get("candidates", [])
        if not q or not cands:
            continue

        # 收集该题所有对的 candidate 的 token_id (去重保稳定顺序)
        seen = set()
        target_ids = []
        valid_cands = []
        for cand in cands:
            tid = cand.get("token_id")
            ttxt = cand.get("token_text", "")
            ans = cand.get("answer", "")
            if tid is None:
                continue
            if not ttxt:
                n_skipped_no_ttxt += 1
                continue
            if not ans:
                n_skipped_no_ans += 1
                continue
            tid_int = int(tid)
            if tid_int not in seen:
                seen.add(tid_int)
                target_ids.append(tid_int)
            valid_cands.append((tid_int, ttxt, ans))

        if not valid_cands:
            continue
        n_q_kept += 1

        for tid_int, ttxt, ans in valid_cands:
            out.append({
                "source": "pool_dist",
                "question": q,
                "answer": ans,
                "fill_token_id": tid_int,
                "fill_token_text": ttxt,
                "target_token_ids": list(target_ids),  # 该题共享
                "question_idx": qidx,
                "ref_answer": ref,
                "origin_pool": origin_pool,
            })
    logger.info(
        "  [%s] 题数: 总 %d → 保留 %d; 样本 %d; 跳过 no_ttxt=%d no_ans=%d",
        origin_pool, n_q_total, n_q_kept, len(out),
        n_skipped_no_ttxt, n_skipped_no_ans,
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_path", type=str, default=DEFAULT_POOL_PATH)
    parser.add_argument("--roll_path", type=str, default=DEFAULT_ROLL_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    logger.info("加载两池...")
    pool = _load_json(args.pool_path)
    roll = _load_json(args.roll_path)
    logger.info("  pool: %d 题 from %s", len(pool), args.pool_path)
    logger.info("  roll: %d 题 from %s", len(roll), args.roll_path)

    pool_samples = _expand(pool, "pool")
    roll_samples = _expand(roll, "roll")

    merged = pool_samples + roll_samples
    total = len(merged)
    logger.info("展开后样本数: pool=%d roll=%d total=%d",
                len(pool_samples), len(roll_samples), total)

    # 统计 target_token_ids 大小分布
    ts_sizes = sorted({(s["question_idx"], s["origin_pool"], len(s["target_token_ids"]))
                       for s in merged}, key=lambda x: x[2])
    sizes = [t[2] for t in ts_sizes]
    if sizes:
        logger.info(
            "target_token_ids 大小: min=%d max=%d avg=%.2f median=%d (按题去重统计 %d 题)",
            sizes[0], sizes[-1], sum(sizes) / len(sizes),
            sizes[len(sizes) // 2], len(sizes),
        )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 抽样核对
    logger.info("=== 抽样核对 ===")
    for origin in ("pool", "roll"):
        s = next((x for x in merged if x["origin_pool"] == origin), None)
        if s is None:
            logger.warning("  [%s] 0 样本!", origin)
            continue
        ans_head = s["answer"][:60].replace("\n", " ")
        logger.info(
            "  [%s] q_idx=%s fill_tok=(%d, %r) |target|=%d ans_starts_with_tok=%s\n"
            "        target_token_ids[:8]=%s answer[:60]=%r",
            origin, s["question_idx"], s["fill_token_id"], s["fill_token_text"],
            len(s["target_token_ids"]),
            s["answer"].startswith(s["fill_token_text"]),
            s["target_token_ids"][:8], ans_head,
        )

    logger.info("写出: %s (%d 样本)", args.out_path, total)
    return total


if __name__ == "__main__":
    main()

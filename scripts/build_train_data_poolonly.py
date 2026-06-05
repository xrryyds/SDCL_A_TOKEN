"""build_train_data_poolonly.py — 纯 pool 单池 (V3 简化版)

只用 fill 救回的 pool 池, 每题随机抽 N 个 candidate (默认 3), 不要 corr/roll。
loss 复用 V3 pool 口径 (trainer 实现): 首位 CE on fill_token_id + 其余反向 KL。

来源:
  pool : datasets/exam/fill_multi_pool.json  (每题 candidates=[{token_id, token_text, answer}])

输出 schema (对齐 a_token_sdcl_train.py 的 pool 样本):
  {
    "source": "pool",
    "question": str,
    "answer":   str,            # 整段对答案 (含首 token)
    "fill_token_id":   int,
    "fill_token_text": str,
    "question_idx":    int,
    "ref_answer":      str,
  }

用法:
  python scripts/build_train_data_poolonly.py \
    --pool_path datasets/exam/fill_multi_pool.json \
    --out_path datasets/train/train_data_poolonly.json \
    --n_per_q 3 --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_POOL_PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json")
DEFAULT_OUT_PATH = os.path.join(_PROJECT_ROOT, "datasets", "train", "train_data_poolonly.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expand_pool(pool_items, n_per_q, rng):
    out = []
    n_skipped_no_ttxt = 0
    n_lt_n = 0
    for item in pool_items:
        q = item.get("question", "")
        ref = str(item.get("ref_answer", ""))
        qidx = item.get("question_idx", -1)
        # 先过滤出合法 candidate (有 token_text + answer + token_id)
        valid = []
        for cand in item.get("candidates", []):
            ans = cand.get("answer", "")
            tid = cand.get("token_id")
            ttxt = cand.get("token_text", "")
            if not q or not ans or tid is None:
                continue
            if not ttxt:
                n_skipped_no_ttxt += 1
                continue
            valid.append((int(tid), ttxt, ans))
        if not valid:
            continue
        # 每题随机抽 n_per_q 个 (不足则全要)
        if len(valid) <= n_per_q:
            chosen = valid
            if len(valid) < n_per_q:
                n_lt_n += 1
        else:
            chosen = rng.sample(valid, n_per_q)
        for tid, ttxt, ans in chosen:
            out.append({
                "source": "pool",
                "question": q,
                "answer": ans,
                "fill_token_id": tid,
                "fill_token_text": ttxt,
                "question_idx": qidx,
                "ref_answer": ref,
            })
    if n_skipped_no_ttxt > 0:
        logger.info("跳过 %d 条缺 token_text 的 candidate", n_skipped_no_ttxt)
    if n_lt_n > 0:
        logger.info("%d 题 candidate 数 < n_per_q, 全取", n_lt_n)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_path", type=str, default=DEFAULT_POOL_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--n_per_q", type=int, default=3, help="每题随机抽几个 candidate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("加载 pool ← %s", args.pool_path)
    pool = _load_json(args.pool_path)
    logger.info("  pool: %d 题", len(pool))

    samples = _expand_pool(pool, args.n_per_q, rng)
    logger.info("展开后样本数: %d (题数 %d × ≤%d)", len(samples), len(pool), args.n_per_q)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("写出 → %s", args.out_path)

    # 抽样核对
    if samples:
        s = samples[0]
        logger.info("=== 抽样核对 (第 1 条) ===")
        logger.info("  source=%s fill_token_id=%s fill_token_text=%r",
                    s["source"], s["fill_token_id"], s["fill_token_text"])
        logger.info("  question 前80: %r", s["question"][:80])
        logger.info("  answer 前80: %r", s["answer"][:80])


if __name__ == "__main__":
    main()

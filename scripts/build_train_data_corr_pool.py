"""build_train_data_corr_pool.py — corr + pool 两池 (poolonly 加 corr 锚定版)

poolonly 只训 pool 导致灾难性遗忘 (corr -11.58%, math -8%), 加回 corr 池锚定。
去掉 roll 池。

  corr : datasets/exam/corr_DS_MATH_pool.json   每题 1 条, source=corr_answer
         loss = 全段反向 KL(student||teacher)
  pool : datasets/exam/fill_multi_pool.json      每题随机 N 个 candidate (默认3)
         loss = 首位 CE on fill_token_id + 其余反向 KL

输出 schema 对齐 a_token_sdcl_train.py:
  corr_answer: {source, question, answer, question_idx, ref_answer}
  pool:        {source, question, answer, fill_token_id, fill_token_text, question_idx, ref_answer}

用法:
  python scripts/build_train_data_corr_pool.py \
    --corr_path datasets/exam/corr_DS_MATH_pool.json \
    --pool_path datasets/exam/fill_multi_pool.json \
    --out_path datasets/train/train_data_corr_pool.json \
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

DEFAULT_CORR_PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "corr_DS_MATH_pool.json")
DEFAULT_POOL_PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json")
DEFAULT_OUT_PATH = os.path.join(_PROJECT_ROOT, "datasets", "train", "train_data_corr_pool.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expand_corr(corr_items):
    out = []
    for item in corr_items:
        q = item.get("question", "")
        a = item.get("answer", "")
        if not q or not a:
            continue
        out.append({
            "source": "corr_answer",
            "question": q,
            "answer": a,
            "question_idx": item.get("question_idx", -1),
            "ref_answer": str(item.get("ref_answer", "")),
        })
    return out


def _expand_pool(pool_items, n_per_q, rng):
    out = []
    n_skipped_no_ttxt = 0
    n_lt_n = 0
    for item in pool_items:
        q = item.get("question", "")
        ref = str(item.get("ref_answer", ""))
        qidx = item.get("question_idx", -1)
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
        logger.info("pool 跳过 %d 条缺 token_text 的 candidate", n_skipped_no_ttxt)
    if n_lt_n > 0:
        logger.info("pool %d 题 candidate 数 < n_per_q, 全取", n_lt_n)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr_path", type=str, default=DEFAULT_CORR_PATH)
    parser.add_argument("--pool_path", type=str, default=DEFAULT_POOL_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--n_per_q", type=int, default=3, help="pool 每题随机抽几个 candidate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("加载 corr ← %s", args.corr_path)
    corr = _load_json(args.corr_path)
    logger.info("加载 pool ← %s", args.pool_path)
    pool = _load_json(args.pool_path)
    logger.info("  corr: %d 题, pool: %d 题", len(corr), len(pool))

    corr_samples = _expand_corr(corr)
    pool_samples = _expand_pool(pool, args.n_per_q, rng)

    merged = corr_samples + pool_samples
    total = len(merged)
    logger.info("展开后样本数: corr=%d (题×1), pool=%d (题×≤%d), total=%d",
                len(corr_samples), len(pool_samples), args.n_per_q, total)
    logger.info("占比 corr=%.1f%% pool=%.1f%%",
                100 * len(corr_samples) / total, 100 * len(pool_samples) / total)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info("写出 → %s", args.out_path)

    # 抽样核对
    for src in ("corr_answer", "pool"):
        s = next((x for x in merged if x["source"] == src), None)
        if s:
            logger.info("[核对 %s] question前60=%r answer前60=%r %s",
                        src, s["question"][:60], s["answer"][:60],
                        f"fill_token={s.get('fill_token_text')!r}" if src == "pool" else "")


if __name__ == "__main__":
    main()

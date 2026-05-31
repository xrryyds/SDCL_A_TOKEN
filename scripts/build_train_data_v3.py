"""build_train_data_v3.py — V3 三池 merge → 单个 train_data_v3.json

三池来源:
  corr_answer : datasets/exam/corr_DS_MATH_pool.json      (5456 题 × 1 候选)
  roll        : datasets/exam/fill_multi_pool_roll.json   (1296 题 × avg 1.65 候选)
  pool        : datasets/exam/fill_multi_pool.json        (500 题 × avg 48.65 候选)

每题候选完全展开(不去重)成独立训练样本,带 source 字段。

Schema(对齐 a_token_sdcl_train.py:_load_train_data / _encode_sample):
  corr_answer / roll:
    {
      "source": "corr_answer" | "roll",
      "question": str,
      "answer":   str,            # 整段对答案(含首 token)
      "question_idx": int,
      "ref_answer": str,
    }

  pool:
    {
      "source": "pool",
      "question": str,
      "answer":   str,            # 整段对答案(含首 token)
      "fill_token_id":   int,     # 首位强制塞的 token id
      "fill_token_text": str,
      "question_idx":    int,
      "ref_answer":      str,
    }

Loss 形式(由 trainer 实现,本脚本只准备数据):
  corr_answer / roll : full-span 反向 KL(student || teacher),首位也算
  pool               : 首位 CE on fill_token_id + 其余反向 KL
"""

import argparse
import json
import logging
import os
import sys

# Project root on PYTHONPATH
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CORR_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "corr_DS_MATH_pool.json"
)
DEFAULT_ROLL_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_pool_roll.json"
)
DEFAULT_POOL_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json"
)
DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "train", "train_data_v3.json"
)


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


def _expand_roll(roll_items):
    out = []
    for item in roll_items:
        q = item.get("question", "")
        ref = str(item.get("ref_answer", ""))
        qidx = item.get("question_idx", -1)
        for cand in item.get("candidates", []):
            ans = cand.get("answer", "")
            ttxt = cand.get("token_text", "")
            if not q or not ans:
                continue
            out.append({
                "source": "roll",
                "question": q,
                "answer": ans,
                "fill_token_text": ttxt,  # SDFT hint 用; V3 反向 KL 路径忽略
                "question_idx": qidx,
                "ref_answer": ref,
            })
    return out


def _expand_pool(pool_items):
    out = []
    for item in pool_items:
        q = item.get("question", "")
        ref = str(item.get("ref_answer", ""))
        qidx = item.get("question_idx", -1)
        for cand in item.get("candidates", []):
            ans = cand.get("answer", "")
            tid = cand.get("token_id")
            ttxt = cand.get("token_text", "")
            if not q or not ans or tid is None:
                continue
            out.append({
                "source": "pool",
                "question": q,
                "answer": ans,
                "fill_token_id": int(tid),
                "fill_token_text": ttxt,
                "question_idx": qidx,
                "ref_answer": ref,
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr_path", type=str, default=DEFAULT_CORR_PATH)
    parser.add_argument("--roll_path", type=str, default=DEFAULT_ROLL_PATH)
    parser.add_argument("--pool_path", type=str, default=DEFAULT_POOL_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    logger.info("加载三池...")
    corr = _load_json(args.corr_path)
    roll = _load_json(args.roll_path)
    pool = _load_json(args.pool_path)
    logger.info("  corr: %d 题 from %s", len(corr), args.corr_path)
    logger.info("  roll: %d 题 from %s", len(roll), args.roll_path)
    logger.info("  pool: %d 题 from %s", len(pool), args.pool_path)

    corr_samples = _expand_corr(corr)
    roll_samples = _expand_roll(roll)
    pool_samples = _expand_pool(pool)

    logger.info("展开后样本数:")
    logger.info("  corr_answer : %d (题数 %d × 1)", len(corr_samples), len(corr))
    logger.info("  roll        : %d (题数 %d × avg %.2f)",
                len(roll_samples), len(roll), len(roll_samples) / max(len(roll), 1))
    logger.info("  pool        : %d (题数 %d × avg %.2f)",
                len(pool_samples), len(pool), len(pool_samples) / max(len(pool), 1))

    merged = corr_samples + roll_samples + pool_samples
    total = len(merged)
    logger.info("  total       : %d", total)
    logger.info("  占比 corr=%.1f%% roll=%.1f%% pool=%.1f%%",
                100 * len(corr_samples) / total,
                100 * len(roll_samples) / total,
                100 * len(pool_samples) / total)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 抽样核对
    logger.info("=== 抽样核对 ===")
    for src in ("corr_answer", "roll", "pool"):
        sample = next((s for s in merged if s["source"] == src), None)
        if sample is None:
            logger.warning("  [%s] 0 样本!", src)
            continue
        keys = list(sample.keys())
        ans_head = sample["answer"][:60].replace("\n", " ")
        line = f"  [{src}] keys={keys} q_idx={sample['question_idx']}"
        if src == "pool":
            tid = sample["fill_token_id"]
            ttxt = sample["fill_token_text"]
            matches = sample["answer"].startswith(ttxt)
            line += f" fill_tok_id={tid} fill_tok_text={ttxt!r} ans_starts_with_tok={matches}"
        line += f"\n            answer[:60]={ans_head!r}"
        logger.info(line)

    logger.info("写出: %s (%d 样本)", args.out_path, total)
    return total


if __name__ == "__main__":
    main()

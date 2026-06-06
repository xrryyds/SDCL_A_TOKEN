"""build_train_data_orpo.py — 构造 ORPO 训练数据 (4096 口径)

每条样本: (prompt, chosen, rejected)
  prompt   = mistake 池里的 question
  rejected = mistake 池里的 base greedy 错答案 (it["answer"])
  chosen 分两源:
    - 题在 unsolve_pool (roll-8 全错) → 在 fill_unsolve_pool 里救回的 → 随机选 1 个 candidate.answer
    - 题不在 unsolve_pool (roll-8 至少做对一次) → 从 roll8_base_*.jsonl 里 8 sample 中做对的随机 1 个

跳过: unsolve 池里 fill 也救不回的题 (fill_unsolve_unresolved.json) → 没 chosen, 跳过

输入:
  datasets/exam/mistake_DS_MATH_pool.json     (2023 题, rejected 来源)
  datasets/exam/unsolve_pool.json             (1094 题, roll-8 全错的硬骨头)
  datasets/exam/fill_unsolve_pool.json        (~? 题, fill 救回的, chosen 来源 1)
  datasets/exam/fill_unsolve_unresolved.json  (硬骨头中 376 token 都救不回的, 跳过)
  scripts/tmp/roll8_base_<TS>.jsonl           (mistake 池 roll-8 raw, 非 unsolve 题的 chosen 来源 2)

输出:
  datasets/train/train_data_orpo.json
  schema:
    {
      "question_idx": int,
      "question": str,
      "ref_answer": str,
      "prompt": str,        # 即 question (build_train_data 阶段不加 chat template)
      "chosen": str,        # 做对的答案
      "rejected": str,      # base greedy 错答案
      "chosen_source": str  # "fill" (fill 救回) / "roll8" (roll-8 做对)
    }

用法:
  python scripts/build_train_data_orpo.py \\
    --roll8_jsonl scripts/tmp/roll8_base_20260606_173234.jsonl \\
    --out_path datasets/train/train_data_orpo.json \\
    --seed 42
"""
import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Set

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.data_utils import extract_boxed_content, normalize_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MISTAKE = os.path.join(_PROJECT_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")
DEFAULT_UNSOLVE = os.path.join(_PROJECT_ROOT, "datasets", "exam", "unsolve_pool.json")
DEFAULT_FILL_UNSOLVE = os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_unsolve_pool.json")
DEFAULT_OUT = os.path.join(_PROJECT_ROOT, "datasets", "train", "train_data_orpo.json")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_correct(text: str, ref: str) -> bool:
    """boxed 字符串 normalize 比较 (与 teacher_mark_paper / build_fill_pool_token 一致)。"""
    b = extract_boxed_content(text or "")
    if not b:
        return False
    return normalize_answer(b) == normalize_answer(str(ref))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistake_path", type=str, default=DEFAULT_MISTAKE)
    parser.add_argument("--unsolve_path", type=str, default=DEFAULT_UNSOLVE)
    parser.add_argument("--fill_unsolve_path", type=str, default=DEFAULT_FILL_UNSOLVE)
    parser.add_argument("--roll8_jsonl", type=str, required=True,
                        help="阶段2 跑的 roll8_base_*.jsonl 路径")
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ---- 加载数据 ----
    logger.info("加载 mistake 池 ← %s", args.mistake_path)
    mistake = _load_json(args.mistake_path)
    logger.info("  mistake: %d 题", len(mistake))

    logger.info("加载 unsolve 池 ← %s", args.unsolve_path)
    unsolve = _load_json(args.unsolve_path)
    unsolve_qidx: Set = set(it["question_idx"] for it in unsolve)
    logger.info("  unsolve: %d 题 (roll-8 全错)", len(unsolve))

    logger.info("加载 fill_unsolve_pool ← %s", args.fill_unsolve_path)
    if os.path.exists(args.fill_unsolve_path):
        fill_unsolve = _load_json(args.fill_unsolve_path)
        fill_by_qidx: Dict = {it["question_idx"]: it for it in fill_unsolve}
        logger.info("  fill_unsolve: %d 题 (fill 救回)", len(fill_unsolve))
    else:
        logger.warning("  fill_unsolve_pool 不存在 (阶段3 还没跑完?), 没有 chosen 来源1, 只用来源2 (roll8)")
        fill_by_qidx = {}

    # 加载 roll8: 每行 {question_idx, ref_answer, n_correct_of_8, samples}
    logger.info("加载 roll8 ← %s", args.roll8_jsonl)
    roll8_by_qidx: Dict = {}
    with open(args.roll8_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            roll8_by_qidx[r["question_idx"]] = r
    logger.info("  roll8: %d 题", len(roll8_by_qidx))

    # ---- 构造每条样本 ----
    out: List[Dict] = []
    n_chosen_fill = 0
    n_chosen_roll8 = 0
    n_skip_unresolved = 0
    n_skip_no_correct_roll8 = 0
    n_skip_no_rejected = 0

    for it in mistake:
        qi = it["question_idx"]
        question = it.get("question", "")
        ref = str(it.get("ref_answer", ""))
        rejected = it.get("answer", "")

        if not question or not rejected:
            n_skip_no_rejected += 1
            continue

        chosen: Optional[str] = None
        chosen_source: Optional[str] = None

        if qi in unsolve_qidx:
            # 这题是 unsolve (roll-8 全错), 去 fill_unsolve_pool 找救回的
            fill_item = fill_by_qidx.get(qi)
            if fill_item is None or not fill_item.get("candidates"):
                # 这题进了 unsolve 但 fill 也救不回 (在 fill_unsolve_unresolved 里), 跳过
                n_skip_unresolved += 1
                continue
            cand = rng.choice(fill_item["candidates"])
            chosen = cand.get("answer", "")
            chosen_source = "fill"
            if not chosen:
                n_skip_unresolved += 1
                continue
            n_chosen_fill += 1
        else:
            # 这题不在 unsolve, 说明 roll-8 至少做对一次, 从 roll8 里挑做对的
            r8 = roll8_by_qidx.get(qi)
            if r8 is None:
                n_skip_no_correct_roll8 += 1
                continue
            correct_samples = [s for s in r8.get("samples", []) if _is_correct(s, ref)]
            if not correct_samples:
                # 不该发生 (因为这题不在 unsolve), 但保险跳过
                n_skip_no_correct_roll8 += 1
                continue
            chosen = rng.choice(correct_samples)
            chosen_source = "roll8"
            n_chosen_roll8 += 1

        out.append({
            "question_idx": qi,
            "question": question,
            "ref_answer": ref,
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_source": chosen_source,
        })

    # ---- 写盘 ----
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info("ORPO 训练数据构造完成")
    logger.info("=" * 70)
    logger.info(f"  总样本数: {len(out)}")
    logger.info(f"  chosen 来自 fill (unsolve→fill 救回): {n_chosen_fill}")
    logger.info(f"  chosen 来自 roll8 (非 unsolve, roll-8 做对): {n_chosen_roll8}")
    logger.info(f"  跳过: fill 也救不回的硬骨头 = {n_skip_unresolved}")
    logger.info(f"  跳过: roll8 缺数据或没做对 = {n_skip_no_correct_roll8}")
    logger.info(f"  跳过: 缺 question/rejected = {n_skip_no_rejected}")
    logger.info(f"  写出 → {args.out_path}")

    # 抽样核对
    if out:
        s = out[0]
        logger.info("=== 抽样核对 (第 1 条) ===")
        logger.info(f"  qi={s['question_idx']} ref={s['ref_answer']!r}")
        logger.info(f"  chosen_source={s['chosen_source']}")
        logger.info(f"  prompt 前80: {s['prompt'][:80]!r}")
        logger.info(f"  chosen 前80: {s['chosen'][:80]!r}")
        logger.info(f"  rejected 前80: {s['rejected'][:80]!r}")


if __name__ == "__main__":
    main()

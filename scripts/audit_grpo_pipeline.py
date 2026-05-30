"""GRPO 三池流水线审计脚本。

把 build_grpo_pool → merge → smoke 各阶段的关键中间变量 / 字段完整性 /
随机抽样数据全部写到一个 JSON 报告里。

用法:
    python scripts/audit_grpo_pipeline.py \\
        --stage grpo_pool \\
        --grpo_pool_path datasets/exam/grpo_DS_MATH_pool.json \\
        --out_dir logs/grpo_pipeline_<ts>

    python scripts/audit_grpo_pipeline.py \\
        --stage merged \\
        --merged_path datasets/exam/a_token_train_data_with_grpo.json \\
        --out_dir logs/grpo_pipeline_<ts>

跑完 grpo_pool / merged 两阶段后，out_dir/audit_<stage>.json 即审计报告，
out_dir/audit_<stage>.txt 是人类可读摘要(给我贴这个就够)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _save(out_dir: str, stage: str, report: Dict[str, Any], summary_lines: List[str]):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"audit_{stage}.json")
    txt_path = os.path.join(out_dir, f"audit_{stage}.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines), flush=True)
    print(f"\n[audit] JSON  -> {json_path}", flush=True)
    print(f"[audit] TXT   -> {txt_path}", flush=True)


def audit_grpo_pool(args):
    path = args.grpo_pool_path
    report: Dict[str, Any] = {
        "stage": "grpo_pool",
        "timestamp": datetime.now().isoformat(),
        "path": path,
    }
    summary: List[str] = []
    summary.append("=" * 70)
    summary.append(f"[audit grpo_pool] {path}")
    summary.append("=" * 70)

    if not os.path.exists(path):
        report["error"] = "file not found"
        summary.append("ERROR: file not found")
        _save(args.out_dir, "grpo_pool", report, summary)
        return

    with open(path, "r", encoding="utf-8") as f:
        grpo = json.load(f)

    n_total = len(grpo)
    distinct = len(set(g.get("question_idx") for g in grpo))
    required = {
        "question_idx", "question", "ref_answer", "anchor_answer",
        "anchor_first_token_id", "anchor_first_token_text",
        "n_correct_of_k", "k", "source",
    }
    missing = [i for i, g in enumerate(grpo) if not required.issubset(g.keys())]
    n_correct_dist = sorted(Counter(g.get("n_correct_of_k") for g in grpo).items())
    k_values = sorted(set(g.get("k") for g in grpo))
    sources = sorted(set(g.get("source") for g in grpo))
    n_anchor_empty = sum(1 for g in grpo if not g.get("anchor_answer"))
    n_ref_empty = sum(1 for g in grpo if not g.get("ref_answer"))

    report.update({
        "entries": n_total,
        "distinct_question_idx": distinct,
        "missing_field_entries": len(missing),
        "missing_field_sample_idx": missing[:5],
        "n_correct_of_k_dist": n_correct_dist,
        "k_values": k_values,
        "source_values": sources,
        "n_anchor_empty": n_anchor_empty,
        "n_ref_empty": n_ref_empty,
    })

    # 抽样 5 条
    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(n_total), min(5, n_total)) if n_total > 0 else []
    samples = []
    for i in sample_idx:
        g = grpo[i]
        anchor_text = g.get("anchor_answer", "")
        samples.append({
            "idx": i,
            "question_idx": g.get("question_idx"),
            "question_head": (g.get("question") or "")[:140],
            "ref_answer": g.get("ref_answer"),
            "anchor_answer_head": anchor_text[:120],
            "anchor_answer_tail": anchor_text[-120:],
            "anchor_first_token_id": g.get("anchor_first_token_id"),
            "anchor_first_token_text": g.get("anchor_first_token_text"),
            "n_correct_of_k": g.get("n_correct_of_k"),
            "k": g.get("k"),
            "source": g.get("source"),
        })
    report["random_samples"] = samples

    # Tokenizer 一致性:anchor_first_token_id == tokenize(anchor_answer)[0]
    if args.model_path and n_total > 0:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                args.model_path, trust_remote_code=True, use_fast=False
            )
            check_idx = rng.sample(range(n_total), min(50, n_total))
            mismatch = []
            for i in check_idx:
                g = grpo[i]
                ids = tok.encode(g.get("anchor_answer", ""), add_special_tokens=False)
                if not ids:
                    mismatch.append({"idx": i, "reason": "empty token ids"})
                    continue
                if ids[0] != g.get("anchor_first_token_id"):
                    mismatch.append({
                        "idx": i,
                        "expect": g.get("anchor_first_token_id"),
                        "actual": int(ids[0]),
                        "actual_text": tok.decode([int(ids[0])]),
                    })
            report["tokenizer_check"] = {
                "checked": len(check_idx),
                "mismatch_count": len(mismatch),
                "mismatch_sample": mismatch[:5],
            }
        except Exception as e:
            report["tokenizer_check"] = {"error": f"{type(e).__name__}: {e}"}

    # === 摘要 ===
    summary.append(f"entries                 = {n_total}")
    summary.append(f"distinct question_idx   = {distinct}  "
                   f"({'OK' if distinct == n_total else 'WARN dup!'})")
    summary.append(f"missing-field entries   = {len(missing)}  "
                   f"({'OK' if not missing else 'FAIL'})")
    summary.append(f"n_correct_of_k dist     = {n_correct_dist}")
    summary.append(f"  (0 应不存在,否则救回逻辑写错)")
    summary.append(f"k values                = {k_values}  (期望 [8])")
    summary.append(f"source values           = {sources}  (期望 ['grpo'])")
    summary.append(f"anchor_answer empty     = {n_anchor_empty}  "
                   f"({'OK' if n_anchor_empty == 0 else 'WARN'})")
    summary.append(f"ref_answer empty        = {n_ref_empty}  "
                   f"({'OK' if n_ref_empty == 0 else 'WARN'})")
    if "tokenizer_check" in report:
        tc = report["tokenizer_check"]
        if "error" in tc:
            summary.append(f"tokenizer check         : ERROR {tc['error']}")
        else:
            summary.append(
                f"tokenizer first-id check: {tc['mismatch_count']}/{tc['checked']} "
                f"mismatch  ({'OK' if tc['mismatch_count'] == 0 else 'WARN'})"
            )
    summary.append(f"\n--- random samples ({len(samples)}) ---")
    for s in samples:
        summary.append(
            f"[#{s['idx']}] q_idx={s['question_idx']} n_correct={s['n_correct_of_k']}/{s['k']} "
            f"first_id={s['anchor_first_token_id']} first_text={s['anchor_first_token_text']!r}"
        )
        summary.append(f"   ref_answer = {s['ref_answer']!r}")
        summary.append(f"   anchor_head= {s['anchor_answer_head']!r}")
        summary.append(f"   anchor_tail= ...{s['anchor_answer_tail']!r}")
    _save(args.out_dir, "grpo_pool", report, summary)


def audit_merged(args):
    path = args.merged_path
    report: Dict[str, Any] = {
        "stage": "merged",
        "timestamp": datetime.now().isoformat(),
        "path": path,
    }
    summary: List[str] = []
    summary.append("=" * 70)
    summary.append(f"[audit merged train_data] {path}")
    summary.append("=" * 70)

    if not os.path.exists(path):
        report["error"] = "file not found"
        summary.append("ERROR: file not found")
        _save(args.out_dir, "merged", report, summary)
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = len(data)
    src_dist = Counter(d.get("source") for d in data)
    grpo_only = [d for d in data if d.get("source") == "grpo"]
    corr_only = [d for d in data if d.get("source") == "corr_answer"]
    fill_only = [d for d in data if d.get("source") == "fill_correct"]

    # grpo 不变量:answer 为空 / fill_token_id 为 None
    grpo_with_stray_answer = [
        i for i, d in enumerate(grpo_only)
        if d.get("answer") not in ("", None)
    ]
    grpo_with_fill = [
        i for i, d in enumerate(grpo_only) if d.get("fill_token_id") is not None
    ]
    grpo_missing_anchor = [
        i for i, d in enumerate(grpo_only)
        if d.get("anchor_first_token_id") is None
    ]
    grpo_missing_question = [
        i for i, d in enumerate(grpo_only) if not d.get("question")
    ]

    # fill 不变量:fill_token_id 是 int
    fill_with_bad_token = [
        i for i, d in enumerate(fill_only)
        if not isinstance(d.get("fill_token_id"), int)
    ]

    # corr 不变量:answer 非空
    corr_empty_answer = [
        i for i, d in enumerate(corr_only) if not d.get("answer")
    ]

    report.update({
        "total": n_total,
        "source_dist": dict(src_dist),
        "grpo_count": len(grpo_only),
        "corr_count": len(corr_only),
        "fill_count": len(fill_only),
        "grpo_with_stray_answer": len(grpo_with_stray_answer),
        "grpo_with_fill_token_id": len(grpo_with_fill),
        "grpo_missing_anchor": len(grpo_missing_anchor),
        "grpo_missing_question": len(grpo_missing_question),
        "fill_with_bad_token": len(fill_with_bad_token),
        "corr_empty_answer": len(corr_empty_answer),
    })

    rng = random.Random(args.seed)
    def _take_samples(lst, n):
        if not lst:
            return []
        idx = rng.sample(range(len(lst)), min(n, len(lst)))
        out = []
        for i in idx:
            d = lst[i]
            out.append({
                "idx_in_subset": i,
                "source": d.get("source"),
                "question_head": (d.get("question") or "")[:140],
                "answer_head": (d.get("answer") or "")[:120],
                "anchor_first_token_id": d.get("anchor_first_token_id"),
                "fill_token_id": d.get("fill_token_id"),
            })
        return out

    report["sample_grpo"] = _take_samples(grpo_only, 3)
    report["sample_corr"] = _take_samples(corr_only, 2)
    report["sample_fill"] = _take_samples(fill_only, 2)

    summary.append(f"total entries           = {n_total}")
    summary.append(f"source dist             = {dict(src_dist)}")
    summary.append(
        f"corr={len(corr_only)}  fill={len(fill_only)}  grpo={len(grpo_only)}"
    )
    def ok(n): return "OK" if n == 0 else "FAIL"
    summary.append(f"grpo stray answer       = {len(grpo_with_stray_answer)}  ({ok(len(grpo_with_stray_answer))})")
    summary.append(f"grpo with fill_token_id = {len(grpo_with_fill)}  ({ok(len(grpo_with_fill))})")
    summary.append(f"grpo missing anchor_id  = {len(grpo_missing_anchor)}  ({ok(len(grpo_missing_anchor))})")
    summary.append(f"grpo missing question   = {len(grpo_missing_question)}  ({ok(len(grpo_missing_question))})")
    summary.append(f"fill with bad token_id  = {len(fill_with_bad_token)}  ({ok(len(fill_with_bad_token))})")
    summary.append(f"corr with empty answer  = {len(corr_empty_answer)}  ({ok(len(corr_empty_answer))})")

    summary.append("\n--- random GRPO samples ---")
    for s in report["sample_grpo"]:
        summary.append(
            f"  q_head={s['question_head']!r}\n"
            f"    answer(expect '')={s['answer_head']!r}  "
            f"anchor_first_token_id={s['anchor_first_token_id']}  "
            f"fill_token_id(expect None)={s['fill_token_id']}"
        )
    summary.append("\n--- random CORR samples ---")
    for s in report["sample_corr"]:
        summary.append(
            f"  q_head={s['question_head']!r}\n    answer_head={s['answer_head']!r}"
        )
    summary.append("\n--- random FILL samples ---")
    for s in report["sample_fill"]:
        summary.append(
            f"  q_head={s['question_head']!r}\n"
            f"    answer_head={s['answer_head']!r}  fill_token_id={s['fill_token_id']}"
        )
    _save(args.out_dir, "merged", report, summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["grpo_pool", "merged"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grpo_pool_path", default="datasets/exam/grpo_DS_MATH_pool.json")
    ap.add_argument("--merged_path", default="datasets/exam/a_token_train_data_with_grpo.json")
    ap.add_argument(
        "--model_path",
        default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
        help="tokenizer 用,做 anchor_first_token_id 一致性校验",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "grpo_pool":
        audit_grpo_pool(args)
    elif args.stage == "merged":
        audit_merged(args)


if __name__ == "__main__":
    main()

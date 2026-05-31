"""V3 三池蒸馏评测脚本。

评测矩阵 (LoRA / Base 各跑一次):
  - corr 池        : datasets/exam/corr_DS_MATH_pool.json (5456 题)  - greedy
  - roll 池        : datasets/exam/fill_multi_pool_roll.json (1296)  - greedy
  - pool 池        : datasets/exam/fill_multi_pool.json (500)        - greedy
  - MATH-500       : Math_500()                                       - greedy
  - MATH-500 多口径 : 同上                                              - roll-8 (pass@1 avg + any@8)

参数 (与训练数据采集对齐):
  max_prompt_length=2048, max_new_tokens=4096
  prompt 用 take_exam.py 的 SYSTEM_PROMPT + apply_chat_template
  判分: extract_boxed_content + normalize_answer

输出: output/eval_v3_<ts>/
  各评测 jsonl 落盘 + summary.json + 终端整理打印

跑完或异常都进 use_worker 保活。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

# 让脚本可以独立 python 运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from scripts.inference.take_exam import TakeExam
from data_math.MATH_500_data_util import Math_500
from utils.data_utils import extract_boxed_content, normalize_answer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================
# 数据加载
# =====================================================
def load_pool(path: str) -> Tuple[List[str], List[str], List[str], List[int]]:
    """加载 corr / roll / pool 池(三个池 question / ref_answer 字段一致;
    roll/pool 没有 ref_solution,缺失时填空串)。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = [it["question"] for it in data]
    ref_answers = [it["ref_answer"] for it in data]
    ref_solutions = [it.get("ref_solution", "") for it in data]
    indices = list(range(len(data)))
    return questions, ref_answers, ref_solutions, indices


def load_math500() -> Tuple[List[str], List[str], List[str], List[int]]:
    data = Math_500()
    return data.problems, data.answers, data.solutions, list(range(len(data.problems)))
    # 注意: Math_500 接口是 (problems, solutions, answers, data_len);
    #      这里返回顺序统一为 (q, ref_ans, ref_sol, idx) 与 load_pool 对齐


# =====================================================
# 推理 wrapper
# =====================================================
def run_greedy(
    questions: List[str],
    ref_answers: List[str],
    ref_solutions: List[str],
    indices: List[int],
    model_path: str,
    lora_path: Optional[str],
    max_prompt_length: int,
    max_new_tokens: int,
    device_ids: Optional[List[int]],
) -> List[Dict]:
    """跑 greedy 推理 (n=1, T=0)。返回每题 {question, answer, ref_answer, ref_solution, question_idx}。"""
    kwargs = dict(
        model_path=model_path,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )
    if lora_path:
        kwargs["use_lora"] = True
        kwargs["adapter_path"] = lora_path

    take_exam = TakeExam(**kwargs)
    try:
        results = take_exam.exam_multi_gpu(
            questions, ref_solutions, ref_answers, indices,
            device_ids=device_ids,
            write_output=False,
            sample_n=1,
            temperature=0.0,
            top_p=1.0,
        )
    finally:
        del take_exam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_roll_k(
    questions: List[str],
    ref_answers: List[str],
    ref_solutions: List[str],
    indices: List[int],
    model_path: str,
    lora_path: Optional[str],
    max_prompt_length: int,
    max_new_tokens: int,
    device_ids: Optional[List[int]],
    sample_n: int = 8,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> List[Dict]:
    """跑 roll-k 推理 (n=8, T=0.6, top_p=0.95)。返回每题 {question, samples, ref_answer, ...}。"""
    kwargs = dict(
        model_path=model_path,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )
    if lora_path:
        kwargs["use_lora"] = True
        kwargs["adapter_path"] = lora_path

    take_exam = TakeExam(**kwargs)
    try:
        results = take_exam.exam_multi_gpu(
            questions, ref_solutions, ref_answers, indices,
            device_ids=device_ids,
            write_output=False,
            sample_n=sample_n,
            temperature=temperature,
            top_p=top_p,
        )
    finally:
        del take_exam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# =====================================================
# 判分
# =====================================================
def judge_one(pred_text: str, ref_answer: str) -> bool:
    pred_norm = normalize_answer(extract_boxed_content(pred_text) or "")
    ref_norm = normalize_answer(ref_answer)
    return pred_norm == ref_norm


def grade_greedy(results: List[Dict]) -> Tuple[int, int, List[Dict]]:
    """返回 (correct, total, graded_items)。"""
    correct = 0
    graded = []
    for r in results:
        ok = judge_one(r["answer"], r["ref_answer"])
        if ok:
            correct += 1
        graded.append({
            "question_idx": r["question_idx"],
            "ref_answer": r["ref_answer"],
            "pred_raw": r["answer"],
            "pred_extracted": normalize_answer(extract_boxed_content(r["answer"]) or ""),
            "correct": ok,
        })
    return correct, len(results), graded


def grade_roll_k(results: List[Dict]) -> Tuple[float, float, int, List[Dict]]:
    """对 roll-k 结果做两种判分:
       - pass@1 averaged: 每题 8 次取平均正确率,再题间平均
       - any@8: 每题 8 次有一次对就算对
    返回 (pass1_avg_pct, any8_pct, total, graded_items)。"""
    n_total = len(results)
    sum_pass1 = 0.0
    n_any8 = 0
    graded = []
    for r in results:
        samples = r.get("samples", [])
        ref = r["ref_answer"]
        per_sample_correct = [judge_one(s, ref) for s in samples]
        n_corr = sum(per_sample_correct)
        n_samp = max(len(samples), 1)
        pass1 = n_corr / n_samp
        any8 = (n_corr > 0)
        sum_pass1 += pass1
        if any8:
            n_any8 += 1
        graded.append({
            "question_idx": r["question_idx"],
            "ref_answer": ref,
            "n_samples": len(samples),
            "n_correct": n_corr,
            "pass1": pass1,
            "any8": bool(any8),
        })
    pass1_avg = (sum_pass1 / n_total * 100.0) if n_total > 0 else 0.0
    any8_pct = (n_any8 / n_total * 100.0) if n_total > 0 else 0.0
    return pass1_avg, any8_pct, n_total, graded


# =====================================================
# 单次评测 (一个数据集 × 一个 LoRA 设置 × 一个口径)
# =====================================================
def eval_one(
    name: str,
    questions: List[str],
    ref_answers: List[str],
    ref_solutions: List[str],
    indices: List[int],
    model_path: str,
    lora_path: Optional[str],
    max_prompt_length: int,
    max_new_tokens: int,
    device_ids: Optional[List[int]],
    output_dir: str,
    mode: str,  # "greedy" or "roll8"
) -> Dict:
    """跑一次评测,落盘 jsonl,返回 summary dict。"""
    tag = f"{name}_{'lora' if lora_path else 'base'}_{mode}"
    logger.info("=" * 70)
    logger.info(f"[eval] {tag}  n_questions={len(questions)}")
    logger.info("=" * 70)
    t0 = time.time()

    if mode == "greedy":
        results = run_greedy(
            questions, ref_answers, ref_solutions, indices,
            model_path=model_path, lora_path=lora_path,
            max_prompt_length=max_prompt_length, max_new_tokens=max_new_tokens,
            device_ids=device_ids,
        )
        correct, total, graded = grade_greedy(results)
        acc = (correct / total * 100.0) if total > 0 else 0.0
        items_path = os.path.join(output_dir, f"{tag}.jsonl")
        with open(items_path, "w", encoding="utf-8") as f:
            for it in graded:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        summary = {
            "tag": tag,
            "dataset": name,
            "lora": bool(lora_path),
            "mode": "greedy",
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "elapsed_sec": elapsed,
            "items_path": items_path,
        }
        logger.info(f"[eval done] {tag}: {correct}/{total} = {acc:.2f}%  ({elapsed:.1f}s)")

    elif mode == "roll8":
        results = run_roll_k(
            questions, ref_answers, ref_solutions, indices,
            model_path=model_path, lora_path=lora_path,
            max_prompt_length=max_prompt_length, max_new_tokens=max_new_tokens,
            device_ids=device_ids,
            sample_n=8, temperature=0.6, top_p=0.95,
        )
        pass1, any8, total, graded = grade_roll_k(results)
        items_path = os.path.join(output_dir, f"{tag}.jsonl")
        with open(items_path, "w", encoding="utf-8") as f:
            for it in graded:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        summary = {
            "tag": tag,
            "dataset": name,
            "lora": bool(lora_path),
            "mode": "roll8",
            "total": total,
            "pass1_avg": pass1,
            "any8": any8,
            "elapsed_sec": elapsed,
            "items_path": items_path,
        }
        logger.info(f"[eval done] {tag}: pass@1_avg={pass1:.2f}%  any@8={any8:.2f}%  ({elapsed:.1f}s)")
    else:
        raise ValueError(f"unknown mode: {mode}")

    return summary


# =====================================================
# 整理打印
# =====================================================
def print_final_table(summaries: List[Dict]):
    """终端打印整理后的实验结果。"""
    print()
    print("=" * 86)
    print(" V3 三池蒸馏 评测结果汇总")
    print("=" * 86)

    # greedy 表 (5 行 × 2 列 base/lora)
    print()
    print(" [Greedy 准确率]  (n=1, T=0)")
    print(" " + "-" * 78)
    print(f" {'Dataset':<28}  {'Base':>15}  {'LoRA':>15}  {'Δ':>10}")
    print(" " + "-" * 78)
    by_key = {(s["dataset"], s["mode"], s["lora"]): s for s in summaries}
    for ds in ["corr", "roll", "pool", "math500"]:
        base_s = by_key.get((ds, "greedy", False))
        lora_s = by_key.get((ds, "greedy", True))
        base_str = f"{base_s['accuracy']:.2f}% ({base_s['correct']}/{base_s['total']})" if base_s else "-"
        lora_str = f"{lora_s['accuracy']:.2f}% ({lora_s['correct']}/{lora_s['total']})" if lora_s else "-"
        delta = f"{lora_s['accuracy'] - base_s['accuracy']:+.2f}%" if (base_s and lora_s) else "-"
        print(f" {ds:<28}  {base_str:>15}  {lora_str:>15}  {delta:>10}")

    # MATH-500 roll-8
    print()
    print(" [MATH-500 Roll-8]  (n=8, T=0.6, top_p=0.95)")
    print(" " + "-" * 78)
    print(f" {'Metric':<28}  {'Base':>15}  {'LoRA':>15}  {'Δ':>10}")
    print(" " + "-" * 78)
    base_r = by_key.get(("math500", "roll8", False))
    lora_r = by_key.get(("math500", "roll8", True))
    if base_r or lora_r:
        for metric in ["pass1_avg", "any8"]:
            label = "pass@1 averaged" if metric == "pass1_avg" else "any@8"
            base_str = f"{base_r[metric]:.2f}%" if base_r else "-"
            lora_str = f"{lora_r[metric]:.2f}%" if lora_r else "-"
            delta = f"{lora_r[metric] - base_r[metric]:+.2f}%" if (base_r and lora_r) else "-"
            print(f" {label:<28}  {base_str:>15}  {lora_str:>15}  {delta:>10}")
    print()
    print("=" * 86)
    print()


# =====================================================
# 主流程
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="V3 三池蒸馏评测")
    parser.add_argument(
        "--lora_path", type=str, required=True,
        help="训练完的 LoRA 路径,例如 output/v3_4card_bs8_<ts>/",
    )
    parser.add_argument(
        "--model_path", type=str,
        default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
    )
    parser.add_argument(
        "--corr_path", type=str,
        default="datasets/exam/corr_DS_MATH_pool.json",
    )
    parser.add_argument(
        "--roll_path", type=str,
        default="datasets/exam/fill_multi_pool_roll.json",
    )
    parser.add_argument(
        "--pool_path", type=str,
        default="datasets/exam/fill_multi_pool.json",
    )
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--device_ids", type=str, default=None,
        help="逗号分隔 GPU id,默认全部可见 GPU。",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device_ids: Optional[List[int]] = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("output", f"eval_v3_{ts}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置文件日志
    file_handler = logging.FileHandler(
        os.path.join(args.output_dir, "eval.log"), encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.info(f"output_dir = {args.output_dir}")
    logger.info(f"lora_path  = {args.lora_path}")
    logger.info(f"model_path = {args.model_path}")
    logger.info(f"max_prompt_length={args.max_prompt_length}  max_new_tokens={args.max_new_tokens}")
    logger.info(f"device_ids = {device_ids if device_ids else 'all'}")

    # ============ 数据加载 ============
    logger.info("加载数据集 ...")
    corr_data = load_pool(args.corr_path)
    roll_data = load_pool(args.roll_path)
    pool_data = load_pool(args.pool_path)
    math500_data = load_math500()
    logger.info(f"  corr    : {len(corr_data[0])} 题")
    logger.info(f"  roll    : {len(roll_data[0])} 题")
    logger.info(f"  pool    : {len(pool_data[0])} 题")
    logger.info(f"  math500 : {len(math500_data[0])} 题")

    datasets = [
        ("corr",    corr_data),
        ("roll",    roll_data),
        ("pool",    pool_data),
        ("math500", math500_data),
    ]

    # ============ 评测顺序: 先 LoRA,后 base ============
    summaries: List[Dict] = []
    common_kwargs = dict(
        model_path=args.model_path,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        device_ids=device_ids,
        output_dir=args.output_dir,
    )

    # ----- LoRA pass -----
    logger.info("\n" + "#" * 70)
    logger.info("# Phase 1/2: LoRA")
    logger.info("#" * 70)
    for name, (q, ra, rs, idx) in datasets:
        s = eval_one(
            name=name, questions=q, ref_answers=ra, ref_solutions=rs, indices=idx,
            lora_path=args.lora_path, mode="greedy",
            **common_kwargs,
        )
        summaries.append(s)
    # MATH-500 roll-8 (LoRA)
    name, (q, ra, rs, idx) = "math500", math500_data
    s = eval_one(
        name=name, questions=q, ref_answers=ra, ref_solutions=rs, indices=idx,
        lora_path=args.lora_path, mode="roll8",
        **common_kwargs,
    )
    summaries.append(s)

    # ----- Base pass -----
    logger.info("\n" + "#" * 70)
    logger.info("# Phase 2/2: Base (no LoRA)")
    logger.info("#" * 70)
    for name, (q, ra, rs, idx) in datasets:
        s = eval_one(
            name=name, questions=q, ref_answers=ra, ref_solutions=rs, indices=idx,
            lora_path=None, mode="greedy",
            **common_kwargs,
        )
        summaries.append(s)
    # MATH-500 roll-8 (Base)
    name, (q, ra, rs, idx) = "math500", math500_data
    s = eval_one(
        name=name, questions=q, ref_answers=ra, ref_solutions=rs, indices=idx,
        lora_path=None, mode="roll8",
        **common_kwargs,
    )
    summaries.append(s)

    # ============ 落盘 summary ============
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "lora_path": args.lora_path,
            "model_path": args.model_path,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "results": summaries,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"summary 落盘 → {summary_path}")

    # ============ 终端打印 ============
    print_final_table(summaries)


if __name__ == "__main__":
    overall = "ok"
    top_err = None
    try:
        main()
    except BaseException as e:
        overall = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("\n" + "=" * 70, flush=True)
        print("评测异常:", flush=True)
        print(top_err, flush=True)
        print("=" * 70, flush=True)
    finally:
        print("\n" + "=" * 70, flush=True)
        print(f"评测状态: {overall}", flush=True)
        print("=" * 70, flush=True)
        print("\n进入 use_worker 保活 (Ctrl+C 退出) ...", flush=True)
        try:
            from main import use_worker
            use_worker()
        except BaseException:
            traceback.print_exc()

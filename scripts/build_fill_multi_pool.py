"""Build fill_multi pool: 对 mistake 池每题做最多 10 轮 rolling-K rollout,
收集所有救回轮中对 rollout 的"首 token id 去重 + 完整对答案"作为多候选。

流程:
  对每题 (active 集合,初始 = 全部 mistake):
    for round in 1..MAX_ROUNDS:
        vLLM rolling-K (n=K, T=0.6, top_p=0.95) 跑一遍 active 集合
        for 每题:
            收集本轮 K 条 rollout 中对的(boxed 字符串相等)
            如果 ≥1 对 → 该题救回:
                - 按 first_token_id 去重(每个 token id 保留首次出现的对 rollout)
                - 这一轮所有去重后的 candidates 写入结果
                - 从 active 集合移除
            否则保留在 active,进入下一轮
        if active 空: break

输出:
  datasets/exam/fill_multi_pool.json     # 救回的题
  datasets/exam/fill_multi_unresolved.json  # 10 轮都没救回的题

复用:
  - utils/data_utils.extract_boxed_content / normalize_answer (评分)
  - take_exam.SYSTEM_PROMPT (chat template)
  - vLLM 直接 LLM.generate,不走 TakeExam class(避免它内部种子干扰)

Env:
  CUDA_VISIBLE_DEVICES 决定 tensor_parallel_size (=GPU 数)
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Project root on PYTHONPATH
_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# vLLM v1 multiprocess fork mode
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from utils.data_utils import extract_boxed_content, normalize_answer  # noqa: E402

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Please reason step by step and put your final answer within \\boxed{}."
)

DEFAULT_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "model", "DS", "DeepSeek-R1-Distill-Qwen-7B"
)
DEFAULT_MISTAKE_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json"
)
DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_pool.json"
)
DEFAULT_UNRESOLVED_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_multi_unresolved.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _is_correct(generated_text: str, ref_answer: str) -> bool:
    """规则评分: boxed 字符串相等 (与 teacher_mark_paper 同源)。"""
    pred = extract_boxed_content(generated_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(str(ref_answer))


def _first_token_id(tokenizer: AutoTokenizer, text: str) -> Optional[int]:
    """text 的首 BPE token id (无 BOS)。空串返回 None。"""
    if not text:
        return None
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return None
    return int(ids[0])


def _detok(tokenizer: AutoTokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mistake_path", type=str, default=DEFAULT_MISTAKE_PATH)
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--unresolved_path", type=str, default=DEFAULT_UNRESOLVED_PATH)
    parser.add_argument("--k", type=int, default=16, help="每轮 rollout 数")
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9
    )
    parser.add_argument(
        "--save_every_round", action="store_true",
        help="每轮结束后保存中间结果 (容错)。"
    )
    parser.add_argument(
        "--shard_idx", type=int, default=0,
        help="当前 shard 编号 (0-based),用于多机分工。"
    )
    parser.add_argument(
        "--shard_total", type=int, default=1,
        help="总 shard 数,默认 1 = 不分片单机跑全集。"
    )
    args = parser.parse_args()

    assert 0 <= args.shard_idx < args.shard_total, \
        f"shard_idx={args.shard_idx} 必须在 [0, {args.shard_total})"

    # ---- 读 mistake 池 ----
    with open(args.mistake_path, "r", encoding="utf-8") as f:
        mistakes_all = json.load(f)
    n_all = len(mistakes_all)

    # 分片: 用 round-robin (i % shard_total == shard_idx) 保证负载均衡
    # (vs 连续切片: 简单题/难题可能集中在某一段,导致两台机器耗时差很多)
    mistakes = [m for i, m in enumerate(mistakes_all) if i % args.shard_total == args.shard_idx]
    logger.info(
        "加载 mistake 池: 全集 %d 题, shard %d/%d → 本机处理 %d 题 (from %s)",
        n_all, args.shard_idx, args.shard_total, len(mistakes), args.mistake_path,
    )

    # 输出路径加 shard 后缀,避免两台机器互相覆盖
    if args.shard_total > 1:
        def _add_shard_suffix(path: str) -> str:
            base, ext = os.path.splitext(path)
            return f"{base}.shard{args.shard_idx}of{args.shard_total}{ext}"
        args.out_path = _add_shard_suffix(args.out_path)
        args.unresolved_path = _add_shard_suffix(args.unresolved_path)
        logger.info("分片输出路径: %s / %s", args.out_path, args.unresolved_path)

    # ---- vLLM 初始化 ----
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_gpu = len([x for x in cuda_visible.split(",") if x.strip()]) if cuda_visible else 1
    if n_gpu == 0:
        n_gpu = 1
    max_model_len = args.max_prompt_length + args.max_new_tokens
    logger.info(
        "vLLM init: TP=%d, max_model_len=%d (=%d prompt + %d gen), K=%d, T=%g, top_p=%g",
        n_gpu, max_model_len, args.max_prompt_length, args.max_new_tokens,
        args.k, args.temperature, args.top_p,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    stop_token_ids = [tokenizer.eos_token_id, 151643, 151645]

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=n_gpu,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="bfloat16",
        seed=42,
    )

    # ---- per-question 状态 ----
    # active: List[idx_in_mistakes] 还未救回的题 (从全集开始)
    # results[mistake_idx] = {question, ref_answer, candidates: [...], n_rounds_used, total_rollouts}
    # candidates 已按 token_id 去重: dict[token_id -> {token_id, token_text, answer, round}]
    active: List[int] = list(range(len(mistakes)))
    results: Dict[int, Dict] = {}
    total_rollouts: Dict[int, int] = {i: 0 for i in active}

    t0 = time.time()
    for round_idx in range(1, args.max_rounds + 1):
        if not active:
            logger.info("所有题已救回,提前结束于 round %d", round_idx - 1)
            break

        logger.info(
            "===== Round %d/%d  active=%d  ===== (elapsed %.1fs)",
            round_idx, args.max_rounds, len(active), time.time() - t0,
        )

        # 构造本轮 prompt
        prompts = [_build_prompt(tokenizer, mistakes[i]["question"]) for i in active]

        sampling = SamplingParams(
            n=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            seed=42 + round_idx,  # 每轮种子不同,保证 rollout 多样性
        )

        outputs = llm.generate(prompts, sampling)

        # 评分 + 收集
        rescued_this_round: List[int] = []
        for local_i, out in enumerate(outputs):
            mistake_idx = active[local_i]
            ref_ans = mistakes[mistake_idx]["ref_answer"]
            total_rollouts[mistake_idx] += args.k

            candidates_dedup: Dict[int, Dict] = {}
            for sample in out.outputs:
                text = sample.text.strip()
                if not _is_correct(text, ref_ans):
                    continue
                tid = _first_token_id(tokenizer, text)
                if tid is None:
                    continue
                if tid in candidates_dedup:
                    continue  # 去重: 保留首次出现的
                candidates_dedup[tid] = {
                    "token_id": tid,
                    "token_text": _detok(tokenizer, tid),
                    "answer": text,
                    "round": round_idx,
                }

            if candidates_dedup:
                # 这题救回了
                results[mistake_idx] = {
                    "question_idx": mistakes[mistake_idx].get("question_idx", mistake_idx),
                    "question": mistakes[mistake_idx]["question"],
                    "ref_answer": str(mistakes[mistake_idx]["ref_answer"]),
                    "candidates": list(candidates_dedup.values()),
                    "n_rounds_used": round_idx,
                    "total_correct_rollouts_this_round": sum(
                        1 for s in out.outputs
                        if _is_correct(s.text.strip(), ref_ans)
                    ),
                    "total_rollouts": total_rollouts[mistake_idx],
                    "source": "fill_multi",
                }
                rescued_this_round.append(mistake_idx)

        # 从 active 移除已救回
        active = [i for i in active if i not in set(rescued_this_round)]
        logger.info(
            "Round %d: rescued=%d, remaining active=%d",
            round_idx, len(rescued_this_round), len(active),
        )

        # 容错: 中途保存
        if args.save_every_round:
            _dump_results(results, mistakes, active, total_rollouts, args)

    # ---- 最终写盘 ----
    _dump_results(results, mistakes, active, total_rollouts, args)

    elapsed = time.time() - t0
    logger.info(
        "=========== Done in %.1fs ===========\n"
        "  rescued: %d / %d (%.2f%%)\n"
        "  unresolved: %d\n"
        "  out:        %s\n"
        "  unresolved: %s",
        elapsed, len(results), len(mistakes),
        100.0 * len(results) / max(len(mistakes), 1),
        len(active), args.out_path, args.unresolved_path,
    )

    # 返回汇总信息给外层 finally 打印 (避免 logger 被 vLLM 日志洪水淹没)
    return {
        "elapsed": elapsed,
        "rescued": len(results),
        "total": len(mistakes),
        "unresolved": len(active),
        "out_path": args.out_path,
        "unresolved_path": args.unresolved_path,
        "rounds_used_dist": _summarize_rounds(results),
        "candidates_dist": _summarize_candidates(results),
    }


def _summarize_rounds(results: Dict[int, Dict]) -> Dict[int, int]:
    """统计 n_rounds_used 分布: {round -> count}。"""
    dist: Dict[int, int] = {}
    for r in results.values():
        n = r["n_rounds_used"]
        dist[n] = dist.get(n, 0) + 1
    return dist


def _summarize_candidates(results: Dict[int, Dict]) -> Dict[str, float]:
    """统计 candidates 数量分布。"""
    if not results:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0}
    counts = sorted(len(r["candidates"]) for r in results.values())
    return {
        "min": counts[0],
        "max": counts[-1],
        "avg": sum(counts) / len(counts),
        "median": counts[len(counts) // 2],
    }


def _print_final_summary(summary: Optional[Dict], status: str, top_err: Optional[str]):
    """tmux-friendly 最终汇总,直接 print 到 stdout (不走 logger 避免被淹)。"""
    bar = "=" * 70
    print("\n" + bar, flush=True)
    print(f" build_fill_multi_pool.py — FINAL SUMMARY  [status={status}]", flush=True)
    print(bar, flush=True)
    if summary is None:
        print(" (no summary — script aborted before main loop completed)", flush=True)
    else:
        total = summary["total"]
        rescued = summary["rescued"]
        unresolved = summary["unresolved"]
        rate = 100.0 * rescued / max(total, 1)
        print(f"  elapsed         : {summary['elapsed']:.1f}s "
              f"({summary['elapsed']/60:.1f} min)", flush=True)
        print(f"  fill 收集题数   : {rescued} / {total} ({rate:.2f}%)", flush=True)
        print(f"  unresolved      : {unresolved} / {total} ({100-rate:.2f}%)", flush=True)
        print(f"  rescued file    : {summary['out_path']}", flush=True)
        print(f"  unresolved file : {summary['unresolved_path']}", flush=True)
        rd = summary["rounds_used_dist"]
        if rd:
            print("  rounds_used 分布:", flush=True)
            for r in sorted(rd.keys()):
                print(f"    round {r:2d}: {rd[r]:5d} 题", flush=True)
        cd = summary["candidates_dist"]
        print(f"  candidates/题   : min={cd['min']} max={cd['max']} "
              f"avg={cd['avg']:.2f} median={cd['median']}", flush=True)
    if top_err:
        print(bar, flush=True)
        print(" TOP-LEVEL EXCEPTION:", flush=True)
        print(top_err, flush=True)
    print(bar + "\n", flush=True)


def _dump_results(
    results: Dict[int, Dict],
    mistakes: List[Dict],
    active: List[int],
    total_rollouts: Dict[int, int],
    args,
):
    """落盘救回 + 未救回两个文件。"""
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    rescued = [results[i] for i in sorted(results.keys())]
    unresolved = [
        {
            "question_idx": mistakes[i].get("question_idx", i),
            "question": mistakes[i]["question"],
            "ref_answer": str(mistakes[i]["ref_answer"]),
            "n_rounds": args.max_rounds,
            "total_rollouts": total_rollouts[i],
        }
        for i in active
    ]

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(rescued, f, ensure_ascii=False, indent=2)
    with open(args.unresolved_path, "w", encoding="utf-8") as f:
        json.dump(unresolved, f, ensure_ascii=False, indent=2)

    logger.info(
        "[checkpoint] rescued=%d → %s ; unresolved=%d → %s",
        len(rescued), args.out_path, len(unresolved), args.unresolved_path,
    )


if __name__ == "__main__":
    overall_status = "ok"
    top_err = None
    summary = None
    try:
        summary = main()
    except BaseException as e:
        import traceback
        overall_status = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error("Top-level exception:\n%s", top_err)
    finally:
        # 最终汇总打印 (tmux-friendly, 直接走 stdout, 不会被 vLLM 日志淹没)
        try:
            _print_final_summary(summary, overall_status, top_err)
        except BaseException as e:
            print(f"[summary print failed: {e}]", flush=True)

        # use_worker 保活 (踩过的坑: vLLM 子进程退出后 main.py 的 use_worker
        # 会在 print torch.cuda.get_device_name 时炸 AssertionError,这里不影响产物)
        try:
            from main import use_worker
            logger.info("Calling use_worker() for keepalive (status=%s)", overall_status)
            use_worker()
        except BaseException as e:
            logger.warning("use_worker failed (non-fatal): %s", e)

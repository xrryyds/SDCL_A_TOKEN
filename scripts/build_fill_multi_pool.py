"""Build fill_multi pool: 对 mistake 池每题做最多 10 轮 rolling-K rollout,
收集所有救回轮中对 rollout 的"首 token id 去重 + 完整对答案"作为多候选。

数据并行 (DP):每张卡一个独立子进程,tensor_parallel_size=1,
mistake 池连续切片分发给各 worker;每 worker 独立跑完 10 轮主循环;
主进程 Queue 收集 + 合并 + 写盘。

流程 (per worker):
  active = 该 worker 分到的 mistake shard
  for round in 1..MAX_ROUNDS:
      vLLM rolling-K (n=K, T=0.6, top_p=0.95) 跑一遍 active
      for 每题:
          收集本轮 K 条 rollout 中对的(boxed 字符串相等)
          如果 ≥1 对 → 该题救回:
              - 按 first_token_id 去重(每个 token id 保留首次出现的对 rollout)
              - 这一轮所有去重后的 candidates 写入结果
              - 从 active 移除
          否则保留在 active,进入下一轮
      if active 空: break

输出:
  datasets/exam/fill_multi_pool.json     # 救回的题
  datasets/exam/fill_multi_unresolved.json  # 10 轮都没救回的题

复用:
  - utils/data_utils.extract_boxed_content / normalize_answer (评分)
  - take_exam.SYSTEM_PROMPT (chat template)
  - 范式照 scripts/train/a_token_sdcl.py:DP + spawn + Queue

Env:
  CUDA_VISIBLE_DEVICES 决定子进程数 (=GPU 数)
"""

import argparse
import json
import logging
import multiprocessing as mp
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
# Helpers (worker 内会重新 import,放模块顶层)
# -----------------------------------------------------------------------------
def _build_prompt(tokenizer, question: str) -> str:
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


def _first_token_id(tokenizer, text: str) -> Optional[int]:
    """text 的首 BPE token id (无 BOS)。空串返回 None。"""
    if not text:
        return None
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        return None
    return int(ids[0])


# -----------------------------------------------------------------------------
# Worker (DP: 一张卡一个子进程)
# -----------------------------------------------------------------------------
def _proc_target(args, q, idx):
    """spawn 子进程入口,把结果回传给主进程。模块顶层函数,可 pickle。"""
    try:
        res = _worker_fill_multi(args)
        q.put((idx, "ok", res))
    except BaseException as e:
        import traceback
        q.put(
            (idx, "err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        )


def _worker_fill_multi(args) -> Dict:
    """单卡子进程:在 device_id 上跑 vLLM,处理 mistake_shard 的 10 轮主循环。

    返回 dict:
      {
        "device_id": int,
        "rescued": List[Dict],     # 救回的题(完整 entry)
        "unresolved": List[Dict],  # 10 轮没救回的题
        "rounds_used_dist": Dict[int, int],
      }
    """
    (
        device_id,
        model_path,
        mistake_shard,           # List[Dict]
        k,
        max_rounds,
        temperature,
        top_p,
        max_prompt_length,
        max_new_tokens,
        gpu_memory_utilization,
        seed,
    ) = args

    # 把当前进程绑定到一张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # 在子进程内才 import vllm,避免父进程提前抢占 GPU
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info(
        "[worker dev=%s] 启动,shard_size=%d,k=%d,max_rounds=%d",
        device_id, len(mistake_shard), k, max_rounds,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    stop_token_ids = [tokenizer.eos_token_id, 151643, 151645]
    max_model_len = max_prompt_length + max_new_tokens

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="bfloat16",
        seed=seed + device_id,
    )

    # active: List[idx_in_shard] 还未救回的题 (从全 shard 开始)
    active: List[int] = list(range(len(mistake_shard)))
    results: Dict[int, Dict] = {}
    total_rollouts: Dict[int, int] = {i: 0 for i in active}

    t0 = time.time()
    for round_idx in range(1, max_rounds + 1):
        if not active:
            logger.info(
                "[worker dev=%s] 所有题已救回,提前结束于 round %d",
                device_id, round_idx - 1,
            )
            break

        logger.info(
            "[worker dev=%s] ===== Round %d/%d  active=%d  ===== (elapsed %.1fs)",
            device_id, round_idx, max_rounds, len(active), time.time() - t0,
        )

        prompts = [_build_prompt(tokenizer, mistake_shard[i]["question"]) for i in active]

        sampling = SamplingParams(
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            seed=seed + device_id * 1000 + round_idx,
        )

        outputs = llm.generate(prompts, sampling)

        rescued_this_round: List[int] = []
        for local_i, out in enumerate(outputs):
            shard_idx = active[local_i]
            ref_ans = mistake_shard[shard_idx]["ref_answer"]
            total_rollouts[shard_idx] += k

            candidates_dedup: Dict[int, Dict] = {}
            for sample in out.outputs:
                text = sample.text.strip()
                if not _is_correct(text, ref_ans):
                    continue
                tid = _first_token_id(tokenizer, text)
                if tid is None:
                    continue
                if tid in candidates_dedup:
                    continue
                candidates_dedup[tid] = {
                    "token_id": tid,
                    "token_text": tokenizer.decode([tid], skip_special_tokens=False),
                    "answer": text,
                    "round": round_idx,
                }

            if candidates_dedup:
                results[shard_idx] = {
                    "question_idx": mistake_shard[shard_idx].get("question_idx", shard_idx),
                    "question": mistake_shard[shard_idx]["question"],
                    "ref_answer": str(mistake_shard[shard_idx]["ref_answer"]),
                    "candidates": list(candidates_dedup.values()),
                    "n_rounds_used": round_idx,
                    "total_correct_rollouts_this_round": sum(
                        1 for s in out.outputs
                        if _is_correct(s.text.strip(), ref_ans)
                    ),
                    "total_rollouts": total_rollouts[shard_idx],
                    "source": "fill_multi",
                }
                rescued_this_round.append(shard_idx)

        active = [i for i in active if i not in set(rescued_this_round)]
        logger.info(
            "[worker dev=%s] Round %d: rescued=%d, remaining active=%d",
            device_id, round_idx, len(rescued_this_round), len(active),
        )

    # 整理产出
    rescued = [results[i] for i in sorted(results.keys())]
    unresolved = [
        {
            "question_idx": mistake_shard[i].get("question_idx", i),
            "question": mistake_shard[i]["question"],
            "ref_answer": str(mistake_shard[i]["ref_answer"]),
            "n_rounds": max_rounds,
            "total_rollouts": total_rollouts[i],
        }
        for i in active
    ]

    rounds_used_dist: Dict[int, int] = {}
    for r in rescued:
        n = r["n_rounds_used"]
        rounds_used_dist[n] = rounds_used_dist.get(n, 0) + 1

    elapsed = time.time() - t0
    logger.info(
        "[worker dev=%s] 完成: rescued=%d/%d unresolved=%d elapsed=%.1fs",
        device_id, len(rescued), len(mistake_shard), len(unresolved), elapsed,
    )

    del llm
    return {
        "device_id": device_id,
        "rescued": rescued,
        "unresolved": unresolved,
        "rounds_used_dist": rounds_used_dist,
        "elapsed": elapsed,
    }


# -----------------------------------------------------------------------------
# Main: DP 调度 + 合并
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
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 读 mistake 池 ----
    with open(args.mistake_path, "r", encoding="utf-8") as f:
        mistakes = json.load(f)
    logger.info("加载 mistake 池: %d 题 from %s", len(mistakes), args.mistake_path)

    # ---- 决定使用哪些 GPU (从 CUDA_VISIBLE_DEVICES 推断) ----
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        device_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        try:
            import torch
            device_ids = list(range(torch.cuda.device_count()))
        except Exception:
            device_ids = [0]
    if not device_ids:
        raise RuntimeError("无可用 GPU")

    # 注意:子进程内会再次设 CUDA_VISIBLE_DEVICES=str(device_id)。
    # 这里 device_ids 是相对当前 CUDA_VISIBLE_DEVICES 的逻辑 id。
    # 例如外部 CUDA_VISIBLE_DEVICES=0,1,2,3 → 子进程逻辑 id 也是 0,1,2,3。
    # 但子进程 spawn 后 env 继承父进程,我们用相对索引重设,所以 device_ids 改成 0..N-1。
    n_devices = len(device_ids)
    logical_device_ids = list(range(n_devices))

    # ---- 切片: 连续切片,保留顺序 (照 a_token_sdcl.py) ----
    n_workers = min(n_devices, len(mistakes))
    total = len(mistakes)
    shard_size = (total + n_workers - 1) // n_workers
    shards = [
        mistakes[i * shard_size: min((i + 1) * shard_size, total)]
        for i in range(n_workers)
    ]

    args_list = [
        (
            dev,
            args.model_path,
            shard,
            args.k,
            args.max_rounds,
            args.temperature,
            args.top_p,
            args.max_prompt_length,
            args.max_new_tokens,
            args.gpu_memory_utilization,
            args.seed,
        )
        for dev, shard in zip(logical_device_ids, shards) if shard
    ]

    logger.info(
        "启动 %d 个 worker,逻辑 devices=%s,shard_size≈%d,K=%d,max_rounds=%d",
        len(args_list), [a[0] for a in args_list], shard_size,
        args.k, args.max_rounds,
    )

    t0 = time.time()

    if len(args_list) == 1:
        # 单卡直接跑
        worker_results = [_worker_fill_multi(args_list[0])]
    else:
        # spawn + 显式 Process + Queue (照 a_token_sdcl.py)
        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        procs = []
        for i, a in enumerate(args_list):
            p = ctx.Process(
                target=_proc_target,
                args=(a, result_q, i),
                daemon=False,
            )
            p.start()
            procs.append(p)

        worker_results_by_idx: Dict[int, Dict] = {}
        first_error = None
        for _ in range(len(procs)):
            idx, status, payload = result_q.get()
            if status == "ok":
                worker_results_by_idx[idx] = payload
            else:
                if first_error is None:
                    first_error = payload
                logger.error("[worker idx=%d] failed:\n%s", idx, payload)

        for p in procs:
            p.join()

        if first_error is not None and not worker_results_by_idx:
            raise RuntimeError(f"全部 worker 失败:\n{first_error}")

        worker_results = [worker_results_by_idx[i] for i in sorted(worker_results_by_idx.keys())]

    # ---- 合并结果 ----
    all_rescued: List[Dict] = []
    all_unresolved: List[Dict] = []
    merged_rounds_dist: Dict[int, int] = {}
    for wr in worker_results:
        all_rescued.extend(wr["rescued"])
        all_unresolved.extend(wr["unresolved"])
        for rnd, cnt in wr["rounds_used_dist"].items():
            merged_rounds_dist[rnd] = merged_rounds_dist.get(rnd, 0) + cnt

    # 按 question_idx 排序
    all_rescued.sort(key=lambda x: x.get("question_idx", 0))
    all_unresolved.sort(key=lambda x: x.get("question_idx", 0))

    # ---- 写盘 ----
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(all_rescued, f, ensure_ascii=False, indent=2)
    with open(args.unresolved_path, "w", encoding="utf-8") as f:
        json.dump(all_unresolved, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    counts = sorted(len(r["candidates"]) for r in all_rescued) if all_rescued else [0]
    candidates_dist = {
        "min": counts[0],
        "max": counts[-1],
        "avg": sum(counts) / len(counts),
        "median": counts[len(counts) // 2],
    }

    logger.info(
        "=========== Done in %.1fs ===========\n"
        "  rescued: %d / %d (%.2f%%)\n"
        "  unresolved: %d\n"
        "  out:        %s\n"
        "  unresolved: %s",
        elapsed, len(all_rescued), len(mistakes),
        100.0 * len(all_rescued) / max(len(mistakes), 1),
        len(all_unresolved), args.out_path, args.unresolved_path,
    )

    return {
        "elapsed": elapsed,
        "rescued": len(all_rescued),
        "total": len(mistakes),
        "unresolved": len(all_unresolved),
        "out_path": args.out_path,
        "unresolved_path": args.unresolved_path,
        "rounds_used_dist": merged_rounds_dist,
        "candidates_dist": candidates_dist,
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
        try:
            _print_final_summary(summary, overall_status, top_err)
        except BaseException as e:
            print(f"[summary print failed: {e}]", flush=True)

        try:
            from main import use_worker
            logger.info("Calling use_worker() for keepalive (status=%s)", overall_status)
            use_worker()
        except BaseException as e:
            logger.warning("use_worker failed (non-fatal): %s", e)

"""Build fill pool by exhaustive first-token池: 对 unsolve_pool 每题,
把首 token 池 (datasets/first_tokens_test.json) 中**所有** token 都塞一次,
T=0 greedy 续写,收集做对的 candidates。

跳过 fill_multi 阶段, 直接对 unsolve 池 (roll-8 全错的硬骨头) 做 376 token × 题 笛卡尔积。

数据并行 (DP):每张卡一个独立子进程,tensor_parallel_size=1,
unsolve 池连续切片分发给各 worker;主进程 Queue 收集 + 合并 + 写盘。

流程 (per worker):
  shard = 该 worker 分到的 unsolve 题
  for each item in shard:
      base_ids = chat_template(question)  (token-id 层,左截断到 prompt_len)
      for tid in 376_pool:
          fill_inputs.append(TokensPrompt(base_ids + [tid]))
      llm.generate(fill_inputs, T=0 greedy, max_new=4096)
      for each (tid, gen_text) of this question:
          full_answer = token_text + gen_text
          if 对 (boxed 字符串相等):
              candidates.append({token_id, token_text, answer})

输出 (新文件名, 不覆盖 fill_multi_pool.json):
  datasets/exam/fill_unsolve_pool.json        # N 题救回(候选完整对答案多 token)
  datasets/exam/fill_unsolve_unresolved.json  # N - 救回 题仍未救回的硬骨头

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
from typing import Dict, List, Optional

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
DEFAULT_UNRESOLVED_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "unsolve_pool.json"
)
DEFAULT_FIRST_TOKEN_POOL_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "first_tokens_test.json"
)
DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_unsolve_pool.json"
)
DEFAULT_OUT_UNRESOLVED_PATH = os.path.join(
    _PROJECT_ROOT, "datasets", "exam", "fill_unsolve_unresolved.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
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
    """boxed 字符串相等 (与 teacher_mark_paper / fill_multi 同源)。"""
    pred = extract_boxed_content(generated_text)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(str(ref_answer))


def _build_stop_token_ids(tokenizer) -> List[int]:
    """与 a_token_sdcl 同口径: eos + 151643 + 151645。"""
    sids = [tokenizer.eos_token_id]
    for extra in (151643, 151645):
        if extra not in sids:
            sids.append(extra)
    return [s for s in sids if s is not None]


# -----------------------------------------------------------------------------
# Worker (DP: 一张卡一个子进程)
# -----------------------------------------------------------------------------
def _proc_target(args, q, idx):
    """spawn 子进程入口。模块顶层函数,可 pickle。"""
    try:
        res = _worker_fill_pool(args)
        q.put((idx, "ok", res))
    except BaseException as e:
        import traceback
        q.put(
            (idx, "err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        )


def _worker_fill_pool(args) -> Dict:
    """单卡子进程:在 device_id 上跑 vLLM,处理 unresolved_shard。

    范式照 scripts/train/a_token_sdcl.py:_worker_fill (token-id 层拼接 + greedy)。
    """
    (
        device_id,
        model_path,
        unresolved_shard,        # List[Dict]
        first_token_pool,        # List[{token_id, token_text}]
        max_prompt_length,
        max_new_tokens,
        gpu_memory_utilization,
        seed,
    ) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    pool_size = len(first_token_pool)
    logger.info(
        "[worker dev=%s] 启动,shard_size=%d,pool_size=%d → 总 rollout=%d",
        device_id, len(unresolved_shard), pool_size,
        len(unresolved_shard) * pool_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        # 优化 1: 显式开 prefix caching。同一题 376 条 prompt 共享前 N-1 token 的 KV cache,
        # vLLM 默认值通常已是 True,这里显式声明防止某些版本被关。算法 0 影响。
        enable_prefix_caching=True,
    )
    stop_ids = _build_stop_token_ids(tokenizer)

    # ── 构造每题的 base prompt token ids (左截断保留 chat template 结尾) ──
    base_prompts: List[str] = []
    for item in unresolved_shard:
        base_prompts.append(_build_prompt(tokenizer, item.get("question", "")))

    prompt_ids_list: List[List[int]] = []
    for p in base_prompts:
        ids = tokenizer(p, add_special_tokens=False).input_ids
        if len(ids) > max_prompt_length:
            ids = ids[-max_prompt_length:]
        prompt_ids_list.append(list(ids))

    # ── 平铺所有 (题, 候选首 token) → 单个大 batch ──
    # 优化 2: 按 (q_idx_in_shard, token_idx_in_pool) 顺序平铺。
    # 同一题的 N=pool_size 条 prompt 连续提交,前 N-1 token 完全相同,
    # 配合 enable_prefix_caching,vLLM 第一条会算完整 prefix KV cache,
    # 后 N-1 条命中 cache 只算 1 个新 token。算法 0 影响。
    fill_inputs: List = []
    fill_owner: List[int] = []          # 该 prompt 属于第几题(在 shard 内)
    fill_token_id: List[int] = []
    fill_token_text: List[str] = []

    for q_idx_in_shard, prompt_ids in enumerate(prompt_ids_list):
        for cand in first_token_pool:
            tid = int(cand["token_id"])
            fill_inputs.append(
                TokensPrompt(prompt_token_ids=list(prompt_ids) + [tid])
            )
            fill_owner.append(q_idx_in_shard)
            fill_token_id.append(tid)
            fill_token_text.append(cand["token_text"])

    if not fill_inputs:
        logger.warning("[worker dev=%s] 无候选 prompt,返回空。", device_id)
        del llm
        return {
            "device_id": device_id,
            "rescued": [],
            "unresolved": list(unresolved_shard),
        }

    fill_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
        seed=seed,
    )
    logger.info(
        "[worker dev=%s] greedy 续写 batch=%d (题=%d × pool=%d)",
        device_id, len(fill_inputs), len(unresolved_shard), pool_size,
    )
    t_gen = time.time()
    fill_outputs = llm.generate(fill_inputs, fill_sampling)
    logger.info(
        "[worker dev=%s] generate 完成,耗时 %.1fs",
        device_id, time.time() - t_gen,
    )

    # ── 收集每题做对的 candidates ──
    per_question_candidates: Dict[int, List[Dict]] = {}
    for owner, tid, ttext, out in zip(
        fill_owner, fill_token_id, fill_token_text, fill_outputs
    ):
        gen_text = out.outputs[0].text
        ref_ans = unresolved_shard[owner].get("ref_answer", "")
        full_answer = ttext + gen_text
        if _is_correct(full_answer, ref_ans):
            per_question_candidates.setdefault(owner, []).append({
                "token_id": int(tid),
                "token_text": ttext,
                "answer": full_answer,
            })

    # ── 整理 rescued / unresolved ──
    rescued: List[Dict] = []
    still_unresolved: List[Dict] = []
    for q_idx_in_shard, item in enumerate(unresolved_shard):
        cands = per_question_candidates.get(q_idx_in_shard, [])
        if cands:
            rescued.append({
                "question_idx": item.get("question_idx", q_idx_in_shard),
                "question": item.get("question", ""),
                "ref_answer": str(item.get("ref_answer", "")),
                "candidates": cands,
                "n_correct_of_pool": len(cands),
                "pool_size": pool_size,
            })
        else:
            still_unresolved.append({
                "question_idx": item.get("question_idx", q_idx_in_shard),
                "question": item.get("question", ""),
                "ref_answer": str(item.get("ref_answer", "")),
                "pool_size": pool_size,
            })

    logger.info(
        "[worker dev=%s] 完成: rescued=%d/%d, still_unresolved=%d",
        device_id, len(rescued), len(unresolved_shard), len(still_unresolved),
    )

    del llm
    return {
        "device_id": device_id,
        "rescued": rescued,
        "unresolved": still_unresolved,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--unresolved_path", type=str, default=DEFAULT_UNRESOLVED_PATH,
        help="输入: unsolve_pool 池 (roll-8 全错的硬骨头, 1094 题 默认)。"
    )
    parser.add_argument(
        "--first_token_pool_path", type=str, default=DEFAULT_FIRST_TOKEN_POOL_PATH,
        help="首 token 池 JSON (datasets/first_tokens_test.json)。"
    )
    parser.add_argument("--out_path", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--out_unresolved_path", type=str, default=DEFAULT_OUT_UNRESOLVED_PATH)
    parser.add_argument("--max_prompt_length", type=int, default=6144)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- 读 unresolved 池 ----
    with open(args.unresolved_path, "r", encoding="utf-8") as f:
        unresolved = json.load(f)
    logger.info("加载 unresolved 池: %d 题 from %s", len(unresolved), args.unresolved_path)

    # ---- 读首 token 池 ----
    with open(args.first_token_pool_path, "r", encoding="utf-8") as f:
        ft_pool_raw = json.load(f)
    if isinstance(ft_pool_raw, dict) and "tokens" in ft_pool_raw:
        first_token_pool = ft_pool_raw["tokens"]
    elif isinstance(ft_pool_raw, list):
        first_token_pool = ft_pool_raw
    else:
        raise ValueError(f"首 token 池格式未知: {args.first_token_pool_path}")
    logger.info(
        "加载首 token 池: %d tokens from %s",
        len(first_token_pool), args.first_token_pool_path,
    )

    # ---- GPU 设备 ----
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
    n_devices = len(device_ids)
    # 子进程内重设 CUDA_VISIBLE_DEVICES,这里用相对索引 0..N-1
    logical_device_ids = list(range(n_devices))

    # ---- 题目切片 (连续切,照 a_token_sdcl.py) ----
    n_workers = min(n_devices, len(unresolved))
    if n_workers == 0:
        logger.warning("unresolved 为空,直接写空文件。")
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        with open(args.out_unresolved_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {"rescued": 0, "unresolved": 0, "out_path": args.out_path,
                "out_unresolved_path": args.out_unresolved_path, "elapsed": 0.0,
                "total": 0, "pool_size": len(first_token_pool)}

    total = len(unresolved)
    shard_size = (total + n_workers - 1) // n_workers
    shards = [
        unresolved[i * shard_size: min((i + 1) * shard_size, total)]
        for i in range(n_workers)
    ]

    args_list = [
        (
            dev,
            args.model_path,
            shard,
            first_token_pool,
            args.max_prompt_length,
            args.max_new_tokens,
            args.gpu_memory_utilization,
            args.seed,
        )
        for dev, shard in zip(logical_device_ids, shards) if shard
    ]

    logger.info(
        "启动 %d 个 worker,逻辑 devices=%s,shard_size≈%d,pool_size=%d",
        len(args_list), [a[0] for a in args_list], shard_size,
        len(first_token_pool),
    )

    t0 = time.time()

    if len(args_list) == 1:
        worker_results = [_worker_fill_pool(args_list[0])]
    else:
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

    # ---- 合并 ----
    all_rescued: List[Dict] = []
    all_unresolved: List[Dict] = []
    for wr in worker_results:
        all_rescued.extend(wr["rescued"])
        all_unresolved.extend(wr["unresolved"])
    all_rescued.sort(key=lambda x: x.get("question_idx", 0))
    all_unresolved.sort(key=lambda x: x.get("question_idx", 0))

    # ---- 写盘 (新文件名, 不覆盖 fill_multi_pool.json) ----
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(all_rescued, f, ensure_ascii=False, indent=2)
    with open(args.out_unresolved_path, "w", encoding="utf-8") as f:
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
        "  still unresolved: %d\n"
        "  pool_size: %d\n"
        "  out (rescued):    %s\n"
        "  out (unresolved): %s",
        elapsed, len(all_rescued), len(unresolved),
        100.0 * len(all_rescued) / max(len(unresolved), 1),
        len(all_unresolved), len(first_token_pool),
        args.out_path, args.out_unresolved_path,
    )

    return {
        "elapsed": elapsed,
        "rescued": len(all_rescued),
        "total": len(unresolved),
        "unresolved": len(all_unresolved),
        "pool_size": len(first_token_pool),
        "out_path": args.out_path,
        "out_unresolved_path": args.out_unresolved_path,
        "candidates_dist": candidates_dist,
    }


def _print_final_summary(summary: Optional[Dict], status: str, top_err: Optional[str]):
    bar = "=" * 70
    print("\n" + bar, flush=True)
    print(f" build_fill_pool_token.py — FINAL SUMMARY  [status={status}]", flush=True)
    print(bar, flush=True)
    if summary is None:
        print(" (no summary — script aborted before main loop completed)", flush=True)
    else:
        total = summary["total"]
        rescued = summary["rescued"]
        unresolved = summary["unresolved"]
        rate = 100.0 * rescued / max(total, 1)
        print(f"  elapsed              : {summary['elapsed']:.1f}s "
              f"({summary['elapsed']/60:.1f} min)", flush=True)
        print(f"  pool_size            : {summary['pool_size']}", flush=True)
        print(f"  fill 救回数          : {rescued} / {total} ({rate:.2f}%)", flush=True)
        print(f"  still unresolved     : {unresolved} / {total} ({100-rate:.2f}%)", flush=True)
        print(f"  rescued file         : {summary['out_path']}", flush=True)
        print(f"  unresolved file      : {summary['out_unresolved_path']}", flush=True)
        cd = summary.get("candidates_dist", {})
        if cd:
            print(f"  candidates/题        : min={cd['min']} max={cd['max']} "
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

"""方法1：随机首 token 填充评测，生成 fill_correct.json。

流程（对 mistake 中每道错题）：
  1. 构建 prompt（apply_chat_template + add_generation_prompt）
  2. 一次 prompt_logprobs greedy 生成，提取首 token 位置的 logits（用于后续在多个候选中挑分最高者）
  3. 从 first_tokens 候选池随机抽 roll_n 个 token
  4. 把每个候选 token 的 text 作为首 token 强制拼到 prompt 末尾，让模型自由续写
  5. 提取 \\boxed{} 与 ref_answer 比较，判断是否答对
  6. 收集所有"做对"的候选；若有多个，用 step2 的 logit 最大者；若 0 个，跳过该题

多卡并行：把 mistake 题目按卡数切片，每卡一个子进程，各起一个 vLLM(tensor_parallel_size=1)。
参考实现：scripts/inference/take_exam.py 的 exam_with_hints / exam_multi_gpu。

输出格式：见 tail_token_training_proposal.md。
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# vLLM 多进程方式：在 import vllm 之前设置
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# 让本文件既能作为脚本运行也能作为模块导入
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.train.a_token_sd import (  # noqa: E402
    SYSTEM_PROMPT,
    _build_stop_token_ids,
    check_correctness,
    normalize_question_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================
# 数据 / 候选池
# =====================================================
def _load_first_token_pool(path: str) -> List[Dict]:
    """读取首 token 候选池 JSON，返回 [{token_id, token_text, count}, ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tokens = data.get("tokens", [])
    if not tokens:
        raise ValueError(f"first token 候选池为空：{path}")
    cleaned = []
    for t in tokens:
        tid = t.get("token_id")
        ttext = t.get("token_text")
        if tid is None or ttext is None:
            continue
        cleaned.append({"token_id": int(tid), "token_text": str(ttext)})
    if not cleaned:
        raise ValueError(f"first token 候选池中无有效条目：{path}")
    return cleaned


def _load_mistake(path: str) -> List[Dict]:
    """读取 mistake 数据。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"mistake 数据应是 list：{path}")
    return data


def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _sample_candidates(
    pool: List[Dict],
    roll_n: int,
    rng: random.Random,
) -> List[Dict]:
    """从候选池中无放回随机抽 roll_n 个；池子比 roll_n 小则全取。"""
    if roll_n >= len(pool):
        return list(pool)
    return rng.sample(pool, roll_n)


# =====================================================
# 单卡 worker：处理一个 mistake 子集
# =====================================================
def _worker_fill(args) -> List[Dict]:
    """单卡子进程：在 device_id 上跑 vLLM，处理 mistake_shard。

    参数打包成 tuple 以便 multiprocessing pickle。
    """
    (
        device_id,
        model_path,
        mistake_shard,           # List[Dict]，每条含 question / ref_answer 等
        first_token_pool,        # List[{token_id, token_text}]
        roll_n,
        max_gen_token,
        prompt_len,
        seed,
    ) = args

    # 把当前进程绑定到一张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # 在子进程内才 import vllm，避免父进程提前抢占 GPU
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(seed + device_id)

    logger.info(
        "[worker dev=%s] 启动，shard_size=%d，roll_n=%d",
        device_id,
        len(mistake_shard),
        roll_n,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_model_len = prompt_len + max_gen_token
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=True,
        seed=seed,
    )
    stop_ids = _build_stop_token_ids(tokenizer)

    # ── Phase A: 一次 prompt_logprobs，拿每题的首 token 位置 logits ─────────────
    # 思路：用 max_tokens=1, prompt_logprobs=10 让 vLLM 给 prompt 末尾位置的 top-K logprobs。
    # 但严格来说我们需要任意候选 token 的分数，单次 prompt_logprobs 仅返回 top-K，
    # 候选可能不在 top-K 里。因此这里改为直接用 logprobs（生成位置）取首 token 分布的 top-K 候选；
    # 候选不在 top-K 时给一个极小 logit（–1e9），仍可在多个"做对"候选间做"分数最大"比较。
    base_prompts: List[str] = []
    for item in mistake_shard:
        q = item.get("question", "")
        base_prompts.append(_build_prompt(tokenizer, q))

    # 把 prompt 文本一次性转成 token ids；超长则从左侧裁掉（保留 chat template 结尾的
    # `<|im_start|>assistant\n` 等关键位）。后续 vLLM 直接走 prompt_token_ids，
    # 避免文本拼接时 BPE 把候选首 token 与上文合并成另一个 token，
    # 导致"强制塞首 token id"语义被破坏。
    prompt_ids_list: List[List[int]] = []
    for p in base_prompts:
        ids = tokenizer(p, add_special_tokens=False).input_ids
        if len(ids) > prompt_len:
            ids = ids[-prompt_len:]
        prompt_ids_list.append(list(ids))

    # 拿首 token 位置 top-K logprobs（K 取较大值以提升候选命中率）
    K_LOGPROBS = 200
    base_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        logprobs=K_LOGPROBS,
        stop_token_ids=stop_ids,
        seed=seed,
    )
    logger.info("[worker dev=%s] Phase A: greedy 1-token + logprobs", device_id)
    base_inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list]
    base_outputs = llm.generate(base_inputs, base_sampling)

    # base_first_logprob[i] = dict[token_id -> logprob_float]
    base_first_logprob: List[Dict[int, float]] = []
    for out in base_outputs:
        gen0 = out.outputs[0]
        # vLLM 在 max_tokens=1 时，gen0.logprobs 是 [dict_at_step0]，dict: tid -> Logprob
        lp_list = gen0.logprobs or []
        if lp_list and lp_list[0] is not None:
            d = {
                int(tid): (lp.logprob if hasattr(lp, "logprob") else float(lp))
                for tid, lp in lp_list[0].items()
            }
        else:
            d = {}
        base_first_logprob.append(d)

    # ── Phase B: 对每题随机抽 roll_n 候选，构造拼接 prompt 并贪心续写 ────────
    # 把所有 (题号, 候选) 平铺到一个大 batch，一次 vLLM.generate
    # 关键：prompt 用 token_ids 拼接（prompt_ids + [cand_token_id]），不走文本层，
    # 避免 BPE 合并破坏"首 token 强制为 cand_token_id"的语义。
    fill_inputs: List = []
    fill_owner: List[int] = []          # 该 prompt 属于第几题（在 shard 内）
    fill_token_id: List[int] = []
    fill_token_text: List[str] = []

    for q_idx_in_shard, prompt_ids in enumerate(prompt_ids_list):
        cands = _sample_candidates(first_token_pool, roll_n, rng)
        for cand in cands:
            tid = int(cand["token_id"])
            fill_inputs.append(
                TokensPrompt(prompt_token_ids=list(prompt_ids) + [tid])
            )
            fill_owner.append(q_idx_in_shard)
            fill_token_id.append(tid)
            fill_token_text.append(cand["token_text"])

    if not fill_inputs:
        logger.warning("[worker dev=%s] 无候选 prompt 被构造，返回空。", device_id)
        del llm
        return []

    fill_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_gen_token,
        stop_token_ids=stop_ids,
        seed=seed,
    )
    logger.info(
        "[worker dev=%s] Phase B: 填充续写 batch=%d", device_id, len(fill_inputs)
    )
    fill_outputs = llm.generate(fill_inputs, fill_sampling)

    # ── Phase C: 收集每题的"做对候选"，挑 base 首 token logit 最大者 ───────────
    # per_question_correct[q_idx_in_shard] = List[(token_id, token_text, gen_text)]
    per_question_correct: Dict[int, List[Tuple[int, str, str]]] = {}
    for owner, tid, ttext, out in zip(
        fill_owner, fill_token_id, fill_token_text, fill_outputs
    ):
        gen_text = out.outputs[0].text
        ref_ans = mistake_shard[owner].get("ref_answer", "")
        # 答案来自 fill_token_text + gen_text 的整体（boxed 一般在续写里）
        full_answer = ttext + gen_text
        if check_correctness(full_answer, ref_ans):
            per_question_correct.setdefault(owner, []).append((tid, ttext, gen_text))

    results: List[Dict] = []
    for q_idx_in_shard, correct_list in per_question_correct.items():
        if not correct_list:
            continue
        base_lp = base_first_logprob[q_idx_in_shard]
        NEG_INF = -1e9
        # 多个做对候选时取首 token logprob 最大者
        best = max(correct_list, key=lambda x: base_lp.get(x[0], NEG_INF))
        tid, ttext, gen_text = best

        item = mistake_shard[q_idx_in_shard]
        results.append(
            {
                "question_idx": item.get("question_idx"),
                "question": item.get("question", ""),
                "answer": ttext + gen_text,
                "ref_answer": item.get("ref_answer", ""),
                "ref_solution": item.get("ref_solution", ""),
                "fill_token_id": int(tid),
                "fill_token_text": ttext,
                "source": "fill_correct",
            }
        )

    logger.info(
        "[worker dev=%s] 完成：题目=%d，命中=%d",
        device_id,
        len(mistake_shard),
        len(results),
    )

    del llm
    return results


# =====================================================
# 主入口：切片 + 多卡并行
# =====================================================
def generate_fill_correct(
    model_path: str,
    mistake_path: str,
    output_path: str,
    first_token_list_path: Optional[str] = None,
    roll_n: int = 16,
    max_gen_token: int = 2048,
    prompt_len: int = 1024,
    device_ids: Optional[List[int]] = None,
    seed: int = 42,
) -> str:
    """对 mistake 数据做随机首 token 填充评测，生成 fill_correct.json。

    Args:
        model_path:           模型路径（学生/初始模型）。
        mistake_path:         mistake JSON 路径。
        output_path:          fill_correct.json 输出路径。
        first_token_list_path: 首 token 候选池 JSON（默认 datasets/first_tokens_test.json）。
        roll_n:               每题随机抽取的候选 token 数。
        max_gen_token:        填充后续写的最大 token 数。
        prompt_len:           prompt 最大 token 数（max_model_len = prompt_len + max_gen_token）。
        device_ids:           CUDA 设备 id 列表；None 时自动用全部可见 GPU。
        seed:                 随机种子。

    Returns:
        实际写出的 output_path。
    """
    if first_token_list_path is None:
        first_token_list_path = os.path.join(
            _PROJECT_ROOT, "datasets", "first_tokens_test.json"
        )

    mistake_data = _load_mistake(mistake_path)
    first_token_pool = _load_first_token_pool(first_token_list_path)

    logger.info(
        "加载 mistake=%d，first_token 候选池=%d，roll_n=%d",
        len(mistake_data),
        len(first_token_pool),
        roll_n,
    )

    # 决定使用哪些 GPU
    if device_ids is None:
        try:
            import torch

            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n = 0
        if n <= 0:
            raise RuntimeError("当前环境无可用 CUDA 设备，无法运行 vLLM。")
        device_ids = list(range(n))
    if not device_ids:
        raise ValueError("device_ids 不能为空。")

    n_workers = min(len(device_ids), len(mistake_data))
    if n_workers == 0:
        logger.warning("mistake 数据为空，直接写出空文件。")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return output_path

    # 连续切片，保留顺序
    total = len(mistake_data)
    shard_size = (total + n_workers - 1) // n_workers
    shards = [
        mistake_data[i * shard_size : min((i + 1) * shard_size, total)]
        for i in range(n_workers)
    ]
    # 与 device_ids 一一配对（连续切片不会产生空 shard，但仍 zip 防御）
    args_list = [
        (
            dev,
            model_path,
            shard,
            first_token_pool,
            roll_n,
            max_gen_token,
            prompt_len,
            seed,
        )
        for dev, shard in zip(device_ids, shards)
        if shard
    ]

    logger.info(
        "启动 %d 个 worker，devices=%s，每 shard 题数≈%d",
        len(args_list),
        [a[0] for a in args_list],
        shard_size,
    )

    if len(args_list) == 1:
        # 单卡直接跑（避免 spawn 开销，同时方便调试）
        merged = _worker_fill(args_list[0])
    else:
        # 不能用 multiprocessing.Pool —— Pool 的 worker 默认 daemon=True，
        # 而 vLLM v1 内部还要再 fork EngineCore 子进程，daemon 进程不允许有子进程
        # 会触发 "AssertionError: daemonic processes are not allowed to have children"。
        # 改用 spawn 上下文 + 显式 Process（默认 daemon=False），并通过 Queue 收集结果。
        ctx = mp.get_context("spawn")

        def _proc_target(args, q, idx):
            try:
                res = _worker_fill(args)
                q.put((idx, "ok", res))
            except BaseException as e:
                import traceback
                q.put((idx, "err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

        result_q = ctx.Queue()
        procs = []
        for i, args in enumerate(args_list):
            p = ctx.Process(
                target=_proc_target,
                args=(args, result_q, i),
                daemon=False,  # 关键：non-daemon，才能让 vLLM 再起子进程
            )
            p.start()
            procs.append(p)

        # 收集结果（Queue 必须在 join 之前先 drain，避免子进程因管道写满阻塞）
        results_by_idx: Dict[int, List[Dict]] = {}
        first_error = None
        for _ in range(len(procs)):
            idx, status, payload = result_q.get()
            if status == "ok":
                results_by_idx[idx] = payload
            else:
                if first_error is None:
                    first_error = payload
                logger.error("[worker idx=%d] failed:\n%s", idx, payload)

        for p in procs:
            p.join()

        if first_error is not None:
            raise RuntimeError(f"至少一个 fill worker 失败：\n{first_error}")

        merged: List[Dict] = []
        for i in range(len(procs)):
            merged.extend(results_by_idx.get(i, []))

    # 按 question_idx 升序（若存在），稳定输出
    def _sort_key(r):
        v = r.get("question_idx")
        return (0, v) if isinstance(v, (int, float)) else (1, 0)

    merged.sort(key=_sort_key)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        "fill_correct 生成完成：命中题数=%d / mistake=%d，写入 %s",
        len(merged),
        total,
        output_path,
    )
    return output_path


# =====================================================
# CLI
# =====================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="方法1：随机首 token 填充评测，生成 fill_correct.json"
    )
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument(
        "--mistake_path", type=str, required=True, help="mistake 数据 JSON 路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_correct.json"),
        help="fill_correct.json 输出路径",
    )
    parser.add_argument(
        "--first_token_list_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "first_tokens_test.json"),
        help="首 token 候选池 JSON 路径",
    )
    parser.add_argument(
        "--roll_n", type=int, default=16, help="每题随机抽取的候选 token 数"
    )
    parser.add_argument("--max_gen_token", type=int, default=2048)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU id，例如 '0,1,2,3'；不传则用全部可见 GPU",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = _parse_args()
    device_ids = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    generate_fill_correct(
        model_path=args.model_path,
        mistake_path=args.mistake_path,
        output_path=args.output_path,
        first_token_list_path=args.first_token_list_path,
        roll_n=args.roll_n,
        max_gen_token=args.max_gen_token,
        prompt_len=args.prompt_len,
        device_ids=device_ids,
        seed=args.seed,
    )


# =====================================================
# 方法2：合并 corr_answer.json + fill_correct.json → a_token_train_data.json
# =====================================================
def merge_to_train_data(
    corr_answer_path: str,
    fill_correct_path: str,
    output_path: str,
    dedup: bool = True,
) -> str:
    """合并 corr_answer.json 与 fill_correct.json 为 a_token_train_data.json。

    规则（见 tail_token_training_proposal.md 第三节）：
      - corr_answer 中每条加 source="corr_answer"，fill_token_id/text 置 None
      - fill_correct 中每条已有 source="fill_correct" / fill_token_id / fill_token_text
      - 两者拼接为一个 list 写出

    Args:
        dedup : 若为 True，按 question_idx 去重；当同一 question_idx 同时出现在
                corr_answer 与 fill_correct 中时，保留 corr_answer（模型本来就答对了，
                教师分布更可信，避免被 fill_correct 的"绕路答对"覆盖）。

    Returns:
        实际写出的 output_path。
    """
    with open(corr_answer_path, "r", encoding="utf-8") as f:
        corr_data = json.load(f)
    with open(fill_correct_path, "r", encoding="utf-8") as f:
        fill_data = json.load(f)

    if not isinstance(corr_data, list):
        raise ValueError(f"corr_answer 数据应为 list：{corr_answer_path}")
    if not isinstance(fill_data, list):
        raise ValueError(f"fill_correct 数据应为 list：{fill_correct_path}")

    merged: List[Dict] = []
    seen_idx = set()  # 仅在 dedup=True 时使用

    for item in corr_data:
        qi = item.get("question_idx")
        if dedup and qi is not None:
            if qi in seen_idx:
                continue
            seen_idx.add(qi)
        merged.append(
            {
                "question_idx": qi,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "ref_answer": item.get("ref_answer", ""),
                "ref_solution": item.get("ref_solution", ""),
                "source": "corr_answer",
                "fill_token_id": None,
                "fill_token_text": None,
            }
        )

    n_skipped = 0
    for item in fill_data:
        qi = item.get("question_idx")
        if dedup and qi is not None and qi in seen_idx:
            n_skipped += 1
            continue
        if dedup and qi is not None:
            seen_idx.add(qi)
        # 已有 source/fill_token_*；为防御缺字段做一次兜底
        merged.append(
            {
                "question_idx": qi,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "ref_answer": item.get("ref_answer", ""),
                "ref_solution": item.get("ref_solution", ""),
                "source": item.get("source", "fill_correct"),
                "fill_token_id": item.get("fill_token_id"),
                "fill_token_text": item.get("fill_token_text"),
            }
        )
    if n_skipped > 0:
        logger.warning(
            "去重：%d 条 fill_correct 与 corr_answer 在 question_idx 上重复，已跳过", n_skipped
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    n_corr = sum(1 for x in merged if x["source"] == "corr_answer")
    n_fill = sum(1 for x in merged if x["source"] == "fill_correct")
    logger.info(
        "合并完成：corr_answer=%d，fill_correct=%d，total=%d，写入 %s",
        n_corr,
        n_fill,
        len(merged),
        output_path,
    )
    return output_path


def _parse_merge_args():
    parser = argparse.ArgumentParser(
        description="方法2：合并 corr_answer + fill_correct → a_token_train_data.json"
    )
    parser.add_argument(
        "--corr_answer_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "exam", "corr_answer.json"),
    )
    parser.add_argument(
        "--fill_correct_path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_correct.json"),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(
            _PROJECT_ROOT, "datasets", "exam", "a_token_train_data.json"
        ),
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="按 question_idx 去重（同 idx 时保留 corr_answer），默认开启",
    )
    return parser.parse_args()


def main_merge():
    args = _parse_merge_args()
    merge_to_train_data(
        corr_answer_path=args.corr_answer_path,
        fill_correct_path=args.fill_correct_path,
        output_path=args.output_path,
        dedup=args.dedup,
    )


if __name__ == "__main__":
    # 默认入口仍是方法1；用 `python a_token_sdcl.py merge ...` 触发方法2
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        sys.argv.pop(1)
        main_merge()
    else:
        main()

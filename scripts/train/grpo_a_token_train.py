"""GRPO + first-token-fill 训练 (路线 G)。

设计 (用户 2026-06-01 拍板):
  数据   : Math_All(train=True) ~7500 题
  Rollout: 每题 8 条 = 4 自 roll + 4 fill
           - 自 roll: 标准 vLLM 生成, T=0.6, top_p=0.95
           - fill   : 从 376 候选首 token (datasets/first_tokens_test.json) 中
                      随机选 4 个不同 token, prepend 到 prompt, 再续写
  Reward : extract_boxed_content + normalize_answer == ref_answer, binary 1/0
  Adv    : (r - mean) / (std + 1e-8), 全同组跳过, 8 条一组
  Loss   : PPO clip ε=0.2 + KL_K3 (β=0.001) ref=Base
  PPO    : epoch=1 (on-policy 严格)
  fill 首 token old logp: log(1/376) (uniform placeholder, 让 TRL 自动算 IS ratio)

参数对齐 V3:
  - LR 1e-5, max_prompt 2048, max_new 4096, seed 42
  - 4 卡 colocate vLLM (各卡训练 + 各卡跑 vLLM 引擎, 共享显存)
  - 保守 bs: per_device_train_batch_size=1, grad_accum=4

跑完/异常都进 use_worker (launcher 兜底)。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

import torch

# 让 from main import / from data_math import 等能跑
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data_math.MATH_util import Math_All
from utils.data_utils import extract_boxed_content, normalize_answer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================
# 1) 数据准备
# =====================================================
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_dataset(tokenizer, max_train: Optional[int] = None) -> Dataset:
    """Math_All(train=True) → HF Dataset with 'prompt' + 'ref_answer' 列。

    prompt 走 chat template (system + user), 与 take_exam.py 同口径。
    """
    data = Math_All(train=True, subset_name="all")
    problems = data.problems
    answers = data.answers

    if max_train is not None and max_train > 0:
        problems = problems[:max_train]
        answers = answers[:max_train]

    prompts: List[str] = []
    for q in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    ds = Dataset.from_dict({"prompt": prompts, "ref_answer": answers})
    logger.info(f"[data] Math_All(train=True): {len(ds)} 题")
    return ds


# =====================================================
# 2) Reward function
# =====================================================
def reward_correctness(
    prompts: List[str],
    completions: List[str],
    ref_answer: List[str],
    **kwargs: Any,
) -> List[float]:
    """每条 completion 判分: boxed + normalize 比对 ref_answer, 1/0。

    TRL 会按 (B*G) 顺序传 prompts/completions, ref_answer 来自 dataset 列。
    """
    out: List[float] = []
    for comp, ref in zip(completions, ref_answer):
        pred = normalize_answer(extract_boxed_content(comp) or "")
        ref_norm = normalize_answer(ref)
        out.append(1.0 if (pred and pred == ref_norm) else 0.0)
    return out


# =====================================================
# 3) Fill rollout function
# =====================================================
class FillRolloutFunc:
    """每题 8 rollout = 4 自 roll + 4 fill, 通过 trainer.vllm_generation 跑。

    fill: 从 pool_tids 随机抽 4 个不同 token, prepend prompt 后让 vLLM 续写。
    fill 首 token 的 logp 用 log(1/|pool|) (uniform placeholder, 给 PPO ratio 起点)。
    """

    def __init__(
        self,
        pool_token_path: str,
        num_self_roll: int = 4,
        num_fill: int = 4,
        seed: int = 42,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 4096,
    ):
        with open(pool_token_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cand = d["tokens"] if isinstance(d, dict) else d
        self.pool_tids: List[int] = [int(c["token_id"]) for c in cand]
        if num_fill > len(self.pool_tids):
            raise ValueError(
                f"num_fill={num_fill} 超过 pool 候选 {len(self.pool_tids)}"
            )
        self.num_self_roll = num_self_roll
        self.num_fill = num_fill
        self.num_gen = num_self_roll + num_fill
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.fill_logp_placeholder = -math.log(len(self.pool_tids))  # log(1/376)
        self.rng = random.Random(seed)
        logger.info(
            f"[FillRolloutFunc] pool_size={len(self.pool_tids)}, "
            f"num_self_roll={num_self_roll}, num_fill={num_fill}, "
            f"placeholder_logp={self.fill_logp_placeholder:.4f}"
        )

    def __call__(self, prompts: List[str], trainer: GRPOTrainer) -> Dict[str, Any]:
        """TRL 协议: 返回 {prompt_ids, completion_ids, logprobs}, 每题输出 num_gen 条。

        trainer.vllm_generation 是 TRL 起的 vLLM, 已自动 sync_weights。
        """
        tokenizer = trainer.processing_class
        vllm = trainer.vllm_generation  # 由 TRL 在 use_vllm=True 时挂上

        # 1) tokenize 每题 prompt
        prompt_ids_list: List[List[int]] = []
        for p in prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            prompt_ids_list.append(ids)

        # 2) 构造 (B*num_gen) 条 generate input
        all_prompt_ids: List[List[int]] = []     # 喂给 vLLM 的 input ids
        # 标记每条是 self/fill, 以及 fill 用了哪个 token
        is_fill_flags: List[bool] = []
        fill_tids_per_row: List[Optional[int]] = []

        for pid in prompt_ids_list:
            # 4 条自 roll: 直接用 pid
            for _ in range(self.num_self_roll):
                all_prompt_ids.append(list(pid))
                is_fill_flags.append(False)
                fill_tids_per_row.append(None)
            # 4 条 fill: prepend 4 个不同 token
            fill_tids = self.rng.sample(self.pool_tids, self.num_fill)
            for tid in fill_tids:
                all_prompt_ids.append(list(pid) + [tid])
                is_fill_flags.append(True)
                fill_tids_per_row.append(tid)

        # 3) 调 vLLM 一次性 generate
        gen_out = vllm.generate(
            prompts=all_prompt_ids,
            n=1,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        # vllm.generate 返回的格式 (TRL 包装过): 每条一个 dict-like, 含
        #   "completion_ids": List[int],  "logprobs": List[float]
        # 不同 TRL 版本字段名可能微调; 这里做 best-effort 兼容。
        comp_ids_list: List[List[int]] = []
        logp_list: List[List[float]] = []
        for o in gen_out:
            if isinstance(o, dict):
                cids = o.get("completion_ids") or o.get("token_ids") or o.get("output_ids")
                lps = o.get("logprobs") or o.get("logprob")
            else:
                cids = getattr(o, "completion_ids", None) or getattr(o, "token_ids", None)
                lps = getattr(o, "logprobs", None) or getattr(o, "logprob", None)
            if cids is None or lps is None:
                raise RuntimeError(
                    f"vllm.generate 返回缺字段, type={type(o)}, 内容样例={o}"
                )
            comp_ids_list.append(list(cids))
            logp_list.append(list(lps))

        # 4) 对 fill 条: 把 fill_tid prepend 到 completion_ids, logp 填 placeholder
        #    (原因: 我们送给 vLLM 的 input 已经包含了 fill_tid, vLLM 的输出是从 tid
        #     之后开始的; 但 TRL/GRPO 需要把 fill_tid 视为 completion 的第一个 token,
        #     这样 PPO loss 才会在该位置计算 ratio & advantage。)
        final_prompt_ids: List[List[int]] = []
        final_comp_ids: List[List[int]] = []
        final_logp: List[List[float]] = []
        for raw_pid, is_fill, fill_tid, cids, lps in zip(
            all_prompt_ids, is_fill_flags, fill_tids_per_row, comp_ids_list, logp_list
        ):
            if is_fill:
                # raw_pid = orig_prompt + [fill_tid]; 还原 orig_prompt
                orig_pid = raw_pid[:-1]
                final_prompt_ids.append(orig_pid)
                final_comp_ids.append([fill_tid] + cids)
                final_logp.append([self.fill_logp_placeholder] + lps)
            else:
                final_prompt_ids.append(raw_pid)
                final_comp_ids.append(cids)
                final_logp.append(lps)

        return {
            "prompt_ids": final_prompt_ids,
            "completion_ids": final_comp_ids,
            "logprobs": final_logp,
        }


# =====================================================
# 4) 主流程
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="GRPO + first-token-fill 训练")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--pool_token_path", type=str,
        default=os.path.join(_PROJECT_ROOT, "datasets", "first_tokens_test.json"),
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_self_roll", type=int, default=4)
    parser.add_argument("--num_fill", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.001, help="KL 系数")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument(
        "--max_train", type=int, default=None,
        help="只取前 N 题 (debug 用)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization", type=float, default=0.3,
        help="colocate 模式 vLLM 占多少显存 (训练剩 1-x 给优化器+梯度)",
    )
    args = parser.parse_args()

    if args.num_self_roll + args.num_fill != args.num_generations:
        raise ValueError(
            f"num_self_roll({args.num_self_roll}) + num_fill({args.num_fill}) "
            f"!= num_generations({args.num_generations})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"output_dir = {args.output_dir}")

    # ============ tokenizer & dataset ============
    logger.info(f"加载 tokenizer ← {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(tokenizer, max_train=args.max_train)

    # ============ rollout func ============
    rollout = FillRolloutFunc(
        pool_token_path=args.pool_token_path,
        num_self_roll=args.num_self_roll,
        num_fill=args.num_fill,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    # ============ GRPOConfig ============
    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=args.epsilon,
        seed=args.seed,
        logging_steps=args.log_interval,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.max_prompt_length + args.max_new_tokens,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # ============ Trainer ============
    logger.info("构造 GRPOTrainer ...")
    trainer = GRPOTrainer(
        model=args.model_path,
        args=grpo_cfg,
        reward_funcs=[reward_correctness],
        train_dataset=train_ds,
        processing_class=tokenizer,
        rollout_func=rollout,
    )

    logger.info("开始训练 ...")
    trainer.train()

    logger.info(f"保存最终 model → {args.output_dir}")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

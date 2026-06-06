"""orpo_train.py — ORPO 训练 (4096 口径, 调官方 orpo/src/orpo_trainer.py)

口径:
  prompt 上限: 2048 token
  answer 上限: 4096 token
  总长度: 2048 + 4096 = 6144

训练数据: datasets/train/train_data_orpo.json
  schema: {prompt, chosen, rejected, question_idx, ref_answer, chosen_source}

Loss (ORPO 论文 Eq.6-7):
  L_ORPO = L_SFT + λ · L_OR
  L_SFT  = NLL on chosen response (即 chosen 部分 next-token CE)
  L_OR   = -log σ(log odds_θ(y_w|x) / odds_θ(y_l|x))

Reference-free: 用 SFT 项当 anchor 替代 KL-to-ref, 不需要 frozen ref model。

用法 (4 卡 H800, DDP):
  scripts/train/run_orpo_train.py 是 launcher, 调 torchrun 启动本脚本。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# 让 orpo/ 子目录可 import
_ORPO_ROOT = os.path.join(_PROJECT_ROOT, "orpo")
if _ORPO_ROOT not in sys.path:
    sys.path.insert(0, _ORPO_ROOT)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# 调官方 ORPOTrainer (orpo/src/orpo_trainer.py)
from src.orpo_trainer import ORPOTrainer  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 与 take_exam.py / a_token_sdcl_train.py 一致, 池构造时同款 SYSTEM_PROMPT
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."


# =====================================================
# 数据集
# =====================================================
class ORPODataset(Dataset):
    """train_data_orpo.json → (prompt_ids, chosen_ids, rejected_ids) tokenized.

    遵循官方 orpo/main.py preprocess_dataset 逻辑:
      prompt   = chat_template([system, user]) + assistant 起始
      chosen   = chat_template([system, user, assistant=chosen_answer])
      rejected = chat_template([system, user, assistant=rejected_answer])

    返回每条: {
      input_ids        : [pad,...,prompt..., pad]  (左 pad 到 prompt_max)
      attention_mask   : 同上
      positive_input_ids       : chosen full sequence (左 pad 到 response_max)
      positive_attention_mask  : 同上
      negative_input_ids       : rejected full sequence (左 pad 到 response_max)
      negative_attention_mask  : 同上
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        prompt_max_length: int,
        response_max_length: int,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)
        self.tokenizer = tokenizer
        self.prompt_max = prompt_max_length
        self.response_max = response_max_length
        # 过滤超长 prompt: 留出至少一些 chosen 空间
        self._build_chat_strings()

    def _build_chat_strings(self):
        valid: List[Dict] = []
        n_too_long_prompt = 0
        for it in self.data:
            q = it["prompt"]
            chosen = it["chosen"]
            rejected = it["rejected"]
            prompt_str = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_str = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": chosen},
                ],
                tokenize=False,
            )
            rejected_str = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": rejected},
                ],
                tokenize=False,
            )
            # 提前过滤 prompt 超长
            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
            if len(prompt_ids) > self.prompt_max:
                n_too_long_prompt += 1
                continue
            valid.append({
                "prompt_str": prompt_str,
                "chosen_str": chosen_str,
                "rejected_str": rejected_str,
            })
        if n_too_long_prompt > 0:
            logger.warning(
                "过滤 prompt 超过 %d token 的样本: %d 条",
                self.prompt_max, n_too_long_prompt,
            )
        self.samples = valid
        logger.info("加载训练数据: %d 条 (有效)", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        prompt_enc = self.tokenizer(
            s["prompt_str"],
            max_length=self.prompt_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            s["chosen_str"],
            max_length=self.response_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            s["rejected_str"],
            max_length=self.response_max,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": prompt_enc["input_ids"][0],
            "attention_mask": prompt_enc["attention_mask"][0],
            "positive_input_ids": pos_enc["input_ids"][0],
            "positive_attention_mask": pos_enc["attention_mask"][0],
            "negative_input_ids": neg_enc["input_ids"][0],
            "negative_attention_mask": neg_enc["attention_mask"][0],
        }


# =====================================================
# 主训练
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--prompt_max_length", type=int, default=2048)
    parser.add_argument("--response_max_length", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="L_OR 权重 (论文 λ); 0.2 = 激进版 (Llama-2 默认)")
    parser.add_argument("--warmup_steps", type=int, default=25,
                        help="数据小, 论文默认 5000 不适用")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_prompt_loss", action="store_true",
                        help="不在 prompt token 上算 NLL (官方默认 False, 即 prompt 也算 NLL)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false",
                        dest="gradient_checkpointing")
    # LoRA
    parser.add_argument("--enable_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)

    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

    if is_main:
        logger.info("=" * 70)
        logger.info("ORPO 训练 (4096 口径)")
        logger.info("=" * 70)
        logger.info(f"  model_path  = {args.model_path}")
        logger.info(f"  data_path   = {args.data_path}")
        logger.info(f"  output_dir  = {args.output_dir}")
        logger.info(f"  epochs={args.num_epochs} bs={args.batch_size} grad_accum={args.gradient_accumulation_steps}")
        logger.info(f"  lr={args.learning_rate} alpha={args.alpha} warmup={args.warmup_steps}")
        logger.info(f"  prompt_max={args.prompt_max_length} response_max={args.response_max_length}")
        logger.info(f"  LoRA r={args.lora_r} α={args.lora_alpha} dropout={args.lora_dropout}")
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # 与官方 orpo/main.py 一致

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if args.enable_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config=peft_config)
        if is_main:
            model.print_trainable_parameters()

    # ---- Dataset ----
    dataset = ORPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        prompt_max_length=args.prompt_max_length,
        response_max_length=args.response_max_length,
    )

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",   # 不打 wandb (官方依赖, 我们去掉)
        seed=args.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,   # ORPOTrainer 自定义 batch 字段, 不能让 trainer 删掉
    )

    # ---- ORPOTrainer (官方 src/orpo_trainer.py) ----
    # 注意: 官方 trainer compute_loss 里有 wandb.log(...), 我们 monkey-patch 替换掉
    import src.orpo_trainer as orpo_module
    if not hasattr(orpo_module, "_wandb_patched"):
        class _NoOpWandb:
            @staticmethod
            def log(*a, **k):
                pass
        orpo_module.wandb = _NoOpWandb()
        orpo_module._wandb_patched = True

    trainer = ORPOTrainer(
        alpha=args.alpha,
        pad=tokenizer.pad_token_id,
        disable_prompt_loss=args.disable_prompt_loss,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    if is_main:
        logger.info("开始训练 ...")
    t0 = time.time()
    trainer.train()
    if is_main:
        logger.info(f"训练完成, 耗时 {time.time()-t0:.0f}s")
        # 显式存最终 ckpt
        final_dir = os.path.join(args.output_dir, "checkpoint_final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"最终 ckpt → {final_dir}")


if __name__ == "__main__":
    main()

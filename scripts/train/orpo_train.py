"""orpo_train.py — ORPO 训练 (4096 口径, 调官方 orpo/src/orpo_trainer.py)

口径:
  prompt 上限: 2048 token
  answer 上限: 4096 token
  总长度: 2048 + 4096 = 6144

训练数据: datasets/train/train_data_orpo.json
  schema: {prompt, chosen, rejected, question_idx, ref_answer, chosen_source}

Loss (修改版, 强化首 token 信号防稀释):
  L_ORPO = L_SFT + λ · L_OR

  L_SFT  = -log P(fill_token | prompt)                        ← response 首 token, 不除
         + 1/(n_r - 1) · Σ_{t=2..n_r} -log P(y_t | prompt, y_<t)  ← 剩余 response token mean
         (prompt 部分不算 loss)

  L_OR   = -log σ(log odds_θ(y_w|x) / odds_θ(y_l|x))          ← 沿用官方实现, 不改

理由: 官方 ORPO 默认 L_SFT 对整个序列 mean, fill token 贡献被稀释到 ~0.03%。
新公式让 fill token 占 50% loss 量级, 信号放大 ~1500 倍, 同时保留续写学习。

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
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

_THIS_FILE = os.path.abspath(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# 调官方 ORPOTrainer (orpo/src/orpo_trainer.py) — 用 importlib 直接加载文件
# 避免依赖 orpo/src/__init__.py (官方仓库没有, 别动它)
import importlib.util as _ilu
_ORPO_TRAINER_PATH = os.path.join(_PROJECT_ROOT, "orpo", "src", "orpo_trainer.py")
if not os.path.exists(_ORPO_TRAINER_PATH):
    raise FileNotFoundError(f"orpo trainer not found: {_ORPO_TRAINER_PATH}")
_spec = _ilu.spec_from_file_location("_orpo_official_trainer", _ORPO_TRAINER_PATH)
_orpo_module = _ilu.module_from_spec(_spec)
# 预先注入 noop wandb, 避免官方 trainer 顶部 import wandb 失败 (它装了, 但执行 wandb.log 会卡)
import sys as _sys
class _NoOpWandb:
    @staticmethod
    def log(*a, **k):
        pass
    @staticmethod
    def init(*a, **k):
        pass
_sys.modules.setdefault("wandb", type(_sys)("wandb"))   # 占位防止 ImportError
import wandb as _wandb_imported
if not hasattr(_wandb_imported, "_orpo_patched"):
    _wandb_imported.log = _NoOpWandb.log
    _wandb_imported.init = _NoOpWandb.init
    _wandb_imported._orpo_patched = True
_spec.loader.exec_module(_orpo_module)
ORPOTrainer = _orpo_module.ORPOTrainer

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
# 自定义 ORPOTrainer: 覆盖 compute_loss, 强化首 token 信号
# =====================================================
class FirstTokenSplitORPOTrainer(ORPOTrainer):
    """覆盖 compute_loss, 让 chosen response 首 token 不被稀释:

      L_SFT = -log P(y_w[0] | prompt)                    [整项独立, 不除]
            + 1/(n_r - 1) · Σ_{t=2..n_r} -log P(y_w[t] | prompt, y_w[<t])
            (prompt 部分跳过, 不算 loss)

    L_OR 沿用父类 compute_logps 逻辑 (chosen/rejected response mean odds), 不改。
    """

    def _compute_split_sft_loss(
        self,
        logits: torch.Tensor,            # [B, T, V]
        input_ids: torch.Tensor,         # [B, T]
        prompt_attention_mask: torch.Tensor,    # [B, T_p]   (与 logits 不同长度, 同 token id 序列)
        chosen_attention_mask: torch.Tensor,    # [B, T]
    ) -> torch.Tensor:
        """新版 L_SFT: 首 token loss + 1/(n-1) 剩余 mean, prompt 跳过。

        - response_mask: chosen attention mask 减去 prompt attention mask 部分,
          只在 response 区段为 1。
        - 首 token = response 区段第一个为 1 的位置。
        """
        # shift: 模型在 t-1 位置预测 t
        shift_logits = logits[:, :-1, :].contiguous()       # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()         # [B, T-1]

        # response mask: chosen[1:] - prompt[1:] (跟 compute_logps 同口径,
        # prompt_attention_mask 用 chosen_attention_mask 的形状对齐前 T_p 位)
        # 这里 prompt_attention_mask 长度可能 != T, 需要对齐
        T = chosen_attention_mask.shape[1]
        T_p = prompt_attention_mask.shape[1]
        # 把 prompt_attention_mask pad 到 T 长度 (左对齐, 后面补 0)
        if T_p < T:
            pad = torch.zeros(prompt_attention_mask.shape[0], T - T_p,
                              dtype=prompt_attention_mask.dtype,
                              device=prompt_attention_mask.device)
            prompt_mask_padded = torch.cat([prompt_attention_mask, pad], dim=1)
        elif T_p > T:
            prompt_mask_padded = prompt_attention_mask[:, :T]
        else:
            prompt_mask_padded = prompt_attention_mask

        # response_mask 在 token 维度对齐 shift_labels (即 [:, 1:])
        response_mask = chosen_attention_mask[:, 1:] - prompt_mask_padded[:, 1:]
        response_mask = response_mask.clamp(min=0)           # 防负数 (理论不会, 保险)
        # response_mask: [B, T-1], 1 = 该位置是 response token, 0 = prompt 或 pad

        # 每 token CE
        ce_per_token = F.cross_entropy(
            shift_logits.transpose(1, 2),                    # [B, V, T-1]
            shift_labels,                                    # [B, T-1]
            reduction="none",
        )                                                    # [B, T-1]

        # 找每条样本 response 第一个 token 的位置
        # response_mask 第一个 1 的 idx
        # (response_mask 必非空, 因为 chosen 至少有一个 response token)
        B = response_mask.shape[0]
        first_idx = torch.argmax(response_mask, dim=1)       # [B], 第一个 1 出现位置

        # 首 token CE
        first_ce = ce_per_token.gather(
            1, first_idx.unsqueeze(1)
        ).squeeze(1)                                         # [B]

        # 剩余 response token CE (response_mask 排除首 token 位置)
        remain_mask = response_mask.clone().to(ce_per_token.dtype)
        remain_mask.scatter_(1, first_idx.unsqueeze(1), 0)   # 把首 token 位置归零
        remain_sum = (ce_per_token * remain_mask).sum(dim=1)         # [B]
        remain_count = remain_mask.sum(dim=1).clamp(min=1.0)         # [B], 防 div0
        remain_mean = remain_sum / remain_count                      # [B]

        # L_SFT 每样本 = first_ce + remain_mean, 然后 batch mean
        sft_per_sample = first_ce + remain_mean              # [B]
        return sft_per_sample.mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            inputs.pop("labels")

        # ---- forward chosen 和 rejected ----
        outputs_pos = model(
            input_ids=inputs["positive_input_ids"],
            attention_mask=inputs["positive_attention_mask"],
            output_hidden_states=True,
        )
        outputs_neg = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
            output_hidden_states=True,
        )

        # ---- L_SFT (新公式: 首 token + 1/(n-1) 剩余 mean, prompt 跳过) ----
        pos_loss = self._compute_split_sft_loss(
            logits=outputs_pos.logits,
            input_ids=inputs["positive_input_ids"],
            prompt_attention_mask=inputs["attention_mask"],
            chosen_attention_mask=inputs["positive_attention_mask"],
        )

        # ---- L_OR (沿用父类 compute_logps, 不改) ----
        pos_prob = self.compute_logps(
            prompt_attention_mask=inputs["attention_mask"],
            chosen_inputs=inputs["positive_input_ids"],
            chosen_attention_mask=inputs["positive_attention_mask"],
            logits=outputs_pos.logits,
        )
        neg_prob = self.compute_logps(
            prompt_attention_mask=inputs["attention_mask"],
            chosen_inputs=inputs["negative_input_ids"],
            chosen_attention_mask=inputs["negative_attention_mask"],
            logits=outputs_neg.logits,
        )

        log_odds = (pos_prob - neg_prob) - (
            torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob))
        )
        sig_ratio = torch.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)

        # ---- 总 loss ----
        loss = (pos_loss - self.alpha * ratio.mean()).to(dtype=torch.bfloat16)

        # 简单日志 (避免依赖 wandb)
        if (
            hasattr(self, "state")
            and self.state is not None
            and self.state.global_step % 5 == 0
            and (not dist.is_initialized() or dist.get_rank() == 0)
        ):
            logger.info(
                "[step %d] L_SFT=%.4f L_OR=%.4f total=%.4f pos_logp=%.4f neg_logp=%.4f",
                self.state.global_step,
                pos_loss.item(),
                ratio.mean().item(),
                loss.item(),
                pos_prob.mean().item(),
                neg_prob.mean().item(),
            )

        return (loss, outputs_pos) if return_outputs else loss


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
        logger.info("ORPO 训练 (4096 口径, 首 token split L_SFT)")
        logger.info("=" * 70)
        logger.info(f"  model_path  = {args.model_path}")
        logger.info(f"  data_path   = {args.data_path}")
        logger.info(f"  output_dir  = {args.output_dir}")
        logger.info(f"  epochs={args.num_epochs} bs={args.batch_size} grad_accum={args.gradient_accumulation_steps}")
        logger.info(f"  lr={args.learning_rate} alpha={args.alpha} warmup={args.warmup_steps}")
        logger.info(f"  prompt_max={args.prompt_max_length} response_max={args.response_max_length}")
        logger.info(f"  LoRA r={args.lora_r} α={args.lora_alpha} dropout={args.lora_dropout}")
        logger.info(f"  L_SFT = -log P(y_w[0]|x) + 1/(n_r-1) · Σ -log P(y_w[t]|...) (prompt 跳过)")
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

    # ---- Custom ORPOTrainer (覆盖 compute_loss, 不依赖 wandb) ----
    trainer = FirstTokenSplitORPOTrainer(
        alpha=args.alpha,
        pad=tokenizer.pad_token_id,
        disable_prompt_loss=False,   # 我们自己实现的 compute_loss 不用这个 flag, 但父类 init 仍要传
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


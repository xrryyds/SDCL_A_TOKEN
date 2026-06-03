"""pool_dist 路线训练脚本 (人为构造首 token 分布 + answer 后续蒸馏).

设计 (用户 2026-06-03 拍板):
  - 数据: datasets/train/train_data_pool_dist.json
          每条样本 = (question, fill_token_id, answer 含首 token, target_token_ids)
  - 序列布局: prompt + fill_token + answer_rest  (student / teacher 完全相同)
  - 首 token 位置 (即 prompt 末尾, 预测 fill_token 那一步):
        target_dist[v] = 1/N   if v in target_token_ids
                         0     otherwise
        L_first = KL(student || target_dist)
  - 后续 answer_rest 段 (从 fill_token 之后第 1 个 token 开始):
        L_rest = KL(student || teacher)  (teacher 现跑 = 冻结 base, student 挂 LoRA)
  - loss = L_first + L_rest_sum  (直接相加, 无加权)

对比 SDFT 路线: SDFT 是 prompt asymmetric (teacher 加 hint), 这里是 prompt 相同
              但首 token 目标分布人为构造 (N 个对的 candidate token uniform mask).

数据来源(由 scripts/build_train_data_pool_dist.py 产出):
  fill_multi_pool.json + fill_multi_pool_roll.json 的 rescued 题, 每题 N 个对的
  candidate 全展开为 N 条独立样本, 同题样本共享 target_token_ids.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 必须在 import torch 之前
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

# 让本文件既能作为脚本运行也能作为模块导入
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.train.a_token_sd import (  # noqa: E402
    SYSTEM_PROMPT,
    _first_int,
    _infer_lora_target_modules,
    normalize_question_text,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================
# DDP 工具
# =====================================================
def _is_torchrun_launched() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ddp_setup() -> Tuple[int, int, int, bool]:
    if not _is_torchrun_launched():
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(minutes=30),
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def _ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_rank(rank: int) -> bool:
    return rank == 0


# =====================================================
# Prompt 构造
# =====================================================
def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """student / teacher 共用的 prompt: SYSTEM + user(question)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =====================================================
# 数据加载与编码
# =====================================================
def _load_train_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    n_no_target = 0
    n_fill_not_in_target = 0
    for i, item in enumerate(data):
        src = item.get("source")
        if src != "pool_dist":
            logger.warning("跳过非 pool_dist 样本 idx=%d source=%r", i, src)
            continue
        if not item.get("question") or not item.get("answer"):
            logger.warning("跳过空 question/answer 样本 idx=%d", i)
            continue
        if item.get("fill_token_id") is None:
            logger.warning("跳过缺 fill_token_id 样本 idx=%d", i)
            continue
        target_ids = item.get("target_token_ids") or []
        if not target_ids:
            n_no_target += 1
            continue
        ftid = int(item["fill_token_id"])
        if ftid not in target_ids:
            # 防御: fill_token_id 必须在 target_token_ids 集合内
            n_fill_not_in_target += 1
            continue
        cleaned.append(item)
    if n_no_target > 0:
        logger.warning("跳过 %d 条无 target_token_ids 样本", n_no_target)
    if n_fill_not_in_target > 0:
        logger.warning(
            "跳过 %d 条 fill_token_id 不在 target_token_ids 集合内的样本",
            n_fill_not_in_target,
        )
    logger.info("加载训练数据 %d → 有效 %d", len(data), len(cleaned))
    return cleaned


def _encode_sample(
    tokenizer: AutoTokenizer,
    sample: Dict,
    max_prompt_length: int,
    max_answer_length: int,
) -> Optional[Dict]:
    """编码 pool_dist 样本.

    序列布局: prompt_ids + answer_ids
              其中 answer_ids[0] == fill_token_id (强制不变量)

    返回:
      input_ids        : List[int]
      prompt_len       : int
      answer_len       : int
      fill_token_id    : int
      fill_pos_in_seq  : int     == prompt_len  (fill_token 在 input_ids 中的索引)
      target_token_ids : List[int]
    """
    question = sample["question"]
    answer_text = str(sample.get("answer", ""))
    ftid = int(sample["fill_token_id"])
    target_ids = list(sample["target_token_ids"])

    prompt_text = _build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if len(answer_ids) == 0:
        return None
    if len(answer_ids) > max_answer_length:
        answer_ids = answer_ids[:max_answer_length]

    # answer 文本"应当"以 fill_token_text 开头, 但 BPE 可能合并: 强制对齐 token-id 层
    if answer_ids[0] != ftid:
        answer_ids = [ftid] + answer_ids
        if len(answer_ids) > max_answer_length:
            answer_ids = answer_ids[:max_answer_length]

    input_ids = list(prompt_ids) + list(answer_ids)
    return {
        "input_ids": input_ids,
        "prompt_len": len(prompt_ids),
        "answer_len": len(answer_ids),
        "fill_token_id": ftid,
        "fill_pos_in_seq": len(prompt_ids),
        "target_token_ids": target_ids,
    }


# =====================================================
# Batch padding (左填充)
# =====================================================
def _pad_left_batch(
    rows: List[List[int]],
    pad_token_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    max_len = max(len(r) for r in rows)
    batch_ids = torch.full(
        (len(rows), max_len), pad_token_id, dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        L = len(row)
        batch_ids[i, -L:] = torch.tensor(row, dtype=torch.long, device=device)
        attn_mask[i, -L:] = 1
    return batch_ids, attn_mask, max_len


# =====================================================
# Teacher / Student 构建
# =====================================================
def _build_teacher(model_path: str, device: str, dtype: torch.dtype):
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False
    logger.info("教师模型已加载到 %s", device)
    return teacher


def _build_student(
    model_path: str, device: str, dtype: torch.dtype,
    use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float,
    gradient_checkpointing: bool,
):
    student = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    if gradient_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    if use_lora:
        target_modules = _infer_lora_target_modules(student)
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
            lora_dropout=lora_dropout, task_type="CAUSAL_LM", bias="none",
        )
        student = get_peft_model(student, cfg).to(device)
    student.config.use_cache = False
    return student


# =====================================================
# 单 batch 计算 loss
# =====================================================
def _compute_batch_loss(
    student,
    teacher,
    encoded_batch: List[Dict],
    pad_token_id: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """pool_dist loss.

    student / teacher 输入序列完全相同 (prompt+fill_token+answer_rest):
      - 首 token 位置 (logits[fill_pos-1]):
          target_dist 为 N 个 target_token_ids 上 uniform 1/N
          L_first = KL(student || target_dist)
      - answer_rest 段 (logits[fill_pos .. end]):
          L_rest_per_tok = KL(student || teacher)
          L_rest_sample = L_rest_per_tok.sum()  # token-sum
      - sample_loss = L_first + L_rest_sample
      - 全 batch loss_sum = Σ sample_loss

    返回:
      loss_sum : 标量 tensor (带梯度)
      metrics  : per-sample mean kl_first / per-token mean kl_rest 等
    """
    rows = [s["input_ids"] for s in encoded_batch]
    s_input_ids, s_attn_mask, max_len = _pad_left_batch(rows, pad_token_id, device)
    row_lens = s_attn_mask.sum(dim=1).tolist()

    # 学生前向 (带梯度)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        student_logits = student(
            input_ids=s_input_ids, attention_mask=s_attn_mask
        ).logits  # [B, S, V]

    # 教师前向 (无梯度, 输入相同)
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            teacher_logits = teacher(
                input_ids=s_input_ids, attention_mask=s_attn_mask
            ).logits  # [B, S, V]

    loss_sum = student_logits.sum() * 0.0  # 零标量保留计算图
    n_samples = 0
    sum_kl_first = 0.0
    sum_kl_rest = 0.0
    n_rest_tok = 0
    sum_target_size = 0
    sum_p_first_at_target = 0.0  # 诊断: student 首 token 位置在 target 集合上的总概率

    for i, sample in enumerate(encoded_batch):
        L = row_lens[i]
        prompt_len = sample["prompt_len"]
        answer_len = sample["answer_len"]
        start = max_len - L

        # fill_token 位置 (绝对): start + prompt_len
        # 首 token 预测来自 logits[fill_pos_abs - 1] = logits[start + prompt_len - 1]
        fill_pos_abs = start + prompt_len
        first_logit_row = fill_pos_abs - 1

        # answer span = [fill_pos_abs, start + L - 1] (含)
        # 预测 answer 第 k 个 token 用 logits[fill_pos_abs - 1 + k]:
        #   k=1 → first_logit_row    (首 token, fill_token_id)
        #   k=2..answer_len → rest 段
        # 即 rest 段的 logits 行: [first_logit_row + 1, start + L - 1]
        rest_lo = first_logit_row + 1
        rest_hi = start + L - 1

        if first_logit_row < 0:
            continue

        target_ids = sample["target_token_ids"]
        N = len(target_ids)
        if N == 0:
            continue

        # ---- 首 token: forward KL(target || student) ----
        # target_dist[v] = 1/N if v in target_ids else 0
        # KL(target || student) = Σ_v target(v) * (log target(v) - log p_s(v))
        #                       = -log N - (1/N) * Σ_{v∈target} log p_s(v)
        # const -log N 不带梯度, 实际 loss 项:
        #   L_first = -(1/N) * Σ_{v∈target} log p_s(v)   (= N 个 target token 上 mean CE)
        # 数值有限, 梯度直接拉 N 个 target token 的 logit 上升.
        s_logits_first = student_logits[i, first_logit_row, :].float()  # [V]
        s_logp_first = F.log_softmax(s_logits_first, dim=-1)             # [V]
        target_idx = torch.tensor(
            target_ids, dtype=torch.long, device=device,
        )
        kl_first = -s_logp_first.index_select(0, target_idx).mean()
        loss_sum = loss_sum + kl_first

        # ---- answer rest 段: KL(student || teacher) token-sum ----
        if rest_hi >= rest_lo:
            s_logits_rest = student_logits[i, rest_lo : rest_hi + 1, :].float()
            t_logits_rest = teacher_logits[i, rest_lo : rest_hi + 1, :].float()
            s_logp_rest = F.log_softmax(s_logits_rest, dim=-1)
            t_logp_rest = F.log_softmax(t_logits_rest, dim=-1)
            kl_rest_per_tok = (
                s_logp_rest.exp() * (s_logp_rest - t_logp_rest)
            ).sum(dim=-1).clamp(min=0.0)  # [T-1]
            kl_rest_sample_sum = kl_rest_per_tok.sum()
            loss_sum = loss_sum + kl_rest_sample_sum
            sum_kl_rest += kl_rest_per_tok.mean().detach().item()
            n_rest_tok += s_logits_rest.size(0)

        n_samples += 1
        sum_kl_first += kl_first.detach().item()
        sum_target_size += N
        with torch.no_grad():
            p_first = F.softmax(s_logits_first, dim=-1)
            sum_p_first_at_target += float(p_first.index_select(0, target_idx).sum().item())

    if n_samples == 0:
        zero = student_logits.sum() * 0.0
        return zero, {
            "loss": 0.0, "n_samples": 0,
            "kl_first": 0.0, "kl_rest": 0.0,
            "avg_target_size": 0.0,
            "avg_p_first_at_target": 0.0,
            "loss_sum_raw": 0.0,
        }

    metrics = {
        "n_samples": n_samples,
        "kl_first": sum_kl_first / n_samples,
        "kl_rest": sum_kl_rest / max(n_samples, 1),
        "avg_target_size": sum_target_size / n_samples,
        "avg_p_first_at_target": sum_p_first_at_target / n_samples,
        "loss_sum_raw": loss_sum.detach().item(),
    }
    return loss_sum, metrics


# =====================================================
# 训练主入口
# =====================================================
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
            logger.removeHandler(h)
            h.close()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return (
        os.path.join(output_dir, "step_metrics.jsonl"),
        os.path.join(output_dir, "epoch_metrics.jsonl"),
    )


def train_a_token_pool_dist(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_prompt_length: int = 2048,
    max_answer_length: int = 4096,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    gradient_checkpointing: bool = True,
    log_interval: int = 10,
    save_total_limit: int = 5,
    save_steps: int = 0,
    seed: int = 42,
    device_ids: Optional[List[int]] = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("pool_dist 训练需要 CUDA。")
    torch.manual_seed(seed)

    rank, local_rank, world_size, is_ddp = _ddp_setup()
    is_main = _is_main_rank(rank)

    if is_ddp:
        device = f"cuda:{local_rank}"
        if is_main:
            logger.info("DDP 模式: world_size=%d", world_size)
    else:
        if device_ids is None:
            device_ids = [0]
        device = f"cuda:{device_ids[0]}"
        logger.info("单卡模式: device=%s", device)

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        step_log_file, epoch_log_file = setup_logging(output_dir)
    else:
        step_log_file, epoch_log_file = None, None
    if is_ddp:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        eos_int = _first_int(tokenizer.eos_token_id)
        if eos_int is None:
            raise ValueError("tokenizer 缺 pad_token_id 且 eos 不可解析。")
        tokenizer.pad_token_id = eos_int

    raw_data = _load_train_data(data_path)
    encoded: List[Dict] = []
    for s in raw_data:
        e = _encode_sample(tokenizer, s, max_prompt_length, max_answer_length)
        if e is not None:
            encoded.append(e)
    if is_main:
        logger.info("编码后训练样本数 (全量): %d", len(encoded))
    if not encoded:
        raise RuntimeError("训练数据为空。")

    dtype = torch.bfloat16
    teacher = _build_teacher(model_path, device, dtype)
    student = _build_student(
        model_path, device, dtype,
        use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, gradient_checkpointing=gradient_checkpointing,
    )
    if is_ddp:
        student = DDP(
            student, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    use_amp = True
    amp_dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    pad_id = tokenizer.pad_token_id
    global_step = 0

    if is_ddp:
        n_full = len(encoded)
        if n_full % world_size != 0:
            pad_n = world_size - (n_full % world_size)
            encoded_balanced = encoded + encoded[:pad_n]
        else:
            encoded_balanced = encoded
        local_encoded = encoded_balanced[rank::world_size]
        if is_main:
            logger.info(
                "rank=%d/%d 分到 %d 条样本 (全量 %d → 补齐 %d)",
                rank, world_size, len(local_encoded), n_full, len(encoded_balanced),
            )
    else:
        local_encoded = encoded

    rng = torch.Generator().manual_seed(seed + rank)

    n_batches_per_epoch = (len(local_encoded) + batch_size - 1) // batch_size
    opt_steps_per_epoch = max(
        1, (n_batches_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    )
    total_opt_steps = max(1, opt_steps_per_epoch * num_epochs)
    warmup_opt_steps = min(200, max(1, total_opt_steps // 10))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_opt_steps, num_training_steps=total_opt_steps,
    )
    if is_main:
        logger.info(
            "LR schedule: warmup=%d / total=%d optimizer steps (lr_max=%g → 0)",
            warmup_opt_steps, total_opt_steps, learning_rate,
        )

    for epoch in range(1, num_epochs + 1):
        if is_main:
            logger.info("--- Epoch %d/%d ---", epoch, num_epochs)
        student.train()
        order = torch.randperm(len(local_encoded), generator=rng).tolist()

        ep_loss = 0.0
        ep_steps = 0
        ep_n_samples = 0
        win_loss: List[float] = []
        win_kl_first: List[float] = []
        win_kl_rest: List[float] = []
        win_p_at_target: List[float] = []
        win_target_size: List[float] = []
        win_n_samples = 0

        n_batches = (len(local_encoded) + batch_size - 1) // batch_size
        if is_main:
            progress = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        else:
            progress = range(n_batches)
        optimizer.zero_grad()
        for bi in progress:
            ids = order[bi * batch_size : (bi + 1) * batch_size]
            batch = [local_encoded[j] for j in ids]

            is_last = bi == n_batches - 1
            do_step = ((global_step + 1) % gradient_accumulation_steps == 0) or is_last

            if is_ddp and not do_step:
                sync_ctx = student.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()

            with sync_ctx:
                loss_sum, metrics = _compute_batch_loss(
                    student=student, teacher=teacher,
                    encoded_batch=batch,
                    pad_token_id=pad_id,
                    device=device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                )
                loss = loss_sum
                (loss / gradient_accumulation_steps).backward()

            metrics["loss"] = loss.detach().item()

            global_step += 1
            if do_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (
                    is_main and save_steps > 0
                    and global_step % save_steps == 0
                ):
                    ckpt_dir = os.path.join(
                        output_dir, f"checkpoint_step_{global_step}"
                    )
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = student.module if is_ddp else student
                    model_to_save.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Step ckpt saved → %s", ckpt_dir)

            ep_loss += metrics["loss"]
            ep_steps += 1
            ep_n_samples += metrics["n_samples"]
            win_loss.append(metrics["loss"])
            if metrics["n_samples"] > 0:
                win_kl_first.append(metrics["kl_first"])
                win_kl_rest.append(metrics["kl_rest"])
                win_p_at_target.append(metrics["avg_p_first_at_target"])
                win_target_size.append(metrics["avg_target_size"])
                win_n_samples += metrics["n_samples"]

            if is_main and (global_step % log_interval == 0):
                avg_loss = sum(win_loss) / max(len(win_loss), 1)
                avg_kl_first = sum(win_kl_first) / max(len(win_kl_first), 1) if win_kl_first else 0.0
                avg_kl_rest = sum(win_kl_rest) / max(len(win_kl_rest), 1) if win_kl_rest else 0.0
                avg_p_at_target = sum(win_p_at_target) / max(len(win_p_at_target), 1) if win_p_at_target else 0.0
                avg_target_size = sum(win_target_size) / max(len(win_target_size), 1) if win_target_size else 0.0
                rec = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "lr": scheduler.get_last_lr()[0],
                    "avg_loss": avg_loss,
                    "avg_kl_first": avg_kl_first,
                    "avg_kl_rest": avg_kl_rest,
                    "avg_p_first_at_target": avg_p_at_target,
                    "avg_target_size": avg_target_size,
                    "n_samples": win_n_samples,
                }
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info(
                    "[Step %d] epoch=%d lr=%.2e loss=%.4f kl_first=%.4f kl_rest=%.4f "
                    "p_target=%.4f |tgt|=%.2f n=%d",
                    global_step, epoch, scheduler.get_last_lr()[0],
                    avg_loss, avg_kl_first, avg_kl_rest,
                    avg_p_at_target, avg_target_size, win_n_samples,
                )
                win_loss.clear()
                win_kl_first.clear()
                win_kl_rest.clear()
                win_p_at_target.clear()
                win_target_size.clear()
                win_n_samples = 0

        ep_avg_loss = ep_loss / max(ep_steps, 1)
        if is_main:
            ep_record = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "avg_loss": ep_avg_loss,
                "n_samples": ep_n_samples,
                "steps": ep_steps,
            }
            with open(epoch_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(ep_record) + "\n")
            logger.info("=" * 60)
            logger.info("*** EPOCH %d/%d FINISHED ***", epoch, num_epochs)
            logger.info(
                "  avg_loss=%.4f n_samples=%d (rank0)",
                ep_avg_loss, ep_n_samples,
            )
            logger.info("=" * 60)

        if is_ddp:
            dist.barrier()

        if is_main and save_total_limit > 0:
            save_interval = max(1, num_epochs // save_total_limit)
            if epoch % save_interval == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model_to_save = student.module if is_ddp else student
                model_to_save.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info("Checkpoint saved → %s", ckpt_dir)

    if is_main:
        model_to_save = student.module if is_ddp else student
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("训练完成,已保存到 %s", output_dir)

    if is_ddp:
        dist.barrier()
        _ddp_cleanup()


# =====================================================
# CLI
# =====================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="pool_dist 训练 (人为构造首 token 分布 + answer 后续蒸馏)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="train_data_pool_dist.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_answer_length", type=int, default=4096)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_ids", type=str, default=None,
                        help="逗号分隔 GPU id (单卡模式用)")
    return parser.parse_args()


def main():
    args = _parse_args()
    device_ids: Optional[List[int]] = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    train_a_token_pool_dist(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_answer_length=args.max_answer_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
        log_interval=args.log_interval,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        seed=args.seed,
        device_ids=device_ids,
    )


if __name__ == "__main__":
    main()

"""SDFT v3 stage2 训练脚本 (off-policy KD + prompt asymmetry, student 起点是 pool_dist LoRA).

设计 (用户 2026-06-03 拍板):
  - stage1: pool_dist 训出来的 LoRA, 在 pool 上 +26.80% 但 corr 掉 8.43%, MATH -3%
  - stage2: 把 pool_dist LoRA merge 进 Base 作为 student 实际起点 (新 Base),
            然后挂一个全新 LoRA, 用 SDFT v1 的 loss 在 train_data_v3.json 全集
            (corr+roll+pool = 31916) 上继续训
  - teacher 仍是原始 Base (不合并 pool_dist LoRA), 目的是把 corr/MATH 拉回 Base 分布
  - 期望: corr/MATH 恢复, pool 通过 stage1 已经学到的能力 + stage2 的 hint asymmetric KD 保住

关键实现差异 (vs sdft_train.py):
  - _build_student 新增 base_lora_path 参数:
      load Base → PeftModel.from_pretrained(stage1 LoRA) → merge_and_unload()
      → 新 Base (含 stage1 增益)  → 再挂全新 LoRA → student
  - teacher 完全不变, 直接加载原 Base
  其余 loss / 数据格式 / DDP 全跟 sdft_train.py 一致.

继承自 sdft_train.py 的 SDFT 设计:
  - 数据 train_data_v3.json (corr 5456 + roll 2134 + pool 24326 = 31916)
  - corr_answer : teacher prompt == student prompt, 全 span 反向 KL(s‖t)
  - roll / pool : teacher 加 hint "Please start your answer with \"{token_text}\". {q}"
                 student 用正常 prompt; answer span 反向 KL(s‖t) token-一一对应
  - 加权: 全 w=1.0 纯 sum, 不归一化
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
# DDP 工具 (与 V3 一致)
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
# Prompt 构造 (核心: 区分 student / teacher)
# =====================================================
def _build_student_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """学生 prompt: 正常 SYSTEM_PROMPT + user(question)。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_teacher_prompt(
    tokenizer: AutoTokenizer, question: str, fill_token_text: str
) -> str:
    """教师 prompt: SYSTEM_PROMPT + user("Please start your answer with \"<ttxt>\". <q>")。

    SDFT 路线 hint: A 方案 (用户 2026-05-31 拍板)。
    fill_token_text 为空时退化为正常 prompt (corr 池路径)。
    """
    q_norm = normalize_question_text(question)
    if fill_token_text:
        user_content = (
            f'Please start your answer with "{fill_token_text}". {q_norm}'
        )
    else:
        user_content = q_norm
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
    for i, item in enumerate(data):
        src = item.get("source")
        if src not in ("corr_answer", "roll", "pool"):
            logger.warning("跳过未知 source 样本 idx=%d source=%r", i, src)
            continue
        if not item.get("question") or not item.get("answer"):
            logger.warning("跳过空 question/answer 样本 idx=%d", i)
            continue
        if src in ("roll", "pool"):
            # SDFT 路线: roll/pool 都需要 fill_token_text 作为 hint
            if not item.get("fill_token_text"):
                logger.warning(
                    "SDFT 路线 %s 样本缺 fill_token_text idx=%d,跳过", src, i
                )
                continue
        cleaned.append(item)
    logger.info("加载训练数据 %d → 有效 %d", len(data), len(cleaned))
    return cleaned


def _encode_sample(
    tokenizer: AutoTokenizer,
    sample: Dict,
    max_prompt_length: int,
    max_answer_length: int,
) -> Optional[Dict]:
    """编码 SDFT 样本。

    返回:
      student_prompt_ids : List[int]
      teacher_prompt_ids : List[int]  (corr 时 == student_prompt_ids)
      answer_ids         : List[int]  (共同 answer span)
      source             : str
    """
    src = sample["source"]
    question = sample["question"]
    answer_text = str(sample.get("answer", ""))
    fill_token_text = sample.get("fill_token_text", "") if src in ("roll", "pool") else ""

    student_text = _build_student_prompt(tokenizer, question)
    teacher_text = _build_teacher_prompt(tokenizer, question, fill_token_text)

    student_prompt_ids = tokenizer(student_text, add_special_tokens=False).input_ids
    teacher_prompt_ids = tokenizer(teacher_text, add_special_tokens=False).input_ids
    if len(student_prompt_ids) > max_prompt_length:
        student_prompt_ids = student_prompt_ids[-max_prompt_length:]
    if len(teacher_prompt_ids) > max_prompt_length:
        teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if len(answer_ids) == 0:
        return None
    if len(answer_ids) > max_answer_length:
        answer_ids = answer_ids[:max_answer_length]

    return {
        "source": src,
        "student_prompt_ids": list(student_prompt_ids),
        "teacher_prompt_ids": list(teacher_prompt_ids),
        "answer_ids": list(answer_ids),
        "student_input_ids": list(student_prompt_ids) + list(answer_ids),
        "teacher_input_ids": list(teacher_prompt_ids) + list(answer_ids),
        "student_prompt_len": len(student_prompt_ids),
        "teacher_prompt_len": len(teacher_prompt_ids),
        "answer_len": len(answer_ids),
    }


# =====================================================
# Batch padding (左填充,与 V3 一致)
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
# Teacher / Student 构建 (与 V3 一致)
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
    base_lora_path: Optional[str] = None,
):
    """stage2 student 构建.

    流程:
      1) Load Base
      2) 如果 base_lora_path 不为空:
           PeftModel.from_pretrained(Base, base_lora_path) → merge_and_unload()
         → 新 Base (含 stage1 LoRA 增益, 不再有 PEFT 包装)
      3) 在 (新 Base 或原 Base) 上挂一个全新 LoRA (use_lora=True 时)
    """
    from peft import PeftModel

    student = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # ---- stage2 关键: 把 stage1 LoRA merge 进 student 的 Base 权重 ----
    if base_lora_path is not None and base_lora_path != "":
        logger.info("加载 stage1 LoRA 并 merge 进 student Base: %s", base_lora_path)
        student = PeftModel.from_pretrained(student, base_lora_path).to(device)
        student = student.merge_and_unload()  # 返回普通 nn.Module, 不再有 PEFT 包装
        logger.info("stage1 LoRA 已 merge, student 实际权重 = Base + stage1_LoRA")

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
# 单 batch 计算 loss (SDFT 核心)
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
    """SDFT loss.

    每个样本:
      student forward [student_prompt + answer]  → logits, 取 answer span
      teacher forward [teacher_prompt + answer]  → logits, 取 answer span
      KL(student ‖ teacher) 在 answer span 上 token-一一对应

    返回:
      loss_sum : 标量 tensor (带梯度) — 全 batch token-sum KL
      metrics  : per-source per-token mean KL 等
    """
    # student 和 teacher 序列长度可能不同 (teacher 多 hint tokens),
    # 分别 pad 成两个 batch tensor。
    s_rows = [s["student_input_ids"] for s in encoded_batch]
    t_rows = [s["teacher_input_ids"] for s in encoded_batch]
    s_input_ids, s_attn_mask, s_max_len = _pad_left_batch(s_rows, pad_token_id, device)
    t_input_ids, t_attn_mask, t_max_len = _pad_left_batch(t_rows, pad_token_id, device)
    s_row_lens = s_attn_mask.sum(dim=1).tolist()
    t_row_lens = t_attn_mask.sum(dim=1).tolist()

    # 学生前向 (带梯度)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        student_logits = student(
            input_ids=s_input_ids, attention_mask=s_attn_mask
        ).logits  # [B, S_s, V]

    # 教师前向 (无梯度)
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            teacher_logits = teacher(
                input_ids=t_input_ids, attention_mask=t_attn_mask
            ).logits  # [B, S_t, V]

    # 逐样本算 KL
    kl_sum = student_logits.sum() * 0.0  # 零标量保留计算图
    n_corr = 0
    n_roll = 0
    n_pool = 0
    sum_kl_corr = 0.0
    sum_kl_roll = 0.0
    sum_kl_pool = 0.0

    for i, sample in enumerate(encoded_batch):
        src = sample["source"]
        answer_len = sample["answer_len"]
        s_prompt_len = sample["student_prompt_len"]
        t_prompt_len = sample["teacher_prompt_len"]
        s_L = s_row_lens[i]
        t_L = t_row_lens[i]
        s_start = s_max_len - s_L  # 左 padding 起点
        t_start = t_max_len - t_L

        # 预测 answer 第 k 个 token 用 logits[start+prompt_len-1+k]
        # answer 第 1..answer_len 个 token 的 logits 行: [pred_lo, pred_hi] (含)
        s_pred_lo = s_start + s_prompt_len - 1
        s_pred_hi = s_start + s_L - 1
        t_pred_lo = t_start + t_prompt_len - 1
        t_pred_hi = t_start + t_L - 1

        if s_pred_lo < 0 or s_pred_hi < s_pred_lo:
            continue
        if t_pred_lo < 0 or t_pred_hi < t_pred_lo:
            continue

        # 两边 answer span 长度应相同 (== answer_len)
        s_span_len = s_pred_hi - s_pred_lo + 1
        t_span_len = t_pred_hi - t_pred_lo + 1
        if s_span_len != t_span_len:
            # 防御: 极端情况 answer 编码错位,跳过
            logger.warning(
                "answer span len mismatch i=%d src=%s s=%d t=%d, 跳过",
                i, src, s_span_len, t_span_len,
            )
            continue

        s_logits = student_logits[i, s_pred_lo : s_pred_hi + 1, :].float()  # [T, V]
        t_logits = teacher_logits[i, t_pred_lo : t_pred_hi + 1, :].float()  # [T, V]
        s_logp = F.log_softmax(s_logits, dim=-1)
        t_logp = F.log_softmax(t_logits, dim=-1)
        # 反向 KL(student ‖ teacher)
        kl_per_tok = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).clamp(min=0.0)  # [T]

        # 2026-06-02 改: roll/pool 池首 token KL 单独算 + 其余 token 取 mean.
        # 原因: SDFT teacher 加 hint, student 不加, 整段 sum KL 把首 token 信号
        # 稀释到 ~2000 token, pool 评测 +0.40% 学不到. 拆开后让首 token 信号
        # 不再被后续稀释, 期望 pool 池能从 +0.40% 提升.
        # corr 池 teacher_prompt == student_prompt, 跟原来一样 sum 累加 (不变).
        if src in ("roll", "pool") and kl_per_tok.numel() >= 2:
            sample_kl = kl_per_tok[0] + kl_per_tok[1:].mean()
        else:
            sample_kl = kl_per_tok.sum()
        kl_sum = kl_sum + sample_kl

        kl_mean = kl_per_tok.mean().detach().item()
        if src == "corr_answer":
            n_corr += 1
            sum_kl_corr += kl_mean
        elif src == "roll":
            n_roll += 1
            sum_kl_roll += kl_mean
        elif src == "pool":
            n_pool += 1
            sum_kl_pool += kl_mean

    if (n_corr + n_roll + n_pool) == 0:
        zero = student_logits.sum() * 0.0
        return zero, {
            "loss": 0.0, "n_corr": 0, "n_roll": 0, "n_pool": 0,
            "kl_corr": 0.0, "kl_roll": 0.0, "kl_pool": 0.0,
            "kl_sum_raw": 0.0,
        }

    metrics = {
        "n_corr": n_corr,
        "n_roll": n_roll,
        "n_pool": n_pool,
        "kl_corr": sum_kl_corr / max(n_corr, 1),
        "kl_roll": sum_kl_roll / max(n_roll, 1),
        "kl_pool": sum_kl_pool / max(n_pool, 1),
        "kl_sum_raw": kl_sum.detach().item(),
    }
    return kl_sum, metrics


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


def train_a_token_sdft(
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
    base_lora_path: Optional[str] = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("SDFT 训练需要 CUDA。")
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
        base_lora_path=base_lora_path,
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

    # 数据按 rank 切分 (补齐到 world_size 整除,防 all-reduce 死锁)
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

    # LR scheduler: warmup + cosine decay
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
        ep_n_corr = 0
        ep_n_roll = 0
        ep_n_pool = 0
        win_loss: List[float] = []
        win_kl_corr: List[float] = []
        win_kl_roll: List[float] = []
        win_kl_pool: List[float] = []
        win_kl_raw: List[float] = []
        win_n_corr = 0
        win_n_roll = 0
        win_n_pool = 0

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
                kl_sum, metrics = _compute_batch_loss(
                    student=student, teacher=teacher,
                    encoded_batch=batch,
                    pad_token_id=pad_id,
                    device=device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                )
                # 三池纯 sum,不归一化 (与 V3 设计点 3 一致)
                loss = kl_sum
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
            ep_n_corr += metrics["n_corr"]
            ep_n_roll += metrics["n_roll"]
            ep_n_pool += metrics["n_pool"]
            win_loss.append(metrics["loss"])
            win_kl_raw.append(float(metrics.get("kl_sum_raw", 0.0)))
            if metrics["n_corr"] > 0:
                win_kl_corr.append(metrics["kl_corr"])
                win_n_corr += metrics["n_corr"]
            if metrics["n_roll"] > 0:
                win_kl_roll.append(metrics["kl_roll"])
                win_n_roll += metrics["n_roll"]
            if metrics["n_pool"] > 0:
                win_kl_pool.append(metrics["kl_pool"])
                win_n_pool += metrics["n_pool"]

            if is_main and (global_step % log_interval == 0):
                avg_loss = sum(win_loss) / max(len(win_loss), 1)
                avg_kl_corr = sum(win_kl_corr) / max(len(win_kl_corr), 1) if win_kl_corr else 0.0
                avg_kl_roll = sum(win_kl_roll) / max(len(win_kl_roll), 1) if win_kl_roll else 0.0
                avg_kl_pool = sum(win_kl_pool) / max(len(win_kl_pool), 1) if win_kl_pool else 0.0
                avg_kl_raw = sum(win_kl_raw) / max(len(win_kl_raw), 1)
                rec = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "lr": scheduler.get_last_lr()[0],
                    "avg_loss": avg_loss,
                    "avg_kl_corr": avg_kl_corr,
                    "avg_kl_roll": avg_kl_roll,
                    "avg_kl_pool": avg_kl_pool,
                    "avg_kl_sum_raw": avg_kl_raw,
                    "n_corr": win_n_corr,
                    "n_roll": win_n_roll,
                    "n_pool": win_n_pool,
                }
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info(
                    "[Step %d] epoch=%d lr=%.2e loss=%.6f kl_corr=%.4f kl_roll=%.4f kl_pool=%.4f "
                    "n_corr=%d n_roll=%d n_pool=%d",
                    global_step, epoch, scheduler.get_last_lr()[0],
                    avg_loss, avg_kl_corr, avg_kl_roll, avg_kl_pool,
                    win_n_corr, win_n_roll, win_n_pool,
                )
                win_loss.clear()
                win_kl_corr.clear()
                win_kl_roll.clear()
                win_kl_pool.clear()
                win_kl_raw.clear()
                win_n_corr = 0
                win_n_roll = 0
                win_n_pool = 0

        ep_avg_loss = ep_loss / max(ep_steps, 1)
        if is_main:
            ep_record = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "avg_loss": ep_avg_loss,
                "n_corr": ep_n_corr,
                "n_roll": ep_n_roll,
                "n_pool": ep_n_pool,
                "steps": ep_steps,
            }
            with open(epoch_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(ep_record) + "\n")
            logger.info("=" * 60)
            logger.info("*** EPOCH %d/%d FINISHED ***", epoch, num_epochs)
            logger.info(
                "  avg_loss=%.6f n_corr=%d n_roll=%d n_pool=%d (rank0)",
                ep_avg_loss, ep_n_corr, ep_n_roll, ep_n_pool,
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
        description="SDFT v3 stage2 训练 (起点是 pool_dist LoRA, 用 SDFT v1 loss)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="train_data_v3.json (复用 V3 三池数据)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--base_lora_path", type=str, required=True,
        help="stage1 LoRA 路径 (会 merge 进 student Base 作为实际起点); "
             "传空字符串可退化为标准 SDFT v1.",
    )
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
    train_a_token_sdft(
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
        base_lora_path=args.base_lora_path,
    )


if __name__ == "__main__":
    main()

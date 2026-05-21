"""方法3：混合蒸馏训练（按 source 字段分支）。

训练数据：a_token_train_data.json（由 scripts/train/a_token_sdcl.py merge 产出）
  - source == "corr_answer"  : 序列  prompt + answer，每个 answer 位置用 KL(学生 || 教师)
  - source == "fill_correct" : 序列  prompt + fill_token + 续写
        · 首 token 位置（prompt 末尾预测下一个 token 的位置）：CE(学生 logits, fill_token_id)
        · 后续 token 位置                                     : KL(学生 || 教师)

教师统一为初始模型（model_path 指向的原始模型，参数冻结、不参与梯度）。

实现要点（与 scripts/train/a_token_sd.py 风格一致）：
  - LoRA + bf16 + eager；LoRA 只挂在学生上，教师为冻结的全参 base
  - 学生与教师同一 batch 同时前向；教师在 torch.no_grad 下走，detach
  - left-pad 用于稳定取 prompt 末尾位置；标签用 attention mask + answer mask 控制
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
    """是否由 torchrun / torch.distributed 启动。"""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ddp_setup() -> Tuple[int, int, int, bool]:
    """初始化 DDP 进程组。

    Returns:
        rank, local_rank, world_size, is_ddp
    """
    if not _is_torchrun_launched():
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not dist.is_initialized():
        # 加 timeout,避免 rank 之间数据切片不齐 / forward 异常时 NCCL 永远等待 → 进程
        # hang 在 D 状态,nvidia-smi 还显示显存在但 process GPU memory=0(僵尸 context)。
        # 30 分钟够覆盖单次 forward+backward 即使是 16K 长样本。
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=30),
        )
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def _ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_rank(rank: int) -> bool:
    return rank == 0


# =====================================================
# 数据
# =====================================================
def _load_train_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"训练数据应为 list：{path}")
    cleaned = []
    for i, item in enumerate(data):
        src = item.get("source")
        if src not in ("corr_answer", "fill_correct"):
            logger.warning(
                "跳过未知 source 的样本 idx=%d source=%r", i, src
            )
            continue
        if not item.get("question") or not item.get("answer"):
            logger.warning("跳过空 question/answer 样本 idx=%d", i)
            continue
        if src == "fill_correct":
            if item.get("fill_token_id") is None:
                logger.warning(
                    "fill_correct 样本缺少 fill_token_id idx=%d，跳过", i
                )
                continue
        cleaned.append(item)
    logger.info("加载训练数据 %d → 有效 %d", len(data), len(cleaned))
    return cleaned


def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =====================================================
# 单样本编码
# =====================================================
def _encode_sample(
    tokenizer: AutoTokenizer,
    sample: Dict,
    max_prompt_length: int,
    max_answer_length: int,
) -> Optional[Dict]:
    """把一条样本编成 token ids。

    返回 dict:
        input_ids        : List[int]  完整序列 = prompt_ids + answer_ids
        prompt_len       : int        prompt 部分 token 数
        answer_len       : int        answer 部分 token 数
        source           : str
        fill_token_id    : Optional[int]
        fill_pos_in_seq  : Optional[int]  fill_correct 时 = prompt_len（即首 token 在序列中的位置 = 教师/学生在 input_ids[prompt_len-1] 位置预测出的那个 token）
                            None 表示样本不需要在该位置施加 CE。
    """
    prompt_text = _build_prompt(tokenizer, sample["question"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    answer_text = str(sample.get("answer", ""))
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if len(answer_ids) == 0:
        return None
    if len(answer_ids) > max_answer_length:
        answer_ids = answer_ids[:max_answer_length]

    src = sample["source"]
    fill_token_id: Optional[int] = None
    fill_pos_in_seq: Optional[int] = None

    if src == "fill_correct":
        ftid = int(sample["fill_token_id"])
        # answer 文本里"应当"以 fill_token_text 开头；为稳健起见做一致性校验：
        # 若 answer 第一个 token 已经是 ftid，就保留；否则在 prompt 后强制塞 ftid
        # 作为 answer 的起首（与方法1的拼接方式保持一致）。
        if answer_ids[0] != ftid:
            # 罕见：tokenizer 把 fill_token_text 跟后续字符合并了。
            # 用 token id 拼接保证首 token 严格等于 ftid。
            answer_ids = [ftid] + answer_ids
        fill_token_id = ftid
        # 在完整序列 input_ids 中，fill token 位于 index = len(prompt_ids)
        # 训练时学生在 logits[len(prompt_ids) - 1] 位置预测它（teacher-forcing 偏移）。
        fill_pos_in_seq = len(prompt_ids)

    input_ids = list(prompt_ids) + list(answer_ids)
    return {
        "input_ids": input_ids,
        "prompt_len": len(prompt_ids),
        "answer_len": len(answer_ids),
        "source": src,
        "fill_token_id": fill_token_id,
        "fill_pos_in_seq": fill_pos_in_seq,
    }


# =====================================================
# Batch padding（左填充：与 a_token_sd.py 保持一致）
# =====================================================
def _pad_left_batch(
    rows: List[List[int]],
    pad_token_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """左填充：保证最后一个有效 token 在 max_len-1 位置。"""
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
# 教师 / 学生构建
# =====================================================
def _build_teacher(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    device_map=None,
    max_memory: Optional[Dict] = None,
):
    """初始模型作为教师。冻结、eval 模式、不挂 LoRA。

    Args:
        device     : 当 device_map=None 时整模型放到该 device。
        device_map : 多卡分布字符串 / dict（"auto"/"balanced"/{"":0} 等）。
                     设置后忽略 device 参数（accelerate 自动放置）。
        max_memory : 配合 device_map="balanced" 用，限制各卡可用显存
                     （例如把学生卡设成 0 防止教师抢占）。
    """
    if device_map is not None:
        kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        teacher = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        logger.info(
            "教师模型已加载，device_map=%s，max_memory=%s",
            device_map,
            max_memory,
        )
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        logger.info("教师模型已加载到 %s", device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False
    return teacher


def _build_student(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
):
    student = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # 重要：grad ckpt 必须在 base model 上启用，且需要让 input 的 grad 接通，
    # 否则 LoRA 的梯度链不通（PEFT + grad_ckpt 的常见坑）。
    # 顺序：先在 base 上启用 grad_ckpt → enable_input_require_grads → 包 LoRA
    if gradient_checkpointing:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()

    if use_lora:
        target_modules = _infer_lora_target_modules(student)
        cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
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
    student_device: str,
    teacher_device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    ce_weight: float,
    use_ema: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """对一个混合 batch 计算 loss 的分项 sum。

    返回:
      ce_sum   : 本 batch 内所有 fill_correct 样本首 token CE 的总和(标量 tensor,带梯度)
      kl_sum   : 本 batch 内所有 token-level KL 的总和(标量 tensor,带梯度)
                 包括 corr_answer 全 answer span 的 KL,以及 fill_correct 后续 token 的 KL
                 ⚠ use_ema=False 时,kl_sum 改为存"样本平均的 (ce_weight*CE + KL_mean) legacy
                    loss",ce_sum 强制为 0,调用方直接用 kl_sum 当 loss(语义不对名字也是这个,
                    保留旧签名免改外层结构)。
      metrics  : 日志/统计用,per-token 平均后的指标(纯标量,不带梯度)

    Loss 构成(每条样本独立算):
      EMA 模式 (use_ema=True,默认):
        - corr_answer  : token 级 KL(student || teacher) 在 answer span 上做 sum,累入 kl_sum
        - fill_correct : 首 token 处 CE,累入 ce_sum;后续 token 的 KL 做 sum,累入 kl_sum
        外层做 EMA 归一化后用 lambda_ce / lambda_kl 加权。

      Legacy 模式 (use_ema=False):
        - corr_answer  : KL.mean()  (per-token 平均,与最早老版本一致)
        - fill_correct : ce_weight * CE + KL_rest.mean()
        每条样本独立算 loss,样本间取平均 → 直接放进 kl_sum 返回。
        ⚠ 此模式下 kl_sum.mean() ≈ 老版本 ce_weight ≈ 2 时的 loss 量级 ~15。

    多卡:student / teacher 可能在不同 device 上。
    """
    # 左填充 batch（按 student device 建张量；teacher 只需相同 token id，搬一次）
    rows = [s["input_ids"] for s in encoded_batch]
    s_input_ids, s_attn_mask, max_len = _pad_left_batch(
        rows, pad_token_id, student_device
    )
    if teacher_device == student_device:
        t_input_ids = s_input_ids
        t_attn_mask = s_attn_mask
    else:
        t_input_ids = s_input_ids.to(teacher_device, non_blocking=True)
        t_attn_mask = s_attn_mask.to(teacher_device, non_blocking=True)

    row_lens = s_attn_mask.sum(dim=1).tolist()  # List[int]，每行有效长度

    # 学生前向（带梯度）
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        student_logits = student(
            input_ids=s_input_ids, attention_mask=s_attn_mask
        ).logits  # [B, S, V] on student_device

    # 教师前向（无梯度），随后搬回 student_device 与学生 logits 对齐设备
    with torch.no_grad():
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            teacher_logits = teacher(
                input_ids=t_input_ids, attention_mask=t_attn_mask
            ).logits  # [B, S, V] on teacher_device
    if teacher_device != student_device:
        teacher_logits = teacher_logits.to(student_device, non_blocking=True)

    # 注意：不要在这里整体 .float() 上升精度。
    # [B, S, V] 在 V≈15w 时 float32 会吃 ~7GB/张，单 batch 就 OOM。
    # 改为在每个样本内部对 sliced span（长度 T ≪ S）做 .float()。

    # 逐样本算 loss:
    #   EMA 模式:用 ce_sum / kl_sum 两个 token-sum 标量(由调用方做 EMA 归一化)
    #   Legacy 模式:用 sample_losses 收集每条样本的 (ce_weight*CE + KL.mean()) 标量,
    #              函数末尾 .stack().mean() 后通过 kl_sum 字段返回
    ce_sum = student_logits.sum() * 0.0   # 零标量,保留计算图设备
    kl_sum = student_logits.sum() * 0.0
    sample_losses: List[torch.Tensor] = []  # legacy 用
    n_corr = 0
    n_fill = 0
    sum_ce = 0.0
    sum_kl_corr = 0.0
    sum_kl_fill = 0.0
    n_kl_corr_tok = 0
    n_kl_fill_tok = 0
    n_ce_tok = 0

    for i, sample in enumerate(encoded_batch):
        L = row_lens[i]                  # 有效 token 数
        prompt_len = sample["prompt_len"]
        answer_len = sample["answer_len"]
        # 该行有效 token 的起始位置（左填充）
        start = max_len - L

        # 在序列层面：prompt 占据 [start, start+prompt_len)，answer 占据 [start+prompt_len, start+L)
        # 预测 input_ids[i, t] 来自 logits[i, t-1]，所以预测 answer 第 k 个 token 用 logits[i, start+prompt_len-1+k]
        pred_lo = start + prompt_len - 1     # 预测 answer 第 1 个 token 的 logits 行
        pred_hi = start + L - 1              # 预测 answer 最后 1 个 token 的 logits 行（含）
        # 对应的标签 input_ids 索引区间 [pred_lo+1, pred_hi+1]（含）
        if pred_lo < 0 or pred_hi < pred_lo:
            # 安全跳过（不应发生）
            continue

        if sample["source"] == "corr_answer":
            # 全 answer span KL；只对 sliced span 升 float32
            s_logits = student_logits[i, pred_lo : pred_hi + 1, :].float()   # [T, V]
            t_logits = teacher_logits[i, pred_lo : pred_hi + 1, :].float()   # [T, V]
            s_logp = F.log_softmax(s_logits, dim=-1)
            t_logp = F.log_softmax(t_logits, dim=-1)
            kl_per_tok = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).clamp(min=0.0)  # [T]
            if use_ema:
                # token-sum,EMA 模式拉齐量级
                kl_sum = kl_sum + kl_per_tok.sum()
            else:
                # legacy:per-token mean,与老版本一致
                sample_losses.append(kl_per_tok.mean())
            n_corr += 1
            sum_kl_corr += kl_per_tok.mean().detach().item()
            n_kl_corr_tok += s_logits.size(0)

        else:  # fill_correct
            # fill token 在序列中的位置（绝对）
            fill_pos_abs = start + sample["fill_pos_in_seq"]   # = start + prompt_len
            ce_logits = student_logits[i, fill_pos_abs - 1, :].float()    # [V]
            ce_target = torch.tensor(
                sample["fill_token_id"], dtype=torch.long, device=student_device
            )
            ce = F.cross_entropy(ce_logits.unsqueeze(0), ce_target.unsqueeze(0))
            if use_ema:
                ce_sum = ce_sum + ce
            sum_ce += ce.detach().item()
            n_ce_tok += 1

            # 后续 token KL：从 answer 第 2 个 token 开始（即 logits[i, pred_lo+1 .. pred_hi]）
            rest_lo = pred_lo + 1
            rest_hi = pred_hi
            kl_rest_mean: Optional[torch.Tensor] = None
            if rest_hi >= rest_lo:
                s_logits = student_logits[i, rest_lo : rest_hi + 1, :].float()
                t_logits = teacher_logits[i, rest_lo : rest_hi + 1, :].float()
                s_logp = F.log_softmax(s_logits, dim=-1)
                t_logp = F.log_softmax(t_logits, dim=-1)
                kl_rest_per_tok = (
                    (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).clamp(min=0.0)
                )  # [T-1]
                if use_ema:
                    kl_sum = kl_sum + kl_rest_per_tok.sum()
                else:
                    kl_rest_mean = kl_rest_per_tok.mean()
                sum_kl_fill += kl_rest_per_tok.mean().detach().item()
                n_kl_fill_tok += s_logits.size(0)
            if not use_ema:
                # legacy:每条 fill 样本的 loss = ce_weight*CE + KL_rest.mean()
                if kl_rest_mean is not None:
                    sample_losses.append(ce_weight * ce + kl_rest_mean)
                else:
                    sample_losses.append(ce_weight * ce)
            n_fill += 1

    if (n_corr + n_fill) == 0:
        # 兜底，返回 0 标量保持图存在
        zero = student_logits.sum() * 0.0
        return (
            zero,
            zero,
            {
                "loss": 0.0,
                "n_corr": 0,
                "n_fill": 0,
                "ce": 0.0,
                "kl_corr": 0.0,
                "kl_fill": 0.0,
            },
        )

    if not use_ema:
        # legacy:把样本平均 loss 塞进 kl_sum 字段返回(语义复用),ce_sum 置 0
        legacy_loss = torch.stack(sample_losses).mean()
        ce_sum = student_logits.sum() * 0.0
        kl_sum = legacy_loss

    metrics = {
        "n_corr": n_corr,
        "n_fill": n_fill,
        "ce": sum_ce / max(n_ce_tok, 1),
        "kl_corr": sum_kl_corr / max(n_corr, 1),
        "kl_fill": sum_kl_fill / max(n_fill, 1),
        "ce_sum_raw": ce_sum.detach().item(),
        "kl_sum_raw": kl_sum.detach().item(),
    }
    return ce_sum, kl_sum, metrics


# =====================================================
# 训练主入口
# =====================================================
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(
            log_path
        ):
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


def train_a_token_sdcl(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_prompt_length: int = 1024,
    max_answer_length: int = 4096,
    use_lora: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    gradient_checkpointing: bool = True,
    log_interval: int = 10,
    save_total_limit: int = 5,
    save_steps: int = 0,
    ce_weight: float = 1.0,
    use_ema: bool = True,
    lambda_ce: float = 0.5,
    lambda_kl: float = 0.5,
    ema_decay: float = 0.99,
    seed: int = 42,
    device_ids: Optional[List[int]] = None,
):
    """混合蒸馏训练主入口（DDP 数据并行）。

    多卡策略：
      - **优先走 DDP**：通过 `torchrun --nproc_per_node=N` 启动时，每个 rank 装一份完整的
        student + teacher 到自己的 cuda:LOCAL_RANK，学生用 DDP 包裹做梯度同步。
      - **未通过 torchrun 启动则退化为单卡**：student / teacher 都放 device_ids[0]
        （或 cuda:0），与改造前的"单 GPU 跑"行为一致。
      - 旧的 `device_ids` 参数仅用于单卡选卡；DDP 模式下由 `LOCAL_RANK` 接管，本参数被忽略。

    启动示例（3 卡 DDP）：
        CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 \\
            scripts/train/a_token_sdcl_train.py --model_path ... --data_path ... --output_dir ...

    Args:
        use_ema    : True(默认)走 EMA 归一化路径,用 lambda_ce / lambda_kl 加权 token-sum 后的 ce/kl;
                     False 走 legacy 路径,每条样本算 ce_weight*CE + KL.mean(),样本间取平均。
        lambda_ce  : EMA 模式下 CE 分项权重(默认 0.5)。use_ema=False 时此值被忽略。
        lambda_kl  : EMA 模式下 KL 分项权重(默认 0.5)。use_ema=False 时此值被忽略。
        ema_decay  : ce_sum / kl_sum 量级估计的 EMA 衰减系数,默认 0.99。
        ce_weight  : Legacy 模式下 fill_correct 首 token CE 项的权重(默认 1.0)。
                     EMA 模式下此值被忽略。
        device_ids : 仅单卡模式生效。DDP 下被忽略。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("混合蒸馏训练需要 CUDA 设备。")
    torch.manual_seed(seed)

    # ── DDP 初始化（torchrun 启动时生效，否则退化单卡） ─────────────────────────
    rank, local_rank, world_size, is_ddp = _ddp_setup()
    is_main = _is_main_rank(rank)

    if is_ddp:
        student_device = f"cuda:{local_rank}"
        teacher_device = f"cuda:{local_rank}"  # DDP 下教师也在本 rank 自己的卡上
        if is_main:
            logger.info(
                "DDP 模式：world_size=%d，每个 rank 各装一份 student+teacher",
                world_size,
            )
    else:
        if device_ids is None:
            device_ids = [0]
        if not device_ids:
            raise ValueError("device_ids 不能为空。")
        student_device = f"cuda:{device_ids[0]}"
        teacher_device = student_device  # 单卡模式：同卡
        logger.info("单卡模式：student=teacher=%s", student_device)

    # 输出目录与日志：rank0 负责创建/写入
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
            raise ValueError("tokenizer 缺 pad_token_id 且 eos_token_id 不可解析。")
        tokenizer.pad_token_id = eos_int

    raw_data = _load_train_data(data_path)
    encoded: List[Dict] = []
    for s in raw_data:
        e = _encode_sample(tokenizer, s, max_prompt_length, max_answer_length)
        if e is not None:
            encoded.append(e)
    if is_main:
        logger.info("编码后训练样本数（全量，未按 rank 切分）：%d", len(encoded))
    if not encoded:
        raise RuntimeError("训练数据为空，无法训练。")

    dtype = torch.bfloat16

    # 教师：每个 rank 各加载一份完整模型到本 rank 卡上（DDP 标准做法）。
    # 单卡模式下 teacher_device == student_device。
    teacher = _build_teacher(
        model_path,
        teacher_device,
        dtype,
        device_map=None,
        max_memory=None,
    )

    student = _build_student(
        model_path,
        student_device,
        dtype,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )
    if is_ddp:
        student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            # LoRA 路径里所有可训练参数都参与反传，没有 unused
            find_unused_parameters=False,
            # grad_ckpt + DDP 兼容
            gradient_as_bucket_view=True,
        )

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    use_amp = True
    amp_dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    pad_id = tokenizer.pad_token_id
    global_step = 0

    # 数据按 rank 切分：先补齐到 world_size 整除，再 encoded[rank::world_size]。
    # 不补齐会导致最后一个 batch 各 rank 数据量差 1 步，DDP all-reduce 等不到对端 → 训练在
    # 末尾 step 死锁(表现为 "Epoch X: 402/403" 卡住,跑了 2h+ 不动)。
    if is_ddp:
        n_full = len(encoded)
        if n_full % world_size != 0:
            pad_n = world_size - (n_full % world_size)
            # 用前 pad_n 条复制补齐(这些样本会在本 epoch 多见一次,
            # 数量 ≤ world_size-1 条,对 7k-100k 数据集影响 < 0.1%)
            encoded_balanced = encoded + encoded[:pad_n]
        else:
            encoded_balanced = encoded
        local_encoded = encoded_balanced[rank::world_size]
        if is_main:
            logger.info(
                "rank=%d/%d 分到 %d 条样本（全量 %d，补齐到 %d）",
                rank, world_size, len(local_encoded), n_full, len(encoded_balanced),
            )
    else:
        local_encoded = encoded

    # 简单 shuffle 索引（每个 rank 用同 seed + rank 偏移，保证差异）
    rng = torch.Generator().manual_seed(seed + rank)

    # ----- LR scheduler: warmup + cosine decay to 0 -----
    # total_optimizer_steps = epoch 数 × 每 epoch 的 optimizer.step 调用数
    # 每 epoch optimizer.step 数 = ceil(n_batches_per_rank / gradient_accumulation_steps),
    # 但代码逻辑里"最后一个 batch 强制 do_step"会让最后那次 step 提前触发,
    # 总数据上限按 ceil(n_batches / accum) 计算足够准。
    n_batches_per_epoch = (len(local_encoded) + batch_size - 1) // batch_size
    opt_steps_per_epoch = max(
        1, (n_batches_per_epoch + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    )
    total_opt_steps = max(1, opt_steps_per_epoch * num_epochs)
    warmup_opt_steps = min(200, max(1, total_opt_steps // 10))
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_opt_steps,
        num_training_steps=total_opt_steps,
    )
    if is_main:
        logger.info(
            "LR schedule: warmup=%d / total=%d optimizer steps (lr_max=%g → 0)",
            warmup_opt_steps, total_opt_steps, learning_rate,
        )

    # ----- EMA 归一化 buffer -----
    # 用历史 ce_sum / kl_sum 的 EMA 估计各分项的"自然量级",每步把 ce_sum/ce_ema 与
    # kl_sum/kl_ema 归一化到 ~1 的同量级,再用 lambda_ce / lambda_kl 加权。
    # 这样不需要手调 ce_weight 这个魔法数字,两类样本贡献的梯度量级自适应平衡。
    # 初始 1.0 是占位,前几个 step EMA 还没收敛时归一化误差不大。
    ce_ema = 1.0
    kl_ema = 1.0
    if is_main:
        if use_ema:
            logger.info(
                "Loss mode: EMA(decay=%g), lambda_ce=%g, lambda_kl=%g",
                ema_decay, lambda_ce, lambda_kl,
            )
        else:
            logger.info(
                "Loss mode: LEGACY (per-sample ce_weight*CE + KL.mean()), ce_weight=%g",
                ce_weight,
            )

    for epoch in range(1, num_epochs + 1):
        if is_main:
            logger.info("--- Epoch %d/%d ---", epoch, num_epochs)
        student.train()
        order = torch.randperm(len(local_encoded), generator=rng).tolist()

        ep_loss = 0.0
        ep_steps = 0
        ep_n_corr = 0
        ep_n_fill = 0
        win_loss: List[float] = []
        win_ce: List[float] = []
        win_kl_corr: List[float] = []
        win_kl_fill: List[float] = []
        win_ce_raw: List[float] = []
        win_kl_raw: List[float] = []
        win_n_corr = 0
        win_n_fill = 0

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

            # 梯度累积期间禁用 DDP all-reduce，最后一步再同步：
            # 大幅降低通信开销（标准 DDP + grad accumulation 写法）
            if is_ddp and not do_step:
                sync_ctx = student.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()

            with sync_ctx:
                ce_sum, kl_sum, metrics = _compute_batch_loss(
                    student=student,
                    teacher=teacher,
                    encoded_batch=batch,
                    pad_token_id=pad_id,
                    student_device=student_device,
                    teacher_device=teacher_device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    ce_weight=ce_weight,
                    use_ema=use_ema,
                )
                if use_ema:
                    # EMA 归一化:用 ce_ema / kl_ema 把两个分项拉到 ~1 量级,再用 lambda 加权。
                    # 关键:除以 EMA 时 EMA 是常量(从历史 .item() 拿的纯 python float),
                    # 不会污染计算图 → 反传只通过 ce_sum / kl_sum 走。
                    ce_norm = ce_sum / max(ce_ema, 1e-8)
                    kl_norm = kl_sum / max(kl_ema, 1e-8)
                    loss = lambda_ce * ce_norm + lambda_kl * kl_norm
                else:
                    # legacy:_compute_batch_loss 已经把样本平均 loss 塞进 kl_sum 字段,
                    # 直接当 loss 用,ce_sum 是 0 不参与。
                    loss = kl_sum
                (loss / gradient_accumulation_steps).backward()

            # EMA 模式:用本 step 的 detach 值更新 EMA。Legacy 模式跳过。
            ce_raw = float(metrics.get("ce_sum_raw", 0.0))
            kl_raw = float(metrics.get("kl_sum_raw", 0.0))
            if use_ema:
                if ce_raw > 0:
                    ce_ema = ema_decay * ce_ema + (1.0 - ema_decay) * ce_raw
                if kl_raw > 0:
                    kl_ema = ema_decay * kl_ema + (1.0 - ema_decay) * kl_raw

            metrics["loss"] = loss.detach().item()

            global_step += 1
            if do_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # step 级 ckpt:用 optimizer step 计数(不是 batch 计数)的简化判断,
                # 用 global_step 也够近似(差 < gradient_accumulation_steps 步)。
                if (
                    is_main
                    and save_steps > 0
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
            ep_n_fill += metrics["n_fill"]
            win_loss.append(metrics["loss"])
            win_ce_raw.append(ce_raw)
            win_kl_raw.append(kl_raw)
            if metrics["n_fill"] > 0:
                win_ce.append(metrics["ce"])
                win_kl_fill.append(metrics["kl_fill"])
                win_n_fill += metrics["n_fill"]
            if metrics["n_corr"] > 0:
                win_kl_corr.append(metrics["kl_corr"])
                win_n_corr += metrics["n_corr"]

            if is_main and (global_step % log_interval == 0):
                avg_loss = sum(win_loss) / max(len(win_loss), 1)
                avg_ce = sum(win_ce) / max(len(win_ce), 1) if win_ce else 0.0
                avg_kl_corr = (
                    sum(win_kl_corr) / max(len(win_kl_corr), 1) if win_kl_corr else 0.0
                )
                avg_kl_fill = (
                    sum(win_kl_fill) / max(len(win_kl_fill), 1) if win_kl_fill else 0.0
                )
                avg_ce_raw = sum(win_ce_raw) / max(len(win_ce_raw), 1)
                avg_kl_raw = sum(win_kl_raw) / max(len(win_kl_raw), 1)
                rec = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "lr": scheduler.get_last_lr()[0],
                    "avg_loss": avg_loss,
                    "avg_ce": avg_ce,
                    "avg_kl_corr": avg_kl_corr,
                    "avg_kl_fill": avg_kl_fill,
                    "avg_ce_sum_raw": avg_ce_raw,
                    "avg_kl_sum_raw": avg_kl_raw,
                    "ce_ema": ce_ema,
                    "kl_ema": kl_ema,
                    "lambda_ce": lambda_ce,
                    "lambda_kl": lambda_kl,
                    "n_corr": win_n_corr,
                    "n_fill": win_n_fill,
                }
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info(
                    "[Step %d] epoch=%d lr=%.2e loss=%.6f ce=%.4f kl_corr=%.4f kl_fill=%.4f "
                    "| ce_sum_ema=%.3f kl_sum_ema=%.3f (λ_ce=%.2f λ_kl=%.2f) "
                    "n_corr=%d n_fill=%d",
                    global_step,
                    epoch,
                    scheduler.get_last_lr()[0],
                    avg_loss,
                    avg_ce,
                    avg_kl_corr,
                    avg_kl_fill,
                    ce_ema,
                    kl_ema,
                    lambda_ce,
                    lambda_kl,
                    win_n_corr,
                    win_n_fill,
                )
                win_loss.clear()
                win_ce.clear()
                win_kl_corr.clear()
                win_kl_fill.clear()
                win_ce_raw.clear()
                win_kl_raw.clear()
                win_n_corr = 0
                win_n_fill = 0

        ep_avg_loss = ep_loss / max(ep_steps, 1)
        if is_main:
            ep_record = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "avg_loss": ep_avg_loss,
                "n_corr": ep_n_corr,
                "n_fill": ep_n_fill,
                "steps": ep_steps,
            }
            with open(epoch_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(ep_record) + "\n")
            logger.info("=" * 60)
            logger.info("*** EPOCH %d/%d FINISHED ***", epoch, num_epochs)
            logger.info(
                "  avg_loss=%.6f n_corr=%d n_fill=%d (rank0)",
                ep_avg_loss, ep_n_corr, ep_n_fill,
            )
            logger.info("=" * 60)

        if is_ddp:
            dist.barrier()  # 等所有 rank 跑完本 epoch 再考虑保存

        if is_main and save_total_limit > 0:
            save_interval = max(1, num_epochs // save_total_limit)
            if epoch % save_interval == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                # DDP: 实际模型在 .module 上；非 DDP 时 student 本身就是模型
                model_to_save = student.module if is_ddp else student
                model_to_save.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info("Checkpoint saved → %s", ckpt_dir)

    if is_main:
        model_to_save = student.module if is_ddp else student
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("训练完成，已保存到 %s", output_dir)

    if is_ddp:
        dist.barrier()
        _ddp_cleanup()



# =====================================================
# CLI
# =====================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="方法3：混合蒸馏训练（按 source 字段分支）"
    )
    parser.add_argument("--model_path", type=str, required=True, help="初始模型路径（同时作为教师）")
    parser.add_argument("--data_path", type=str, required=True, help="a_token_train_data.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_answer_length", type=int, default=4096)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--save_steps", type=int, default=0,
        help="每 N 个 optimizer step 存一次中间 ckpt;0=只按 epoch 末存(默认)。",
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Legacy 模式(--no_ema)下 fill_correct 首 token CE 权重,默认 1.0。EMA 模式忽略。",
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-use_ema(或 --no_ema 别名)走 legacy 路径(per-sample ce_weight*CE + KL.mean()),"
             "默认开启 EMA 归一化路径。",
    )
    # 兼容更直观的 --no_ema 写法(等价于 --no-use_ema)
    parser.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false",
        help="同 --no-use_ema,关闭 EMA,走 legacy 路径。",
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=0.5,
        help="EMA 归一化后 CE 分项权重(默认 0.5,与 --lambda_kl 之和建议 = 1)。",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.5,
        help="EMA 归一化后 KL 分项权重(默认 0.5)。",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="ce_sum / kl_sum 量级估计的 EMA 衰减系数,默认 0.99(约 100 step 半衰)。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU id（如 '0,1,2,3'）；不传则用全部可见 GPU。"
             "第 0 个给学生，其余给教师；3 卡及以上时教师走 device_map='balanced'。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    device_ids: Optional[List[int]] = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip() != ""]
    train_a_token_sdcl(
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
        ce_weight=args.ce_weight,
        use_ema=args.use_ema,
        lambda_ce=args.lambda_ce,
        lambda_kl=args.lambda_kl,
        ema_decay=args.ema_decay,
        seed=args.seed,
        device_ids=device_ids,
    )


if __name__ == "__main__":
    main()

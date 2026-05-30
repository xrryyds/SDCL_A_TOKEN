"""方法3：混合蒸馏训练（按 source 字段分支）。

训练数据：a_token_train_data.json（由 scripts/train/a_token_sdcl.py merge 产出）
  - source == "corr_answer"  : 序列  prompt + answer，每个 answer 位置用 KL(学生 || 教师)
  - source == "fill_correct" : 序列  prompt + fill_token + 续写
        · 首 token 位置：KL(学生 || q')，q' = (1-β)·teacher + β·onehot(fill_token_id)
                          (β = beta_fill,默认 0.5,把 fill 硬标签软化进 teacher 分布,
                           等价于 β=1 时退化成 CE,β=0 时退化成纯 teacher KD)
        · 后续 token 位置：KL(学生 || 教师)
    这样整段 loss 都是 KL,无需 lambda_ce / lambda_kl 加权。

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
    build_first_token_target_logprobs,
    normalize_question_text,
)
from utils.data_utils import extract_boxed_content, normalize_answer  # noqa: E402


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
        if src not in ("corr_answer", "fill_correct", "grpo"):
            logger.warning(
                "跳过未知 source 的样本 idx=%d source=%r", i, src
            )
            continue
        if src == "grpo":
            if (
                not item.get("question")
                or not item.get("anchor_answer")
                or item.get("anchor_first_token_id") is None
            ):
                logger.warning(
                    "跳过缺字段的 grpo 样本 idx=%d (need question/anchor_answer/anchor_first_token_id)",
                    i,
                )
                continue
        else:
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

    # GRPO source：只编码 prompt，answer 由 on-policy rollout 在线产出
    if sample.get("source") == "grpo":
        return {
            "input_ids": list(prompt_ids),
            "prompt_len": len(prompt_ids),
            "answer_len": 0,
            "source": "grpo",
            "fill_token_id": None,
            "fill_pos_in_seq": None,
            "anchor_first_token_id": int(sample["anchor_first_token_id"]),
            "ref_answer": sample.get("ref_answer", ""),
            "question": sample["question"],
        }

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
    beta_fill: float = 0.5,
    fill_probe_records: Optional[List[Dict]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """对一个混合 batch 计算 loss 的分项 sum。

    返回:
      ce_sum   : 本 batch 内所有 fill_correct 样本首 token CE 的总和(标量 tensor,带梯度)
                 ⚠ beta_fill >= 0 时(默认),不再计算 CE,ce_sum 恒为 0;首 token 走"软化 teacher"KL。
      kl_sum   : 本 batch 内所有 token-level KL 的总和(标量 tensor,带梯度)
                 包括 corr_answer 全 answer span 的 KL,fill_correct 首 token 的软化 KL,
                 以及 fill_correct 后续 token 的 KL
                 ⚠ use_ema=False 时,kl_sum 改为存"样本平均的 (ce_weight*CE + KL_mean) legacy
                    loss",ce_sum 强制为 0,调用方直接用 kl_sum 当 loss(语义不对名字也是这个,
                    保留旧签名免改外层结构)。
      metrics  : 日志/统计用,per-token 平均后的指标(纯标量,不带梯度)

    Loss 构成(每条样本独立算):
      默认模式 (use_ema=True, beta_fill>=0, 推荐):
        - corr_answer  : token 级 KL(student || teacher) 在 answer span 上做 sum,累入 kl_sum
        - fill_correct : 首 token KL(student || q'),q' = (1-β)·teacher + β·onehot(fill_token_id),
                         累入 kl_sum;后续 token 的 KL 也累入 kl_sum
        外层用 EMA 把 kl_sum 拉到 ~1 量级直接当 loss(lambda_ce 设 0 / lambda_kl 设 1 即可)。

      旧 CE 模式 (use_ema=True, beta_fill<0):
        - corr_answer  : 同上
        - fill_correct : 首 token 处 CE,累入 ce_sum;后续 token KL 累入 kl_sum
        外层做 EMA 归一化后用 lambda_ce / lambda_kl 加权。⚠ 已 deprecated,仅作回退入口。

      Legacy 模式 (use_ema=False):
        - corr_answer  : KL.mean()  (per-token 平均,与最早老版本一致)
        - fill_correct : ce_weight * CE + KL_rest.mean()
        每条样本独立算 loss,样本间取平均 → 直接放进 kl_sum 返回。
        ⚠ 此模式下 kl_sum.mean() ≈ 老版本 ce_weight ≈ 2 时的 loss 量级 ~15。

    beta_fill 参数:
        β ∈ [0, 1]: 走方案 C(软化 teacher),0=纯 teacher KD,1=等价 CE,推荐 0.5。
        β < 0    : 走旧 CE 路径(向后兼容)。

    fill_probe_records:
        若不为 None,本函数会把每条 fill 样本的探针信息(fill_token_id、teacher 原概率、
        软化后概率、teacher top-1 token 是哪个、概率多少)append 进去,供外层写日志验证。

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
            fill_token_id = int(sample["fill_token_id"])
            ce_logits = student_logits[i, fill_pos_abs - 1, :].float()    # [V]
            ce_target = torch.tensor(
                fill_token_id, dtype=torch.long, device=student_device
            )

            use_beta_path = (beta_fill is not None) and (beta_fill >= 0.0)

            if use_beta_path:
                # ============ 方案 C：软化 teacher 分布,首 token 也走 KL ============
                # q'_v = (1-β) * q_v + β * 1[v == k]
                # k = fill_token_id, q = teacher 第 1 个 token 位置的 softmax
                t_logits_first = teacher_logits[i, fill_pos_abs - 1, :].float()  # [V]
                with torch.no_grad():
                    t_prob_first = F.softmax(t_logits_first, dim=-1)             # [V]
                    # mix
                    t_prob_mixed = (1.0 - beta_fill) * t_prob_first
                    t_prob_mixed[fill_token_id] = (
                        t_prob_mixed[fill_token_id] + beta_fill
                    )
                    t_logp_mixed = (t_prob_mixed.clamp(min=1e-12)).log()         # [V]
                    # 探针:把 teacher 原始/软化后概率、top-1 信息写出去
                    if fill_probe_records is not None:
                        q_orig_at_k = float(t_prob_first[fill_token_id].item())
                        q_mix_at_k = float(t_prob_mixed[fill_token_id].item())
                        top1_id = int(t_prob_first.argmax().item())
                        top1_prob = float(t_prob_first[top1_id].item())
                        s_prob_at_k = float(
                            F.softmax(ce_logits, dim=-1)[fill_token_id].item()
                        )
                        fill_probe_records.append({
                            "fill_token_id": fill_token_id,
                            "q_teacher_at_k": q_orig_at_k,
                            "q_mixed_at_k": q_mix_at_k,
                            "teacher_top1_id": top1_id,
                            "teacher_top1_prob": top1_prob,
                            "p_student_at_k": s_prob_at_k,
                            "k_was_top1": (top1_id == fill_token_id),
                            "beta": float(beta_fill),
                        })
                # 首 token 软化 KL: KL(p_s || q')
                s_logp_first = F.log_softmax(ce_logits, dim=-1)                  # [V]
                kl_first = (
                    s_logp_first.exp() * (s_logp_first - t_logp_mixed)
                ).sum().clamp(min=0.0)
                if use_ema:
                    kl_sum = kl_sum + kl_first
                # 也单独记录 ce-equivalent 量(诊断用):-log p_s(k)
                ce_equiv = (-s_logp_first[fill_token_id]).detach()
                sum_ce += float(ce_equiv.item())
                n_ce_tok += 1
                # 让 metrics["ce"] 仍然有值(报"ce-equivalent",方便对比历史曲线)
                # legacy 模式下 beta 路径不允许:防御性塞个 ce 占位
                ce = ce_equiv  # 仅给 legacy 路径兜底用,实际 use_ema=True 不进 sample_losses
            else:
                # ============ 旧 CE 路径(deprecated,仅向后兼容) ============
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
                # (legacy 路径不支持 beta_fill 软化,beta 路径只在 EMA 下生效)
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
# GRPO 单 batch 计算 loss（首 token 软目标 KL）
# =====================================================
def _compute_grpo_loss(
    student,
    rollout_engine,
    tokenizer: AutoTokenizer,
    grpo_samples: List[Dict],
    pad_token_id: int,
    student_device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    alpha: float = 0.2,
    delta: float = 0.1,
    n_rollout: int = 8,
    T: float = 0.6,
    top_p: float = 0.95,
    max_rollout_tokens: int = 4096,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """对一个 grpo batch 计算首 token 软目标 KL。

    流程:
      1) 用 _build_prompt 构造 chat-template 一致的 prompt 字符串
      2) rollout_engine.rollout(no_grad) 拿 [B, n] 条 rollout
      3) 评分:extract_boxed_content + normalize_answer vs ref_answer → reward 0/1
      4) 全 0 reward 回退:把 anchor_first_token_id 当虚拟第 (n+1) 条 reward=1 的 rollout
      5) Student forward (prompt only,左 padding),取 logits[:, -1, :] = 首答案 token 位
      6) 逐样本调 build_first_token_target_logprobs(s_logits.detach(), first_ids, rewards, α, δ)
      7) kl_i = (s_logp.exp() * (s_logp - target_logp)).sum().clamp(min=0)

    Args:
        grpo_samples: 由 _encode_sample 返回的 dict 列表,每条含
                      input_ids / prompt_len / question / ref_answer / anchor_first_token_id
        rollout_engine: GrpoRolloutEngine 实例(已 update_lora 到当前 step)

    Returns:
        kl_sum   : 标量 tensor,本 batch 所有 grpo 样本的首 token KL 之和(带梯度)
        metrics  : 日志字典 {n_grpo, kl_grpo, frac_explore, frac_anchor_used, avg_pass_rate}
    """
    if not grpo_samples:
        zero = torch.zeros((), device=student_device, dtype=torch.float32, requires_grad=False)
        return zero, {
            "n_grpo": 0,
            "kl_grpo": 0.0,
            "frac_explore": 0.0,
            "frac_anchor_used": 0.0,
            "avg_pass_rate": 0.0,
        }

    # 1) 构造 prompts(给 rollout engine)
    prompts = [_build_prompt(tokenizer, s["question"]) for s in grpo_samples]

    # 2) Rollout (no_grad,内部 vLLM 自管显存)
    with torch.no_grad():
        rollouts = rollout_engine.rollout(
            prompts, n=n_rollout, T=T, top_p=top_p, max_tokens=max_rollout_tokens
        )
    assert len(rollouts) == len(grpo_samples), \
        f"rollout count mismatch: {len(rollouts)} vs {len(grpo_samples)}"

    # 3) 评分 + 4) anchor 回退:per sample 收集 (first_token_ids, rewards)
    per_sample_first_ids: List[List[int]] = []
    per_sample_rewards: List[List[float]] = []
    n_anchor_used = 0
    sum_pass_rate = 0.0

    for i, sample in enumerate(grpo_samples):
        ref_norm = normalize_answer(sample.get("ref_answer", ""))
        first_ids: List[int] = []
        rewards: List[float] = []
        n_correct_local = 0

        for text in rollouts[i]:
            boxed = extract_boxed_content(text)
            ok = bool(boxed) and normalize_answer(boxed) == ref_norm and ref_norm != ""
            r = 1.0 if ok else 0.0
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not ids:
                # 罕见空字符串,跳过本条 rollout
                continue
            first_ids.append(int(ids[0]))
            rewards.append(r)
            if ok:
                n_correct_local += 1

        # 全 0 回退:append anchor 作为虚拟正确 rollout
        if not rewards or all(r < 0.5 for r in rewards):
            first_ids.append(int(sample["anchor_first_token_id"]))
            rewards.append(1.0)
            n_anchor_used += 1

        per_sample_first_ids.append(first_ids)
        per_sample_rewards.append(rewards)
        sum_pass_rate += (n_correct_local / max(len(rollouts[i]), 1))

    # 5) Student forward(只 prompt)
    rows = [s["input_ids"] for s in grpo_samples]
    s_input_ids, s_attn_mask, max_len = _pad_left_batch(rows, pad_token_id, student_device)
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        student_logits = student(
            input_ids=s_input_ids, attention_mask=s_attn_mask
        ).logits  # [B, S, V]

    # 左 padding 后,每行最后一个有效 token 在 max_len-1,首答案 token 位 = logits[:, max_len-1, :]
    first_logits = student_logits[:, max_len - 1, :]  # [B, V]

    # 6-7) 逐样本算 KL
    kl_sum = student_logits.sum() * 0.0  # 零标量,带计算图
    n_grpo = 0
    n_explore = 0
    sum_kl = 0.0
    for i in range(len(grpo_samples)):
        s_logits_i = first_logits[i:i + 1]  # [1, V]
        target_logp, is_explore = build_first_token_target_logprobs(
            s_logits_i.detach(),
            per_sample_first_ids[i],
            per_sample_rewards[i],
            alpha=alpha,
            delta=delta,
        )
        s_logp = F.log_softmax(s_logits_i.float(), dim=-1)  # [1, V]
        kl_i = (s_logp.exp() * (s_logp - target_logp)).sum().clamp(min=0.0)
        kl_sum = kl_sum + kl_i
        sum_kl += kl_i.detach().item()
        n_grpo += 1
        if is_explore:
            n_explore += 1

    metrics = {
        "n_grpo": n_grpo,
        "kl_grpo": sum_kl / max(n_grpo, 1),
        "frac_explore": n_explore / max(n_grpo, 1),
        "frac_anchor_used": n_anchor_used / max(n_grpo, 1),
        "avg_pass_rate": sum_pass_rate / max(n_grpo, 1),
    }
    return kl_sum, metrics


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


# =====================================================
# GRPO LoRA snapshot helper
# =====================================================
def _snapshot_student_lora(student, output_dir: str, step: int) -> str:
    """rank-0 把当前 student LoRA save 到 output_dir/_trainee_lora_{step}/。
    后面 GrpoRolloutEngine.update_lora 从该目录加载。"""
    snap_dir = os.path.join(output_dir, f"_trainee_lora_{step}")
    os.makedirs(snap_dir, exist_ok=True)
    inner = student.module if hasattr(student, "module") else student
    inner.save_pretrained(snap_dir)
    return snap_dir


def train_a_token_sdcl(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
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
    ce_weight: float = 1.0,
    use_ema: bool = True,
    lambda_ce: float = 0.0,
    lambda_kl: float = 1.0,
    ema_decay: float = 0.99,
    beta_fill: float = 0.5,
    seed: int = 42,
    device_ids: Optional[List[int]] = None,
    # ───── GRPO 三池 (opt-in,默认全关) ─────
    enable_grpo: bool = False,
    w_grpo: float = 1.0,
    grpo_alpha: float = 0.2,
    grpo_delta: float = 0.1,
    grpo_n: int = 8,
    grpo_temperature: float = 0.6,
    grpo_top_p: float = 0.95,
    grpo_max_tokens: int = 4096,
    grpo_lora_sync_every: int = 4,
    grpo_max_model_len: int = 6144,
    grpo_gpu_mem_util: float = 0.22,
    grpo_max_lora_rank: int = 64,
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
        lambda_ce  : EMA 模式下 CE 分项权重(默认 0.0)。⚠ 启用 beta_fill>=0 时整段 loss 都是 KL,
                     CE 项不存在,此参数被忽略;保留只为向后兼容(beta_fill<0 的旧 CE 路径)。
        lambda_kl  : EMA 模式下 KL 分项权重(默认 1.0)。
        ema_decay  : ce_sum / kl_sum 量级估计的 EMA 衰减系数,默认 0.99。
        ce_weight  : Legacy 模式下 fill_correct 首 token CE 项的权重(默认 1.0)。
                     EMA 模式下此值被忽略。
        beta_fill  : ∈[0,1] 走方案 C(软化 teacher),首 token 也变 KL,默认 0.5。
                     β=0 退化成纯 teacher KD,β=1 退化成 CE。
                     <0 走旧 CE 路径(deprecated,仅保留回退入口)。
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
        # 方案 C 探针日志:每个 step 把 fill 样本的 fill_token / teacher 原概率 / 软化后概率
        # 等信息聚合后写一行,用来事后验证算法确实生效。
        beta_log_file = os.path.join(output_dir, "beta_fill_log.jsonl")
    else:
        step_log_file, epoch_log_file = None, None
        beta_log_file = None
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

    # ───── GRPO rollout engine 初始化 (与 student 同卡 colocate) ─────
    rollout_engine = None
    if enable_grpo:
        from scripts.train.grpo_rollout_engine import GrpoRolloutEngine

        # 每个 rank 一个引擎,占用 cuda:local_rank
        engine_device_id = local_rank if is_ddp else (device_ids[0] if device_ids else 0)
        if is_main:
            logger.info(
                "GRPO 启用:rollout_engine on cuda:%d  max_model_len=%d  gpu_mem_util=%.3f  "
                "lora_sync_every=%d  n=%d T=%g top_p=%g  α=%g δ=%g  w_grpo=%g",
                engine_device_id, grpo_max_model_len, grpo_gpu_mem_util,
                grpo_lora_sync_every, grpo_n, grpo_temperature, grpo_top_p,
                grpo_alpha, grpo_delta, w_grpo,
            )
        rollout_engine = GrpoRolloutEngine(
            model_path=model_path,
            device_id=engine_device_id,
            max_model_len=grpo_max_model_len,
            max_lora_rank=grpo_max_lora_rank,
            gpu_memory_utilization=grpo_gpu_mem_util,
            max_loras=2,
        )
        # 初次 LoRA 同步:rank-0 snapshot → barrier → 所有 rank update_lora
        if is_main:
            _snapshot_student_lora(student, output_dir, step=0)
        if is_ddp:
            dist.barrier()
        snap0 = os.path.join(output_dir, "_trainee_lora_0")
        rollout_engine.update_lora(snap0, step=0)
        if is_main:
            logger.info("GRPO 初次 LoRA 同步完成 (step=0 → %s)", snap0)

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
            if beta_fill is not None and beta_fill >= 0.0:
                logger.info(
                    "Loss mode: EMA + 方案C(soft-teacher fill), beta_fill=%g, "
                    "lambda_ce=%g, lambda_kl=%g  "
                    "(整段 loss 都是 KL,lambda_ce 实际不参与 fill 分支)",
                    beta_fill, lambda_ce, lambda_kl,
                )
            else:
                logger.info(
                    "Loss mode: EMA + 旧 CE 路径(deprecated), lambda_ce=%g, lambda_kl=%g",
                    lambda_ce, lambda_kl,
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
        win_probe: List[Dict] = []  # 方案C探针:本窗口所有 fill 样本的 mix 信息
        # GRPO 窗口聚合
        win_kl_grpo: List[float] = []
        win_n_grpo = 0
        win_frac_explore: List[float] = []
        win_frac_anchor: List[float] = []
        win_pass_rate: List[float] = []

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
                # 切分 batch:监督路径(corr/fill) vs GRPO 路径
                std_batch = [s for s in batch if s.get("source") != "grpo"]
                grpo_batch = [s for s in batch if s.get("source") == "grpo"]

                # 探针 buffer:方案C 路径下,_compute_batch_loss 会把每条 fill 样本的
                # mix 信息 append 进来。只在主 rank 申请,非主 rank 传 None 跳过开销。
                probe_records: Optional[List[Dict]] = [] if is_main else None
                if std_batch:
                    ce_sum, kl_sum, metrics = _compute_batch_loss(
                        student=student,
                        teacher=teacher,
                        encoded_batch=std_batch,
                        pad_token_id=pad_id,
                        student_device=student_device,
                        teacher_device=teacher_device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        ce_weight=ce_weight,
                        use_ema=use_ema,
                        beta_fill=beta_fill,
                        fill_probe_records=probe_records,
                    )
                else:
                    # 整个 batch 全是 grpo:监督分支零贡献,仍需保留计算图占位
                    zero_t = torch.zeros((), device=student_device, dtype=torch.float32)
                    ce_sum, kl_sum = zero_t, zero_t
                    metrics = {
                        "n_corr": 0, "n_fill": 0, "ce": 0.0,
                        "kl_corr": 0.0, "kl_fill": 0.0,
                        "ce_sum_raw": 0.0, "kl_sum_raw": 0.0,
                    }
                if use_ema:
                    # EMA 归一化:用 ce_ema / kl_ema 把两个分项拉到 ~1 量级,再用 lambda 加权。
                    # 关键:除以 EMA 时 EMA 是常量(从历史 .item() 拿的纯 python float),
                    # 不会污染计算图 → 反传只通过 ce_sum / kl_sum 走。
                    ce_norm = ce_sum / max(ce_ema, 1e-8)
                    kl_norm = kl_sum / max(kl_ema, 1e-8)
                    loss_std = lambda_ce * ce_norm + lambda_kl * kl_norm
                else:
                    # legacy:_compute_batch_loss 已经把样本平均 loss 塞进 kl_sum 字段,
                    # 直接当 loss 用,ce_sum 是 0 不参与。
                    loss_std = kl_sum

                # ───── GRPO 分支 ─────
                grpo_metrics = {
                    "n_grpo": 0, "kl_grpo": 0.0,
                    "frac_explore": 0.0, "frac_anchor_used": 0.0, "avg_pass_rate": 0.0,
                }
                if enable_grpo and grpo_batch:
                    kl_grpo_sum, grpo_metrics = _compute_grpo_loss(
                        student=student,
                        rollout_engine=rollout_engine,
                        tokenizer=tokenizer,
                        grpo_samples=grpo_batch,
                        pad_token_id=pad_id,
                        student_device=student_device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        alpha=grpo_alpha,
                        delta=grpo_delta,
                        n_rollout=grpo_n,
                        T=grpo_temperature,
                        top_p=grpo_top_p,
                        max_rollout_tokens=grpo_max_tokens,
                    )
                    loss_grpo = kl_grpo_sum / max(grpo_metrics["n_grpo"], 1)
                    loss = loss_std + w_grpo * loss_grpo
                else:
                    loss = loss_std

                (loss / gradient_accumulation_steps).backward()
            # merge grpo metrics into metrics for downstream logging
            metrics.update(grpo_metrics)

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

                # ───── GRPO 每 K 个 optimizer step 同步一次 LoRA ─────
                if (
                    enable_grpo
                    and rollout_engine is not None
                    and grpo_lora_sync_every > 0
                    and global_step % grpo_lora_sync_every == 0
                ):
                    if is_main:
                        _snapshot_student_lora(student, output_dir, step=global_step)
                    if is_ddp:
                        dist.barrier()
                    snap_dir = os.path.join(output_dir, f"_trainee_lora_{global_step}")
                    rollout_engine.update_lora(snap_dir, step=global_step)
                    if is_main and (global_step % (grpo_lora_sync_every * 5) == 0):
                        # rank-0 GC 旧 snapshot,只保留最近 3 个
                        try:
                            existing = sorted(
                                [d for d in os.listdir(output_dir)
                                 if d.startswith("_trainee_lora_")],
                                key=lambda d: int(d.split("_")[-1]),
                            )
                            for old in existing[:-3]:
                                old_path = os.path.join(output_dir, old)
                                if os.path.isdir(old_path):
                                    import shutil
                                    shutil.rmtree(old_path, ignore_errors=True)
                        except Exception as e:
                            logger.warning("LoRA snapshot GC failed: %s", e)

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
            if probe_records:
                win_probe.extend(probe_records)
            if metrics["n_fill"] > 0:
                win_ce.append(metrics["ce"])
                win_kl_fill.append(metrics["kl_fill"])
                win_n_fill += metrics["n_fill"]
            if metrics["n_corr"] > 0:
                win_kl_corr.append(metrics["kl_corr"])
                win_n_corr += metrics["n_corr"]
            # GRPO accumulation
            if metrics.get("n_grpo", 0) > 0:
                win_kl_grpo.append(metrics["kl_grpo"])
                win_n_grpo += metrics["n_grpo"]
                win_frac_explore.append(metrics["frac_explore"])
                win_frac_anchor.append(metrics["frac_anchor_used"])
                win_pass_rate.append(metrics["avg_pass_rate"])

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
                    "beta_fill": beta_fill,
                    "n_corr": win_n_corr,
                    "n_fill": win_n_fill,
                    # GRPO
                    "n_grpo": win_n_grpo,
                    "kl_grpo": (sum(win_kl_grpo) / max(len(win_kl_grpo), 1)) if win_kl_grpo else 0.0,
                    "frac_explore": (sum(win_frac_explore) / max(len(win_frac_explore), 1)) if win_frac_explore else 0.0,
                    "frac_anchor_used": (sum(win_frac_anchor) / max(len(win_frac_anchor), 1)) if win_frac_anchor else 0.0,
                    "avg_pass_rate": (sum(win_pass_rate) / max(len(win_pass_rate), 1)) if win_pass_rate else 0.0,
                    "w_grpo": w_grpo if enable_grpo else 0.0,
                }
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                # ====== 方案 C 探针日志:聚合窗口内 fill 样本的 mix 信息 ======
                if win_probe:
                    n_probe = len(win_probe)
                    avg_q_orig = sum(p["q_teacher_at_k"] for p in win_probe) / n_probe
                    avg_q_mixed = sum(p["q_mixed_at_k"] for p in win_probe) / n_probe
                    avg_top1_prob = (
                        sum(p["teacher_top1_prob"] for p in win_probe) / n_probe
                    )
                    avg_p_student = sum(p["p_student_at_k"] for p in win_probe) / n_probe
                    n_k_is_top1 = sum(1 for p in win_probe if p["k_was_top1"])
                    # 取前 3 条样例完整记录(看具体 token_id / 概率)
                    sample_probes = win_probe[:3]
                    probe_rec = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "timestamp": datetime.now().isoformat(),
                        "beta_fill": beta_fill,
                        "n_fill_samples": n_probe,
                        "avg_q_teacher_at_k": avg_q_orig,
                        "avg_q_mixed_at_k": avg_q_mixed,
                        "avg_teacher_top1_prob": avg_top1_prob,
                        "avg_p_student_at_k": avg_p_student,
                        "n_k_is_teacher_top1": n_k_is_top1,
                        "frac_k_is_teacher_top1": n_k_is_top1 / n_probe,
                        "samples": sample_probes,
                    }
                    with open(beta_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(probe_rec, ensure_ascii=False) + "\n")
                    logger.info(
                        "[β-probe step %d] β=%.2f n=%d  q_orig@k=%.4f → q_mix@k=%.4f  "
                        "teacher_top1=%.4f  p_student@k=%.4f  k_is_top1=%d/%d (%.1f%%)",
                        global_step, beta_fill, n_probe,
                        avg_q_orig, avg_q_mixed, avg_top1_prob, avg_p_student,
                        n_k_is_top1, n_probe, 100.0 * n_k_is_top1 / n_probe,
                    )

                logger.info(
                    "[Step %d] epoch=%d lr=%.2e loss=%.6f ce=%.4f kl_corr=%.4f kl_fill=%.4f "
                    "| ce_sum_ema=%.3f kl_sum_ema=%.3f (λ_ce=%.2f λ_kl=%.2f β=%.2f) "
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
                    beta_fill if beta_fill is not None else -1.0,
                    win_n_corr,
                    win_n_fill,
                )
                if enable_grpo and win_n_grpo > 0:
                    logger.info(
                        "[GRPO step %d] n_grpo=%d kl_grpo=%.4f frac_explore=%.2f "
                        "frac_anchor_used=%.2f avg_pass_rate=%.2f (w_grpo=%g α=%g δ=%g)",
                        global_step,
                        win_n_grpo,
                        rec["kl_grpo"],
                        rec["frac_explore"],
                        rec["frac_anchor_used"],
                        rec["avg_pass_rate"],
                        w_grpo, grpo_alpha, grpo_delta,
                    )
                win_loss.clear()
                win_ce.clear()
                win_kl_corr.clear()
                win_kl_fill.clear()
                win_ce_raw.clear()
                win_kl_raw.clear()
                win_probe.clear()
                win_n_corr = 0
                win_n_fill = 0
                # GRPO window clear
                win_kl_grpo.clear()
                win_frac_explore.clear()
                win_frac_anchor.clear()
                win_pass_rate.clear()
                win_n_grpo = 0

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

    if rollout_engine is not None:
        try:
            rollout_engine.shutdown()
        except Exception as e:
            logger.warning("rollout_engine.shutdown() failed: %s", e)

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
    parser.add_argument("--max_prompt_length", type=int, default=2048)
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
        default=0.0,
        help="EMA 归一化后 CE 分项权重(默认 0.0)。⚠ 启用 beta_fill>=0(默认)时整段 loss 都是 KL,"
             "CE 项不存在,此参数被忽略;只在 --beta_fill -1(旧 CE 路径)下生效。",
    )
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=1.0,
        help="EMA 归一化后 KL 分项权重(默认 1.0)。",
    )
    parser.add_argument(
        "--beta_fill",
        type=float,
        default=0.5,
        help="方案 C:fill 首 token 软化 teacher 分布的强度,q'=(1-β)q+β·onehot(k)。"
             "β∈[0,1] 推荐 0.5;β=0 退化为纯 teacher KD,β=1 退化为 CE;"
             "β<0 走旧 CE 路径(deprecated,仅向后兼容)。",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="ce_sum / kl_sum 量级估计的 EMA 衰减系数,默认 0.99(约 100 step 半衰)。",
    )
    parser.add_argument("--seed", type=int, default=42)
    # ───── GRPO 三池 (默认全关) ─────
    parser.add_argument(
        "--enable_grpo", action="store_true",
        help="启用 GRPO 三池路径 (与 corr/fill 加性叠加,opt-in)。",
    )
    parser.add_argument("--w_grpo", type=float, default=1.0, help="GRPO loss 权重 (默认 1.0)")
    parser.add_argument("--grpo_alpha", type=float, default=0.2, help="正确 token 提升幅度 α")
    parser.add_argument("--grpo_delta", type=float, default=0.1, help="错误 token 压制幅度 δ")
    parser.add_argument("--grpo_n", type=int, default=8, help="每题 rollout 数")
    parser.add_argument("--grpo_temperature", type=float, default=0.6)
    parser.add_argument("--grpo_top_p", type=float, default=0.95)
    parser.add_argument("--grpo_max_tokens", type=int, default=4096)
    parser.add_argument(
        "--grpo_lora_sync_every", type=int, default=4,
        help="每 N 个 optimizer step 把 student LoRA 热同步到 vLLM (默认 4)",
    )
    parser.add_argument("--grpo_max_model_len", type=int, default=6144)
    parser.add_argument("--grpo_gpu_mem_util", type=float, default=0.22)
    parser.add_argument("--grpo_max_lora_rank", type=int, default=64)
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
        beta_fill=args.beta_fill,
        seed=args.seed,
        device_ids=device_ids,
        enable_grpo=args.enable_grpo,
        w_grpo=args.w_grpo,
        grpo_alpha=args.grpo_alpha,
        grpo_delta=args.grpo_delta,
        grpo_n=args.grpo_n,
        grpo_temperature=args.grpo_temperature,
        grpo_top_p=args.grpo_top_p,
        grpo_max_tokens=args.grpo_max_tokens,
        grpo_lora_sync_every=args.grpo_lora_sync_every,
        grpo_max_model_len=args.grpo_max_model_len,
        grpo_gpu_mem_util=args.grpo_gpu_mem_util,
        grpo_max_lora_rank=args.grpo_max_lora_rank,
    )


if __name__ == "__main__":
    main()

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
import torch.nn.functional as F
from torch.amp import autocast
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
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """对一个混合 batch 计算 loss。

    Loss 构成（每条样本独立算，最后取 mean）：
      - corr_answer  : token 级 KL(student || teacher) 在 answer span 上做 mean
      - fill_correct : 在 fill_pos 处 CE(student, fill_token_id) * ce_weight
                       + 在 fill_pos+1..end 处 token 级 KL(student || teacher) 做 mean
                       两项相加（与文档"loss = loss_first + loss_rest"一致）

    多卡：student / teacher 可能在不同 device 上（默认 cuda:0 / cuda:1）。
    输入张量分别放到各自 device 前向，再把 teacher logits 搬回 student device 做 KL。
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

    # 逐样本算 loss
    sample_losses: List[torch.Tensor] = []
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
            # KL(student || teacher) = sum_v p_s * (log p_s - log p_t)，按 token 平均
            kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).mean().clamp(min=0.0)
            sample_losses.append(kl)
            n_corr += 1
            sum_kl_corr += kl.detach().item()
            n_kl_corr_tok += s_logits.size(0)

        else:  # fill_correct
            # fill token 在序列中的位置（绝对）
            # 注意：encode 时 fill_pos_in_seq = prompt_len（未左填充时）
            fill_pos_abs = start + sample["fill_pos_in_seq"]   # = start + prompt_len
            # 学生预测 fill token 用 logits[i, fill_pos_abs - 1] = logits[i, pred_lo]
            ce_logits = student_logits[i, fill_pos_abs - 1, :].float()    # [V]
            ce_target = torch.tensor(
                sample["fill_token_id"], dtype=torch.long, device=student_device
            )
            ce = F.cross_entropy(ce_logits.unsqueeze(0), ce_target.unsqueeze(0))
            sum_ce += ce.detach().item()
            n_ce_tok += 1

            # 后续 token KL：从 answer 第 2 个 token 开始（即 logits[i, pred_lo+1 .. pred_hi]）
            rest_lo = pred_lo + 1
            rest_hi = pred_hi
            if rest_hi >= rest_lo:
                s_logits = student_logits[i, rest_lo : rest_hi + 1, :].float()
                t_logits = teacher_logits[i, rest_lo : rest_hi + 1, :].float()
                s_logp = F.log_softmax(s_logits, dim=-1)
                t_logp = F.log_softmax(t_logits, dim=-1)
                # KL(student || teacher)
                kl_rest = (
                    (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1).mean().clamp(min=0.0)
                )
                sum_kl_fill += kl_rest.detach().item()
                n_kl_fill_tok += s_logits.size(0)
                sample_losses.append(ce_weight * ce + kl_rest)
            else:
                # answer 长度只有 1（即只有 fill token），没有 rest
                sample_losses.append(ce_weight * ce)
            n_fill += 1

    if not sample_losses:
        # 兜底，返回 0 标量保持图存在
        return (
            student_logits.sum() * 0.0,
            {
                "loss": 0.0,
                "n_corr": 0,
                "n_fill": 0,
                "ce": 0.0,
                "kl_corr": 0.0,
                "kl_fill": 0.0,
            },
        )

    loss = torch.stack(sample_losses).mean()
    metrics = {
        "loss": loss.detach().item(),
        "n_corr": n_corr,
        "n_fill": n_fill,
        "ce": sum_ce / max(n_ce_tok, 1),
        "kl_corr": sum_kl_corr / max(n_corr, 1),
        "kl_fill": sum_kl_fill / max(n_fill, 1),
    }
    return loss, metrics


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
    max_answer_length: int = 2048,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    gradient_checkpointing: bool = True,
    log_interval: int = 10,
    save_total_limit: int = 5,
    ce_weight: float = 1.0,
    seed: int = 42,
    device_ids: Optional[List[int]] = None,
):
    """混合蒸馏训练主入口（支持多卡：学生 / 教师分卡）。

    多卡策略（参考 scripts/train/a_token_sd copy.py 的 4 卡模式风格）：
      - device_ids=None（默认）：自动检测所有可见 GPU
        · 单卡            : 学生 + 教师都放 cuda:0（显存吃紧时建议关 LoRA + 开 grad ckpt）
        · 2 卡            : 学生→cuda:0，教师→cuda:1
        · 3 卡及以上      : 学生→cuda:0，教师走 device_map="balanced" 把权重均匀切到剩余卡上，
                            最大化教师并行度，给学生留出 cuda:0 的全部显存
      - device_ids=[a,b,...]：第 0 个给学生，其余分配给教师；只给一个 id 时退化为同卡

    Args:
        ce_weight  : fill_correct 样本中首 token CE 项的权重（默认 1.0，与文档一致）。
                     若想加重对易错首 token 的纠正可调到 2.0~5.0。
        device_ids : 例如 [0,1] 或 [0,1,2,3]；None 时用全部可见 GPU。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("混合蒸馏训练需要 CUDA 设备。")
    torch.manual_seed(seed)

    # ── 多卡分配 ─────────────────────────────────────────────────────────────
    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n))
    if not device_ids:
        raise ValueError("device_ids 不能为空。")

    student_device = f"cuda:{device_ids[0]}"
    teacher_device_ids = device_ids[1:] if len(device_ids) > 1 else [device_ids[0]]

    # 教师放置策略
    teacher_device_map = None
    teacher_device = f"cuda:{teacher_device_ids[0]}"
    if len(device_ids) >= 3:
        # 教师 device_map：把教师权重均匀分布到 device_ids[1:] 上（多卡 pipeline）
        teacher_device_map = {"": teacher_device_ids[0]}
        # 用 accelerate 的 "balanced_low_0" 风格：让 transformers 自己分布到 teacher 卡上
        # 这里简单起见走 device_map="auto"，accelerate 会扫描可见 cuda 卡
        # 为避免占用 student 卡，先临时设置 CUDA_VISIBLE_DEVICES_FOR_TEACHER
        teacher_device_map = "balanced"   # accelerate 自动均衡
    logger.info(
        "多卡分配：student=%s，teacher=%s%s",
        student_device,
        teacher_device,
        f"（device_map={teacher_device_map}）" if teacher_device_map else "",
    )

    os.makedirs(output_dir, exist_ok=True)
    step_log_file, epoch_log_file = setup_logging(output_dir)

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
    logger.info("编码后训练样本数：%d", len(encoded))
    if not encoded:
        raise RuntimeError("训练数据为空，无法训练。")

    dtype = torch.bfloat16
    # 构造 max_memory：
    #  - 学生卡设为 0：防止 device_map='balanced' 把教师权重切到学生卡
    #  - 不在 device_ids 中的可见卡也设为 0：防止权重外溢到用户未指定的卡
    #  - 教师卡按 "可用显存 - 4GiB" 上报，留 buffer 给 forward 激活
    teacher_max_memory = None
    if teacher_device_map == "balanced":
        teacher_max_memory = {}
        student_id = device_ids[0]
        teacher_id_set = set(device_ids[1:])
        n_visible = torch.cuda.device_count()
        for d in range(n_visible):
            if d == student_id or d not in teacher_id_set:
                teacher_max_memory[d] = "0GiB"
            else:
                free_bytes, _total = torch.cuda.mem_get_info(d)
                avail_gib = max(1, int(free_bytes / (1024**3)) - 4)
                teacher_max_memory[d] = f"{avail_gib}GiB"
        logger.info(
            "教师 max_memory=%s（学生卡 + 未指定卡=0，防止权重外溢）",
            teacher_max_memory,
        )

    teacher = _build_teacher(
        model_path,
        teacher_device,
        dtype,
        device_map=teacher_device_map,
        max_memory=teacher_max_memory,
    )
    # 当教师走 device_map 多卡分布时，embedding 实际所在卡未必是 teacher_device。
    # 把输入送到 embedding 所在卡，accelerate 的 hooks 会负责后续层间搬运。
    if teacher_device_map is not None and hasattr(teacher, "hf_device_map"):
        embed_dev = None
        for key, dev in teacher.hf_device_map.items():
            if "embed_tokens" in key or key.endswith("embed"):
                embed_dev = dev
                break
        if embed_dev is None:
            # 退化：取 device_map 中第一个 cuda 设备
            for dev in teacher.hf_device_map.values():
                if isinstance(dev, int) or (isinstance(dev, str) and dev.startswith("cuda")):
                    embed_dev = dev
                    break
        if embed_dev is not None:
            teacher_device = (
                f"cuda:{embed_dev}" if isinstance(embed_dev, int) else str(embed_dev)
            )
            logger.info("教师 embedding 所在 device=%s（输入将送往这里）", teacher_device)

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

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=learning_rate
    )
    use_amp = True
    amp_dtype = torch.bfloat16
    torch.set_float32_matmul_precision("high")

    pad_id = tokenizer.pad_token_id
    global_step = 0

    # 简单 shuffle 索引
    rng = torch.Generator().manual_seed(seed)

    for epoch in range(1, num_epochs + 1):
        logger.info("--- Epoch %d/%d ---", epoch, num_epochs)
        student.train()
        # 每 epoch 重新 shuffle
        order = torch.randperm(len(encoded), generator=rng).tolist()

        ep_loss = 0.0
        ep_steps = 0
        ep_n_corr = 0
        ep_n_fill = 0
        win_loss: List[float] = []
        win_ce: List[float] = []
        win_kl_corr: List[float] = []
        win_kl_fill: List[float] = []
        win_n_corr = 0
        win_n_fill = 0

        n_batches = (len(encoded) + batch_size - 1) // batch_size
        progress = tqdm(range(n_batches), desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for bi in progress:
            ids = order[bi * batch_size : (bi + 1) * batch_size]
            batch = [encoded[j] for j in ids]

            loss, metrics = _compute_batch_loss(
                student=student,
                teacher=teacher,
                encoded_batch=batch,
                pad_token_id=pad_id,
                student_device=student_device,
                teacher_device=teacher_device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                ce_weight=ce_weight,
            )

            (loss / gradient_accumulation_steps).backward()
            global_step += 1
            is_last = bi == n_batches - 1
            if global_step % gradient_accumulation_steps == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad()

            ep_loss += metrics["loss"]
            ep_steps += 1
            ep_n_corr += metrics["n_corr"]
            ep_n_fill += metrics["n_fill"]
            win_loss.append(metrics["loss"])
            if metrics["n_fill"] > 0:
                win_ce.append(metrics["ce"])
                win_kl_fill.append(metrics["kl_fill"])
                win_n_fill += metrics["n_fill"]
            if metrics["n_corr"] > 0:
                win_kl_corr.append(metrics["kl_corr"])
                win_n_corr += metrics["n_corr"]

            if global_step % log_interval == 0:
                avg_loss = sum(win_loss) / max(len(win_loss), 1)
                avg_ce = sum(win_ce) / max(len(win_ce), 1) if win_ce else 0.0
                avg_kl_corr = (
                    sum(win_kl_corr) / max(len(win_kl_corr), 1) if win_kl_corr else 0.0
                )
                avg_kl_fill = (
                    sum(win_kl_fill) / max(len(win_kl_fill), 1) if win_kl_fill else 0.0
                )
                rec = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "avg_loss": avg_loss,
                    "avg_ce": avg_ce,
                    "avg_kl_corr": avg_kl_corr,
                    "avg_kl_fill": avg_kl_fill,
                    "n_corr": win_n_corr,
                    "n_fill": win_n_fill,
                }
                with open(step_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info(
                    "[Step %d] epoch=%d loss=%.6f ce=%.6f kl_corr=%.6f kl_fill=%.6f "
                    "n_corr=%d n_fill=%d",
                    global_step,
                    epoch,
                    avg_loss,
                    avg_ce,
                    avg_kl_corr,
                    avg_kl_fill,
                    win_n_corr,
                    win_n_fill,
                )
                win_loss.clear()
                win_ce.clear()
                win_kl_corr.clear()
                win_kl_fill.clear()
                win_n_corr = 0
                win_n_fill = 0

        ep_avg_loss = ep_loss / max(ep_steps, 1)
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
            "  avg_loss=%.6f n_corr=%d n_fill=%d", ep_avg_loss, ep_n_corr, ep_n_fill
        )
        logger.info("=" * 60)

        if save_total_limit > 0:
            save_interval = max(1, num_epochs // save_total_limit)
            if epoch % save_interval == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info("Checkpoint saved → %s", ckpt_dir)

    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("训练完成，已保存到 %s", output_dir)


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
    parser.add_argument("--max_answer_length", type=int, default=2048)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="fill_correct 首 token CE 权重，>1 加重纠错强度",
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
        ce_weight=args.ce_weight,
        seed=args.seed,
        device_ids=device_ids,
    )


if __name__ == "__main__":
    main()

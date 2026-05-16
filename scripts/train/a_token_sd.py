import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Sequence

# 必须在 import torch / vLLM 之前设置，PyTorch 在首次 CUDA 分配时读取此变量。
# expandable_segments:True 消除 reserved-but-unallocated 碎片导致的 OOM。
# 新版 PyTorch (2.x) 使用 PYTORCH_ALLOC_CONF，旧版使用 PYTORCH_CUDA_ALLOC_CONF，两个都设。
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

# 尝试导入 vLLM，如果失败则记录错误
try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as exc:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = exc

class _TqdmLoggingHandler(logging.Handler):
    """将日志记录路由到 tqdm.write()，防止进度条被覆盖。"""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm as _tqdm
            _tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

# 基础日志配置（仅控制台）；文件 handler 在 setup_logging() 中按 output_dir 动态添加
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_TqdmLoggingHandler()],
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """在 output_dir 下创建结构化日志文件，参考 student_train_v3.py 的日志模式。

    返回:
        step_log_file  : step_metrics.jsonl  路径（每 log_interval 步写一行）
        epoch_log_file : epoch_metrics.jsonl 路径（每 epoch 结束写一行）

    注意：每次调用前先移除同名 FileHandler，防止多次调用（如 API 模式）时
    日志被重复写入同一文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    # 移除已存在的同路径 FileHandler，避免重复
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
    step_log_file = os.path.join(output_dir, "step_metrics.jsonl")
    epoch_log_file = os.path.join(output_dir, "epoch_metrics.jsonl")
    return step_log_file, epoch_log_file


class StepMetricsTracker:
    """累积每步指标，支持窗口（step）和 epoch 两个粒度的统计。

    字段说明：
        loss      : 经 gradient_accumulation_steps 归一化后的 loss（已 .item()）
        kl        : 每道题的 first-token KL 散度均值
        n_correct : 当前 rollout batch 中首 token 引导后答对的题数
        n_total   : 当前 rollout batch 的题数
    """

    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def reset_window(self):
        self._win_loss: list = []
        self._win_kl: list = []
        self._win_correct: int = 0
        self._win_total: int = 0
        self._win_steps: int = 0

    def reset_epoch(self):
        self._ep_loss: list = []
        self._ep_kl: list = []
        self._ep_correct: int = 0
        self._ep_total: int = 0
        self._ep_steps: int = 0

    def update(self, loss: float, kl_values: list, n_correct: int, n_total: int):
        self._win_loss.append(loss)
        self._win_kl.extend(kl_values)
        self._win_correct += n_correct
        self._win_total += n_total
        self._win_steps += 1

        self._ep_loss.append(loss)
        self._ep_kl.extend(kl_values)
        self._ep_correct += n_correct
        self._ep_total += n_total
        self._ep_steps += 1

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def get_window_stats(self) -> dict:
        return {
            "avg_loss": self._avg(self._win_loss),
            "avg_kl": self._avg(self._win_kl),
            "correct_rate": self._win_correct / max(self._win_total, 1),
            "n_correct": self._win_correct,
            "n_total": self._win_total,
            "steps": self._win_steps,
        }

    def get_epoch_stats(self) -> dict:
        return {
            "avg_loss": self._avg(self._ep_loss),
            "avg_kl": self._avg(self._ep_kl),
            "correct_rate": self._ep_correct / max(self._ep_total, 1),
            "n_correct": self._ep_correct,
            "n_total": self._ep_total,
            "steps": self._ep_steps,
        }

SYSTEM_PROMPT = r"Please reason step by step and put your final answer within \boxed{}."

def _stringify_text(value) -> str:
    """将输入转换为字符串。"""
    if value is None: return ""
    if isinstance(value, str): return value
    return str(value)

def normalize_question_text(value) -> str:
    """规范化问题文本。"""
    return _stringify_text(value).strip()

def extract_answer(text: str) -> str:
    r"""从文本中提取 \boxed{} 中的答案。"""
    text = _stringify_text(text).strip()
    if r"\boxed{" in text:
        start = text.rfind(r"\boxed{") + len(r"\boxed{")
        end = text.find("}", start)
        if end != -1: return text[start:end].strip()
    return text.strip()

def normalize_reference_answer(value) -> str:
    """规范化参考答案。"""
    return extract_answer(value)

def check_correctness(pred: str, ref: str) -> bool:
    """检查预测答案是否正确。"""
    return extract_answer(pred) == extract_answer(ref)

def _is_vllm_available() -> bool:
    """检查 vLLM 是否可用。"""
    return LLM is not None and SamplingParams is not None and torch.cuda.is_available()

def _infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    """自动推断 LoRA 目标模块。"""
    common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    target_modules = [module_name for module_name in common_targets if module_name in available]
    if not target_modules:
        raise ValueError("无法推断 LoRA 目标模块。")
    return target_modules

def _save_merged_lora_for_vllm(student_model: "PeftModel", tokenizer: "AutoTokenizer", tmp_dir: str) -> None:
    """将 LoRA 权重合并到基础模型并保存到临时目录，不破坏原 PeftModel。

    注意：merge_and_unload() 会就地卸载 PeftModel 的 adapter 并返回底层 base_model，
    若直接对 student_model 调用，后续 epoch 的训练将退化为全参数微调（LoRA 丢失）。

    解决方案：在 CPU 上对 student_model 做 deepcopy，对副本调用 merge_and_unload()
    并保存，原 student_model 的 adapter 完全不受影响。
    调用方须在调用前已将 student_model 移至 CPU（student_model.cpu()），
    以避免 deepcopy 时 GPU 显存翻倍。
    """
    logger.info("正在合并 LoRA 权重以供 vLLM 使用（保留原 PeftModel adapter）...")
    import copy as _copy
    # deepcopy 在 CPU 上进行，约占 ~14GB 系统内存（7B bf16），H200 服务器可接受
    cpu_copy = _copy.deepcopy(student_model)
    merged = cpu_copy.merge_and_unload()
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    del merged, cpu_copy
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _build_models(model_path: str, torch_dtype: torch.dtype, device: str, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    """构建学生模型。"""
    student_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    if use_lora:
        target_modules = _infer_lora_target_modules(student_model)
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, task_type="CAUSAL_LM", bias="none")
        student_model = get_peft_model(student_model, peft_config).to(device)
    student_model.config.use_cache = False
    # gradient checkpointing：backward 时重新计算激活，显存减少 ~40%，速度慢 ~20%
    # 对 batch_size=16 的 7B 模型是必要的，否则 backward 时 OOM
    student_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return student_model

def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """构建聊天模板提示词。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": normalize_question_text(question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def tokenize_prompt(tokenizer: AutoTokenizer, text: str, device: str, max_prompt_length: int) -> torch.Tensor:
    """对提示词进行分词。"""
    return tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prompt_length).input_ids.to(device)

def pad_left_batch(rows: Sequence[torch.Tensor], pad_token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """将一组 1-D token 序列左填充成 batch。"""
    max_len = max(row.size(0) for row in rows)
    batch_ids = torch.full((len(rows), max_len), pad_token_id, dtype=rows[0].dtype, device=device)
    attn_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        row_len = row.size(0)
        batch_ids[i, -row_len:] = row
        attn_mask[i, -row_len:] = 1
    return batch_ids, attn_mask

def batch_prompt_ids(tokenizer: AutoTokenizer, prompt_texts: List[str], device: str, max_prompt_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """对多条 prompt 一次性构造左填充 batch。
    
    返回值第三项由原来的 List[int]（各行原始长度）改为 int（batch 的 max_len），
    用于在左填充 batch 中正确定位最后一个有效 token：
      左填充时有效 token 在右侧，最后一个有效 token 固定在 max_len-1 位置，
      而非 row_lens[i]-1（那是错误的，会指向 pad 区域）。
    """
    rows = [tokenize_prompt(tokenizer, text, device, max_prompt_length).squeeze(0) for text in prompt_texts]
    input_ids, attn_mask = pad_left_batch(rows, tokenizer.pad_token_id, device)
    max_len = input_ids.size(1)
    return input_ids, attn_mask, max_len

def vllm_generate(
    model_path: str,
    tokenizer: "AutoTokenizer",
    prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> List[List[str]]:
    """使用 vLLM 对每条 prompt 生成 n 条完整序列。

    返回 List[List[str]]，外层长度 = len(prompts)，内层长度 = n。
    单次调用创建 LLM 实例，用完后立即 del 释放显存。
    """
    if not _is_vllm_available():
        raise RuntimeError(f"vLLM 不可用: {_VLLM_IMPORT_ERROR}")
    sampling = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
    )
    outputs = llm.generate(prompts, sampling)
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [[o.text for o in req.outputs] for req in outputs]


def vllm_eval_and_rollout(
    model_path: str,
    tokenizer: "AutoTokenizer",
    all_prompts: List[str],
    all_answers: List[str],
    n_roll: int,
    max_new_tokens: int,
    rollout_temperature: float,
    gpu_memory_utilization: float,
    max_model_len: int,
):
    """在同一个 vLLM 实例内完成 eval + rollout，节省一次 LoRA 合并/加载开销。

    流程：
      1. 对全部 prompts 做 greedy eval（n=1, temperature=0）→ 识别错题
      2. 对错题 prompts 做 temperature rollout（n=n_roll）
      3. del llm，释放显存

    返回：
        base_predictions : List[str]，长度 = len(all_prompts)，greedy 预测文本
        mistake_indices  : List[int]，错题在 all_prompts 中的下标
        all_rollouts     : List[List[str]]，长度 = len(mistake_indices)，每题 n_roll 条序列
    """
    if not _is_vllm_available():
        raise RuntimeError(f"vLLM 不可用: {_VLLM_IMPORT_ERROR}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
    )
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    # ── Phase A: greedy eval ──────────────────────────────────────────────────
    eval_sampling = SamplingParams(
        n=1, temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=eos_ids
    )
    eval_outputs = llm.generate(all_prompts, eval_sampling)
    base_predictions = [req.outputs[0].text for req in eval_outputs]

    # 识别错题下标
    mistake_indices = [
        i for i, (pred, ref) in enumerate(zip(base_predictions, all_answers))
        if not check_correctness(pred, ref)
    ]

    all_rollouts: List[List[str]] = []
    if mistake_indices:
        # ── Phase B: temperature rollout（仅错题）────────────────────────────
        mistake_prompts = [all_prompts[i] for i in mistake_indices]
        rollout_sampling = SamplingParams(
            n=n_roll,
            temperature=rollout_temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=eos_ids,
        )
        rollout_outputs = llm.generate(mistake_prompts, rollout_sampling)
        all_rollouts = [[o.text for o in req.outputs] for req in rollout_outputs]

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_predictions, mistake_indices, all_rollouts


def build_first_token_target_logprobs(
    student_first_logits: torch.Tensor,
    first_token_ids: List[int],
    rewards: List[float],
    alpha: float,
    delta: float,
) -> torch.Tensor:
    """根据组内 rollout 的二值奖励构建首 token KL 目标分布（log-prob）。

    算法：
      - 对每条 rollout 序列，取其首 token id 和对应的二值奖励（1=正确，0=错误）。
      - 正确序列的首 token：概率向 p_max 靠拢（奖励）。
      - 错误序列的首 token：概率乘以 (1 - delta)（惩罚）。
      - 重新归一化后取 log，作为 KL 散度的目标 log-prob。

    Args:
        student_first_logits: shape [1, vocab_size]，学生模型在首 token 位置的 logits（float32）。
        first_token_ids:      每条 rollout 序列的首 token id，长度 = n_roll。
        rewards:              每条 rollout 的二值奖励（1.0=正确，0.0=错误），长度 = n_roll。
        alpha:                正确首 token 的奖励幅度（向 p_max 插值的比例）。
        delta:                错误首 token 的惩罚幅度（概率乘以 1-delta）。

    Returns:
        log_target: shape [1, vocab_size]，目标分布的 log-prob（float32）。
    """
    with torch.no_grad():
        probs = F.softmax(student_first_logits.float(), dim=-1).clone()  # [1, V]
    p_max = probs.max().item()
    for tid, r in zip(first_token_ids, rewards):
        if r > 0.5:  # 正确：奖励
            p_cur = probs[0, tid].item()
            probs[0, tid] = p_cur + (p_max - p_cur) * alpha
        else:  # 错误：惩罚
            probs[0, tid] = probs[0, tid] * (1.0 - delta)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.log(probs.clamp(min=1e-12))

def train_a_token_sd(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    n_roll: int = 8,
    alpha: float = 0.1,
    delta: float = 0.1,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 4096,
    vllm_gpu_memory_utilization: float = 0.85,
    rollout_temperature: float = 0.8,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    gradient_accumulation_steps: int = 4,
    rollout_batch_size: int = 8,
    log_interval: int = 10,
    save_total_limit: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """GRPO-style A-Token 自蒸馏训练。

    每个 epoch 流程：
      1. vLLM 评估全部训练题，收集错题集。
      2. 对每道错题，vLLM 以温度采样 roll n 条完整序列。
      3. 对每条 rollout 序列计算二值奖励（答对=1，答错=0）。
      4. 用组内奖励修正学生首 token 分布，构造 KL 目标分布。
      5. 仅在首 token 位置计算 KL 损失（后续 token 目标=学生自身→KL=0）。
      6. torch.compile + bf16 autocast + 梯度累积，最大化算力利用率。

    Args:
        rollout_temperature: vLLM rollout 采样温度（>0 保证多样性）。
        log_interval:        每隔多少个 rollout-batch 步写一次 step_metrics.jsonl。
    """
    # CPU 路径无法运行 vLLM（vLLM 强依赖 CUDA），提前报错而非静默失败
    if device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "train_a_token_sd 需要 CUDA 设备（vLLM 不支持 CPU）。"
            f" 当前 device={device!r}, cuda_available={torch.cuda.is_available()}"
        )
    os.makedirs(output_dir, exist_ok=True)
    step_log_file, epoch_log_file = setup_logging(output_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    student_model = _build_models(model_path, torch_dtype, device, use_lora, lora_r, lora_alpha, lora_dropout)

    # TF32：在 Ampere+ 上启用高精度矩阵乘，消除 UserWarning
    torch.set_float32_matmul_precision("high")

    # torch.compile 与 gradient_checkpointing 组合存在已知 bug：
    #   attention backward 的 attn_bias stride 对齐错误（strideH 不是 4 的倍数），
    #   导致 "attn_bias is not correctly aligned" RuntimeError。
    # 禁用 compile，使用 eager 模式，与 gradient checkpointing 完全兼容。
    compiled_model = student_model
    _compile_ok = False
    logger.info("使用 eager 模式（torch.compile 与 gradient_checkpointing 不兼容，已禁用）。")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    # GradScaler 仅在 CUDA 下有效；CPU 路径禁用 AMP
    use_amp = device != "cpu" and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    metrics_tracker = StepMetricsTracker()

    questions = [normalize_question_text(item.get("question", item.get("prompt", ""))) for item in raw_data]
    answers = [normalize_reference_answer(item.get("answer", item.get("ref_answer", ""))) for item in raw_data]
    max_model_len = max_prompt_length + max_new_tokens
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # ── Step 1+2: 合并 eval + rollout，共用一个 vLLM 实例 ─────────────────
        # 节省一次 LoRA 合并/保存/加载（约 2~3 分钟/epoch）
        all_prompts_for_vllm = [build_prompt(tokenizer, q) for q in questions]
        _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_vllm_")
        try:
            student_model.cpu()
            torch.cuda.empty_cache()
            if use_lora:
                _save_merged_lora_for_vllm(student_model, tokenizer, _vllm_tmp)
            else:
                student_model.save_pretrained(_vllm_tmp)
                tokenizer.save_pretrained(_vllm_tmp)
            base_predictions, mistake_indices, all_rollouts = vllm_eval_and_rollout(
                _vllm_tmp, tokenizer,
                all_prompts=all_prompts_for_vllm,
                all_answers=answers,
                n_roll=n_roll,
                max_new_tokens=max_new_tokens,
                rollout_temperature=rollout_temperature,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=max_model_len,
            )
        finally:
            shutil.rmtree(_vllm_tmp, ignore_errors=True)
            student_model.to(device)
            # device 迁移后重置 dynamo 缓存，避免 torch.compile 因 device 变化
            # 触发隐式重编译或产生不稳定行为
            if _compile_ok:
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        n_correct_phase1 = sum(1 for p, a in zip(base_predictions, answers) if check_correctness(p, a))
        mistake_count = len(mistake_indices)
        logger.info(
            f"Epoch {epoch} Eval: total={len(questions)}, correct={n_correct_phase1}, "
            f"mistakes={mistake_count}, acc={n_correct_phase1/max(len(questions),1):.4f}"
        )
        if mistake_count == 0:
            logger.info(f"Epoch {epoch}: no mistakes, skipping training.")
            continue

        # 从 mistake_indices 重建错题的 answers 和 prompts（rollout 已在 vllm_eval_and_rollout 内完成）
        mistake_answers = [normalize_reference_answer(raw_data[i].get("answer", raw_data[i].get("ref_answer", ""))) for i in mistake_indices]
        mistake_prompts = [all_prompts_for_vllm[i] for i in mistake_indices]

        # ── Step 3: 计算每条 rollout 的二值奖励 ──────────────────────────────
        # all_rollouts[i][j] = j-th rollout text for mistake i
        all_rewards: List[List[float]] = []
        for ref_ans, rollout_seqs in zip(mistake_answers, all_rollouts):
            all_rewards.append([
                1.0 if check_correctness(seq, ref_ans) else 0.0
                for seq in rollout_seqs
            ])

        # ── Step 4: 训练 — 首 token KL loss ──────────────────────────────────
        # 对每道错题：
        #   a. 从每条 rollout 序列中提取首 token id（rollout 文本的第一个 token）
        #   b. 用组内二值奖励修正学生首 token 分布 → 目标分布
        #   c. KL(student_logprob[first_token] || target) 作为损失
        #   后续 token 目标 = 学生自身分布 → KL = 0，不参与损失
        student_model.train()
        metrics_tracker.reset_epoch()
        optimizer.zero_grad()

        mistake_progress = tqdm(
            range(0, mistake_count, rollout_batch_size),
            desc=f"Epoch {epoch} Train",
        )
        for i in mistake_progress:
            batch_slice = slice(i, i + rollout_batch_size)
            batch_prompts   = mistake_prompts[batch_slice]
            batch_refs      = mistake_answers[batch_slice]
            batch_rollouts  = all_rollouts[batch_slice]   # List[List[str]]
            batch_rewards   = all_rewards[batch_slice]    # List[List[float]]

            # 提取每条 rollout 序列的首 token id
            # 正确做法：将 prompt + completion 拼接后编码，取 prompt 末尾之后的第一个 token，
            # 而非单独编码 completion（BPE 子词拼接会导致首 token 不同）。
            batch_first_token_ids: List[List[int]] = []
            for prompt_text, rollout_seqs in zip(batch_prompts, batch_rollouts):
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_len = len(prompt_ids)
                first_ids = []
                for seq in rollout_seqs:
                    combined_ids = tokenizer(
                        prompt_text + seq, add_special_tokens=False
                    ).input_ids
                    # combined_ids[:prompt_len] 对应 prompt，[prompt_len] 是真实首 token
                    if len(combined_ids) > prompt_len:
                        first_ids.append(combined_ids[prompt_len])
                    else:
                        first_ids.append(tokenizer.eos_token_id)
                batch_first_token_ids.append(first_ids)

            # 训练前向：仅需 prompt 的 logits（首 token 位置）
            input_ids, attn_mask, prompt_max_len = batch_prompt_ids(
                tokenizer, batch_prompts, device, max_prompt_length
            )
            # 左填充：最后一个有效 token 固定在 max_len-1
            last_idx = prompt_max_len - 1

            with autocast(enabled=use_amp):
                logits_train = compiled_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits  # [B, seq, V]
                # 首 token logits（float32 for numerical stability）
                batch_first_logits = logits_train[:, last_idx, :].float()  # [B, V]
                batch_first_logprobs = F.log_softmax(batch_first_logits, dim=-1)  # [B, V]

            # 在 autocast 外构造目标分布（float32），不参与梯度
            batch_losses: List[torch.Tensor] = []
            batch_kl_vals: List[float] = []
            batch_n_correct = 0
            for j, (first_logits_j, first_logprobs_j, first_tids, rewards_j, ref) in enumerate(
                zip(batch_first_logits, batch_first_logprobs, batch_first_token_ids, batch_rewards, batch_refs)
            ):
                target_logprobs = build_first_token_target_logprobs(
                    first_logits_j.unsqueeze(0).detach(),
                    first_tids,
                    rewards_j,
                    alpha,
                    delta,
                )  # [1, V]
                kl = F.kl_div(
                    first_logprobs_j.unsqueeze(0),
                    target_logprobs,
                    reduction="batchmean",
                    log_target=True,
                )
                batch_losses.append(kl)
                batch_kl_vals.append(kl.detach().item())
                if any(r > 0.5 for r in rewards_j):
                    batch_n_correct += 1

            loss = torch.stack(batch_losses).mean() / gradient_accumulation_steps
            scaler.scale(loss).backward()

            metrics_tracker.update(
                loss=loss.item(),
                kl_values=batch_kl_vals,
                n_correct=batch_n_correct,
                n_total=len(batch_prompts),
            )
            global_step += 1

            # 梯度累积：满足步数或到达末尾时 optimizer.step()
            is_last_batch = (i + rollout_batch_size) >= mistake_count
            if global_step % gradient_accumulation_steps == 0 or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ── 每 log_interval 步写一次 step_metrics.jsonl（参考 StepLogCallback）──
            if global_step % log_interval == 0:
                win_stats = metrics_tracker.get_window_stats()
                step_record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    **win_stats,
                }
                with open(step_log_file, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(step_record) + "\n")
                log_parts = [f"[Step {global_step}]", f"epoch={epoch}"]
                for k, v in win_stats.items():
                    log_parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")
                logger.info(" | ".join(log_parts))
                metrics_tracker.reset_window()

        # ── Epoch 结束：写 epoch_metrics.jsonl（参考 LogicalEpochLogCallback）──
        ep_stats = metrics_tracker.get_epoch_stats()
        epoch_record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "n_mistakes": mistake_count,
            "phase1_acc": n_correct_phase1 / max(len(questions), 1),
            **ep_stats,
        }
        with open(epoch_log_file, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(epoch_record) + "\n")
        logger.info("=" * 60)
        logger.info(f"*** EPOCH {epoch}/{num_epochs} FINISHED ***")
        logger.info(f"  phase1_acc={epoch_record['phase1_acc']:.4f}  mistakes={mistake_count}")
        logger.info(f"  avg_loss={ep_stats['avg_loss']:.6f}  avg_kl={ep_stats['avg_kl']:.6f}")
        logger.info(f"  rollout_correct_rate={ep_stats['correct_rate']:.4f}  ({ep_stats['n_correct']}/{ep_stats['n_total']})")
        logger.info("=" * 60)

        # ── Checkpoint 保存 ──────────────────────────────────────────────────
        # save_total_limit > 0 时：每隔 floor(num_epochs / save_total_limit) 个 epoch 保存一次。
        # 例如 num_epochs=10, save_total_limit=10 → 每 1 个 epoch 保存一次。
        # 例如 num_epochs=10, save_total_limit=5  → 每 2 个 epoch 保存一次。
        if save_total_limit > 0:
            save_interval = max(1, num_epochs // save_total_limit)
            if epoch % save_interval == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved → {ckpt_dir}")

    student_model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")

def train_a_token_sd_api(
    questions,
    answers,
    epoch,
    output_dir=None,
    model_path_override=None,
    use_lora=True,
    learning_rate=5e-5,
    n_roll=8,
    alpha=0.1,
    delta=0.1,
    max_prompt_length=1024,
    max_new_tokens=2048,
    vllm_gpu_memory_utilization=0.85,
    rollout_temperature=0.8,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    gradient_accumulation_steps=4,
    rollout_batch_size=8,
    log_interval=10,
    save_total_limit=0,
    device=None,
):
    """API 包装器。

    Args:
        rollout_temperature: vLLM rollout 采样温度。
        log_interval:        每隔多少个 rollout-batch 步写一次 step_metrics.jsonl。
        save_total_limit:    总共保存几个 checkpoint（0=不保存中间 checkpoint）。
                             保存间隔 = floor(epoch / save_total_limit)。
                             例如 epoch=10, save_total_limit=10 → 每 1 epoch 保存一次。
    """
    if not model_path_override:
        raise ValueError(
            "train_a_token_sd_api: 必须通过 model_path_override 指定模型路径，"
            "当前值为 None 或空字符串。"
        )
    resolved_model_path = model_path_override
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)
    train_samples = [
        {"question": str(q).strip(), "answer": str(a).strip()}
        for q, a in zip(questions, answers)
    ]
    os.makedirs(resolved_output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json",
        prefix="a_token_sd_", dir=resolved_output_dir, delete=False,
    ) as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        temp_data_path = f.name
    try:
        train_a_token_sd(
            model_path=resolved_model_path,
            data_path=temp_data_path,
            output_dir=resolved_output_dir,
            num_epochs=epoch,
            learning_rate=learning_rate,
            n_roll=n_roll,
            alpha=alpha,
            delta=delta,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            rollout_temperature=rollout_temperature,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_accumulation_steps=gradient_accumulation_steps,
            rollout_batch_size=rollout_batch_size,
            log_interval=log_interval,
            save_total_limit=save_total_limit,
            device=resolved_device,
        )
    finally:
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
    return {"output_dir": resolved_output_dir}

def _build_a_token_sd_output_dir(epoch: int) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"a_token_sd_{epoch}ep_{datetime.now().strftime('%m%d_%H%M')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="GRPO-style A-Token 自蒸馏训练（vLLM rollout + 首 token KL loss）"
    )
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据 JSON 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_roll", type=int, default=8, help="每道错题 rollout 条数")
    parser.add_argument("--alpha", type=float, default=0.1, help="正确首 token 奖励幅度")
    parser.add_argument("--delta", type=float, default=0.1, help="错误首 token 惩罚幅度")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--rollout_temperature", type=float, default=0.8, help="vLLM rollout 采样温度")
    parser.add_argument("--rollout_batch_size", type=int, default=16, help="训练 batch size（H200 推荐 16~32）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()
    train_a_token_sd(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        n_roll=args.n_roll,
        alpha=args.alpha,
        delta=args.delta,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        rollout_temperature=args.rollout_temperature,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        rollout_batch_size=args.rollout_batch_size,
        log_interval=args.log_interval,
    )

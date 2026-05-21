"""
GRPO-style A-Token 自蒸馏训练（首 token KL loss）

核心思想（与原始设计一致）：
  - 仅在首 token 位置施加学习信号。
  - 对每道错题，n_roll 条 rollout 各自有二值奖励（答对=1 / 答错=0）。
  - 按 rollout 的首 token id 去重聚合，逐 token 独立奖惩：
      · 该 id 至少有一条 rollout 答对  → 奖励：p ← p + (p_max - p) · α
      · 该 id 全部 rollout 答错        → 压制：p ← p · (1 - δ)
  - 重归一化后取 log，作为 KL 目标分布。
  - 同一题中，奖励和压制可同时发生（针对不同 token id）。
  - 当一组 rollout 全错时（每个出现过的 token 都被压制），归一化会把概率
    自然分给“未尝试过的 token”，从而鼓励模型尝试新方向。
"""

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional, Sequence

# 必须在 import torch / vLLM 之前设置
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

try:
    from vllm import LLM, SamplingParams

    _VLLM_IMPORT_ERROR = None
except Exception as exc:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = exc


# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────
class _TqdmLoggingHandler(logging.Handler):
    """让日志通过 tqdm.write 输出，避免破坏进度条。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm as _tqdm

            _tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_TqdmLoggingHandler()],
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str):
    """在 output_dir 下创建结构化日志文件。"""
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


# ─────────────────────────────────────────────────────────────────────────────
# 指标聚合
# ─────────────────────────────────────────────────────────────────────────────
class StepMetricsTracker:
    """聚合每步的 KL / 正确率统计，支持窗口（log_interval）+ epoch 两个粒度。

    explore_ratio 字段表示：当前窗口/epoch 中，组内 rollout 全错的题目占比，
    即“走纯探索分支”的比例（仅用于诊断，不影响算法）。
    """

    def __init__(self):
        self.reset_window()
        self.reset_epoch()

    def reset_window(self):
        self._win_kl: list = []
        self._win_correct: int = 0
        self._win_total: int = 0
        self._win_explore: int = 0
        self._win_steps: int = 0

    def reset_epoch(self):
        self._ep_kl: list = []
        self._ep_correct: int = 0
        self._ep_total: int = 0
        self._ep_explore: int = 0
        self._ep_steps: int = 0

    def update(
        self, kl_values: list, n_correct: int, n_total: int, n_explore: int = 0
    ):
        self._win_kl.extend(kl_values)
        self._win_correct += n_correct
        self._win_total += n_total
        self._win_explore += n_explore
        self._win_steps += 1
        self._ep_kl.extend(kl_values)
        self._ep_correct += n_correct
        self._ep_total += n_total
        self._ep_explore += n_explore
        self._ep_steps += 1

    @staticmethod
    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def get_window_stats(self) -> dict:
        return {
            "avg_kl": self._avg(self._win_kl),
            "correct_rate": self._win_correct / max(self._win_total, 1),
            "explore_ratio": self._win_explore / max(self._win_total, 1),
            "n_correct": self._win_correct,
            "n_explore": self._win_explore,
            "n_total": self._win_total,
            "steps": self._win_steps,
        }

    def get_epoch_stats(self) -> dict:
        return {
            "avg_kl": self._avg(self._ep_kl),
            "correct_rate": self._ep_correct / max(self._ep_total, 1),
            "explore_ratio": self._ep_explore / max(self._ep_total, 1),
            "n_correct": self._ep_correct,
            "n_explore": self._ep_explore,
            "n_total": self._ep_total,
            "steps": self._ep_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = r"Please reason step by step and put your final answer within \boxed{}."


def _stringify_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_int(value) -> Optional[int]:
    """从 int / list / None 中取第一个 int。

    用途：tokenizer.eos_token_id 在 DeepSeek/Qwen 部分变体中是 list，需要兼容。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, int):
                return v
    return None


def _build_stop_token_ids(tokenizer) -> List[int]:
    """vLLM SamplingParams 的 stop_token_ids。

    强制加入 <|im_end|>（151645），DeepSeek-R1-Distill-Qwen 系列必需，
    否则模型不会停止生成、出现重复循环。
    """
    ids = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        ids.update(int(x) for x in eos if isinstance(x, int))
    elif isinstance(eos, int):
        ids.add(eos)
    pad = tokenizer.pad_token_id
    if isinstance(pad, int):
        ids.add(pad)
    ids.add(151645)  # <|im_end|>
    return list(ids)


def normalize_question_text(value) -> str:
    return _stringify_text(value).strip()


def extract_answer(text: str) -> str:
    r"""从文本中提取最后一个 \boxed{} 的内容（处理嵌套花括号）。"""
    text = _stringify_text(text).strip()
    key = r"\boxed{"
    pos = text.rfind(key)
    if pos != -1:
        start = pos + len(key)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()
    return text.strip()


def normalize_reference_answer(value) -> str:
    return extract_answer(value)


def check_correctness(pred: str, ref: str) -> bool:
    return extract_answer(pred) == extract_answer(ref)


def _is_vllm_available() -> bool:
    return LLM is not None and SamplingParams is not None and torch.cuda.is_available()


def _infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    common_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    target_modules = [m for m in common_targets if m in available]
    if not target_modules:
        raise ValueError("无法推断 LoRA 目标模块。")
    return target_modules


def _soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """软截断：cap * tanh(x / cap)，避免 hard clamp 把 >cap 样本梯度直接清零。"""
    return cap * torch.tanh(x / cap)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA 合并到磁盘（供 vLLM 使用）—— 不再使用 deepcopy(PeftModel)
# ─────────────────────────────────────────────────────────────────────────────
def _save_merged_lora_for_vllm(
    student_model: PeftModel,
    tokenizer: AutoTokenizer,
    tmp_dir: str,
    base_model_path: str,
) -> None:
    """将 LoRA 权重合并到基础模型并保存到 tmp_dir，供 vLLM 加载。

    实现方式：
      1. 仅保存当前 adapter（几十 MB）到 tmp_dir/_adapter_tmp。
      2. 在 CPU 上重新从磁盘加载 base + adapter（独立副本）。
      3. 调用 merge_and_unload() 合并并保存到 tmp_dir。
      4. 删除临时副本，原 student_model 完全不受影响。

    相比 deepcopy(PeftModel)：
      - 不依赖 deepcopy 对 hook / weakref / quant state 的兼容性；
      - 临时模型在 CPU bf16 加载，峰值系统内存 ≈ 14 GB（vs deepcopy 30+ GB）。
    """
    logger.info("合并 LoRA 权重 → 临时目录（磁盘 adapter + 重加载）...")
    adapter_dir = os.path.join(tmp_dir, "_adapter_tmp")
    os.makedirs(adapter_dir, exist_ok=True)

    student_model.save_pretrained(adapter_dir)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # 强制 CPU，避免与后续 vLLM 争抢 GPU 显存导致 OOM
    )
    tmp_peft = PeftModel.from_pretrained(base, adapter_dir)
    merged = tmp_peft.merge_and_unload()
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    del merged, tmp_peft, base
    shutil.rmtree(adapter_dir, ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_models(
    model_path: str,
    torch_dtype: torch.dtype,
    device: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool = True,
):
    student_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    if use_lora:
        target_modules = _infer_lora_target_modules(student_model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
        )
        student_model = get_peft_model(student_model, peft_config).to(device)
    student_model.config.use_cache = False
    if gradient_checkpointing:
        student_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("gradient_checkpointing 已启用。")
    else:
        logger.info("gradient_checkpointing 已禁用。")
    return student_model


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def tokenize_prompt(
    tokenizer: AutoTokenizer, text: str, device: str, max_prompt_length: int
) -> torch.Tensor:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    ).input_ids.to(device)


def pad_left_batch(
    rows: Sequence[torch.Tensor], pad_token_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """左填充：保证最后一个有效 token 在 max_len-1 位置。"""
    max_len = max(row.size(0) for row in rows)
    batch_ids = torch.full(
        (len(rows), max_len), pad_token_id, dtype=rows[0].dtype, device=device
    )
    attn_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        row_len = row.size(0)
        batch_ids[i, -row_len:] = row
        attn_mask[i, -row_len:] = 1
    return batch_ids, attn_mask


def batch_prompt_ids(
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    device: str,
    max_prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = [
        tokenize_prompt(tokenizer, text, device, max_prompt_length).squeeze(0)
        for text in prompt_texts
    ]
    input_ids, attn_mask = pad_left_batch(rows, tokenizer.pad_token_id, device)
    return input_ids, attn_mask, input_ids.size(1)


# ─────────────────────────────────────────────────────────────────────────────
# vLLM 评估 + rollout（同一实例）
# ─────────────────────────────────────────────────────────────────────────────
def vllm_eval_and_rollout(
    model_path: str,
    tokenizer: AutoTokenizer,
    all_prompts: List[str],
    all_answers: List[str],
    n_roll: int,
    max_new_tokens: int,
    rollout_temperature: float,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int = 1,
):
    """同一 vLLM 实例完成 greedy eval + temperature rollout。

    返回：
        base_predictions    : List[str]
        mistake_indices     : List[int]
        all_rollouts        : List[List[str]]            （文本，用于评判 reward）
        all_rollout_tok_ids : List[List[List[int]]]      （token id，用于取首 token）
    """
    if not _is_vllm_available():
        raise RuntimeError(f"vLLM 不可用: {_VLLM_IMPORT_ERROR}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        disable_custom_all_reduce=(tensor_parallel_size > 1),
    )
    stop_ids = _build_stop_token_ids(tokenizer)

    # Phase A: greedy eval
    eval_sampling = SamplingParams(
        n=1, temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=stop_ids
    )
    eval_outputs = llm.generate(all_prompts, eval_sampling)
    base_predictions = [req.outputs[0].text for req in eval_outputs]

    mistake_indices = [
        i
        for i, (pred, ref) in enumerate(zip(base_predictions, all_answers))
        if not check_correctness(pred, ref)
    ]

    all_rollouts: List[List[str]] = []
    all_rollout_tok_ids: List[List[List[int]]] = []

    # Phase B: temperature rollout（仅错题）
    if mistake_indices:
        mistake_prompts = [all_prompts[i] for i in mistake_indices]
        rollout_sampling = SamplingParams(
            n=n_roll,
            temperature=rollout_temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
        )
        rollout_outputs = llm.generate(mistake_prompts, rollout_sampling)
        for req in rollout_outputs:
            all_rollouts.append([o.text for o in req.outputs])
            all_rollout_tok_ids.append([list(o.token_ids) for o in req.outputs])

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_predictions, mistake_indices, all_rollouts, all_rollout_tok_ids


# ─────────────────────────────────────────────────────────────────────────────
# 核心：构造首 token 目标分布（忠实你原始的“逐 token 独立奖惩”设计）
# ─────────────────────────────────────────────────────────────────────────────
def build_first_token_target_logprobs(
    student_first_logits: torch.Tensor,
    first_token_ids: List[int],
    rewards: List[float],
    alpha: float,
    delta: float,
) -> tuple[torch.Tensor, bool]:
    """根据组内 rollout 二值奖励构造首 token KL 目标分布的 log-prob。

    逐 token id 独立奖惩（同一题中奖励/压制可同时发生）：
      - 该 id 至少有一条 rollout 答对 → 奖励：p ← p + (p_max - p) · α
      - 该 id 全部 rollout 答错      → 压制：p ← p · (1 - δ)
    重归一化后取 log。当一组 rollout 全错时（所有出现过的 token 都被压制），
    归一化会自动把概率分给“未尝试过的 token”，从而鼓励探索新方向。

    Args:
        student_first_logits : [1, V]  当前 step 学生在首 token 位置的 logits（float32, detached）
        first_token_ids      : 长度 = n_roll，每条 rollout 的首 token id
        rewards              : 长度 = n_roll，1.0=答对 / 0.0=答错
        alpha                : 正确 token 提升幅度（默认 1.0）
        delta                : 错误 token 压制幅度（默认 1.0；δ=1 等价直接清零）

    Returns:
        log_target : [1, V]，目标分布的 log-prob（float32）
        is_explore : bool，组内全错时为 True（用于日志统计，不影响算法）
    """
    with torch.no_grad():
        probs = F.softmax(student_first_logits.float(), dim=-1).clone()  # [1, V]
    p_max = probs.max().item()

    # 去重 + 聚合：同一 token id 可能多次出现，只奖惩一次
    tid_any_correct: dict = {}
    for tid, r in zip(first_token_ids, rewards):
        ok = r > 0.5
        tid_any_correct[tid] = tid_any_correct.get(tid, False) or ok

    # 逐 token 独立奖惩
    for tid, any_correct in tid_any_correct.items():
        if any_correct:
            p_cur = probs[0, tid].item()
            probs[0, tid] = p_cur + (p_max - p_cur) * alpha
        else:
            probs[0, tid] = probs[0, tid] * (1.0 - delta)

    # 归一化（被压制的概率自动流向“未尝试 token”）
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    is_explore = not any(tid_any_correct.values())  # 仅用于日志
    return torch.log(probs.clamp(min=1e-12)), is_explore


# ─────────────────────────────────────────────────────────────────────────────
# 训练主入口
# ─────────────────────────────────────────────────────────────────────────────
def train_a_token_sd(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    n_roll: int = 8,
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
    vllm_tensor_parallel_size: int = 1,
    gradient_checkpointing: bool = True,
    kl_max: float = 0.5,
    alpha: float = 1.0,
    delta: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """A-Token 自蒸馏训练。

    Args:
        alpha   : 正确 token 概率提升幅度（推荐 0.5–1.0；默认 1.0=拉到 p_max）。
        delta   : 错误 token 压制幅度（推荐 0.5–1.0；δ=1 → 清零，鼓励探索；
                  δ<1 → 软压制，更温和）。
        kl_max  : 单题 KL 软截断上限（cap*tanh(x/cap)，保留梯度）。
    """
    if device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "train_a_token_sd 需要 CUDA 设备（vLLM 不支持 CPU）。"
            f" 当前 device={device!r}"
        )
    os.makedirs(output_dir, exist_ok=True)
    step_log_file, epoch_log_file = setup_logging(output_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        eos_int = _first_int(tokenizer.eos_token_id)
        if eos_int is None:
            raise ValueError(
                "tokenizer 没有 pad_token_id，且 eos_token_id 也无法解析为 int。"
            )
        tokenizer.pad_token_id = eos_int
        logger.info(f"pad_token_id 未设置，已自动用 eos_token_id={eos_int}。")

    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    student_model = _build_models(
        model_path,
        torch_dtype,
        device,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        gradient_checkpointing=gradient_checkpointing,
    )

    torch.set_float32_matmul_precision("high")
    logger.info("使用 eager 模式（不启用 torch.compile）。")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    use_amp = device != "cpu" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16  # bf16 路径，不需要 GradScaler
    metrics_tracker = StepMetricsTracker()

    questions = [
        normalize_question_text(item.get("question", item.get("prompt", "")))
        for item in raw_data
    ]
    answers = [
        normalize_reference_answer(item.get("answer", item.get("ref_answer", "")))
        for item in raw_data
    ]
    max_model_len = max_prompt_length + max_new_tokens
    fallback_first_id = tokenizer.pad_token_id  # 已确保为 int
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        metrics_tracker.reset_epoch()

        # ── Phase 1+2: vLLM eval + rollout（同一实例）─────────────────────
        all_prompts_for_vllm = [build_prompt(tokenizer, q) for q in questions]
        _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_vllm_")
        try:
            student_model.cpu()
            torch.cuda.empty_cache()
            if use_lora:
                _save_merged_lora_for_vllm(
                    student_model, tokenizer, _vllm_tmp, base_model_path=model_path
                )
            else:
                student_model.save_pretrained(_vllm_tmp)
                tokenizer.save_pretrained(_vllm_tmp)

            (
                base_predictions,
                mistake_indices,
                all_rollouts,
                all_rollout_tok_ids,
            ) = vllm_eval_and_rollout(
                _vllm_tmp,
                tokenizer,
                all_prompts=all_prompts_for_vllm,
                all_answers=answers,
                n_roll=n_roll,
                max_new_tokens=max_new_tokens,
                rollout_temperature=rollout_temperature,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=vllm_tensor_parallel_size,
            )
        finally:
            shutil.rmtree(_vllm_tmp, ignore_errors=True)
            student_model.to(device)

        n_correct_phase1 = sum(
            1 for p, a in zip(base_predictions, answers) if check_correctness(p, a)
        )
        mistake_count = len(mistake_indices)
        logger.info(
            f"Epoch {epoch} Eval: total={len(questions)}, correct={n_correct_phase1}, "
            f"mistakes={mistake_count}, acc={n_correct_phase1/max(len(questions),1):.4f}"
        )
        if mistake_count == 0:
            logger.info(f"Epoch {epoch}: no mistakes, skipping training.")
            continue

        mistake_answers = [
            normalize_reference_answer(
                raw_data[i].get("answer", raw_data[i].get("ref_answer", ""))
            )
            for i in mistake_indices
        ]
        mistake_prompts = [all_prompts_for_vllm[i] for i in mistake_indices]

        # 二值奖励
        all_rewards: List[List[float]] = [
            [1.0 if check_correctness(seq, ref) else 0.0 for seq in seqs]
            for ref, seqs in zip(mistake_answers, all_rollouts)
        ]

        # ── Phase 3: 训练 — 首 token KL ─────────────────────────────────
        student_model.train()
        optimizer.zero_grad()

        mistake_progress = tqdm(
            range(0, mistake_count, rollout_batch_size),
            desc=f"Epoch {epoch} Train",
        )
        for i in mistake_progress:
            batch_slice = slice(i, i + rollout_batch_size)
            batch_prompts = mistake_prompts[batch_slice]
            batch_rollout_tok_ids = all_rollout_tok_ids[batch_slice]
            batch_rewards = all_rewards[batch_slice]

            # 直接从 vLLM 返回的 token_ids 取首 token（避免 BPE 拼接边界问题）
            # 空 rollout（tok_ids 长度为 0，极端情况：stop token 被 vLLM 过滤后序列为空）
            # 直接跳过，不参与奖惩；同步过滤对应的 reward，保持两个列表等长。
            batch_first_token_ids: List[List[int]] = []
            batch_rewards_clean: List[List[float]] = []
            for rollout_tok_ids, rewards_raw in zip(batch_rollout_tok_ids, batch_rewards):
                first_ids = []
                rewards_clean = []
                for tok_ids, r in zip(rollout_tok_ids, rewards_raw):
                    if len(tok_ids) > 0:
                        first_ids.append(int(tok_ids[0]))
                        rewards_clean.append(r)
                    # 空 rollout 直接跳过，不用 fallback_first_id 占位
                batch_first_token_ids.append(first_ids)
                batch_rewards_clean.append(rewards_clean)
            batch_rewards = batch_rewards_clean

            input_ids, attn_mask, prompt_max_len = batch_prompt_ids(
                tokenizer, batch_prompts, device, max_prompt_length
            )
            last_idx = prompt_max_len - 1  # 左填充：最后一个有效 token

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits_train = student_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits  # [B, seq, V]
                batch_first_logits = logits_train[:, last_idx, :].float()  # [B, V]
                batch_first_logprobs = F.log_softmax(batch_first_logits, dim=-1)

            batch_losses: List[torch.Tensor] = []
            batch_kl_vals: List[float] = []
            batch_n_correct = 0
            batch_n_explore = 0

            for first_logits_j, first_logprobs_j, first_tids, rewards_j in zip(
                batch_first_logits,
                batch_first_logprobs,
                batch_first_token_ids,
                batch_rewards,
            ):
                target_logprobs, is_explore = build_first_token_target_logprobs(
                    first_logits_j.unsqueeze(0).detach(),
                    first_tids,
                    rewards_j,
                    alpha=alpha,
                    delta=delta,
                )
                kl_raw = F.kl_div(
                    first_logprobs_j.unsqueeze(0),
                    target_logprobs,
                    reduction="batchmean",
                    log_target=True,
                ).clamp(min=0.0)  # KL 理论上 ≥ 0，clamp 消除数值误差产生的小负值，防止梯度反转
                kl = _soft_cap(kl_raw, kl_max)  # 软截断，保留梯度
                batch_losses.append(kl)
                batch_kl_vals.append(kl_raw.detach().item())
                if any(r > 0.5 for r in rewards_j):
                    batch_n_correct += 1
                if is_explore:
                    batch_n_explore += 1

            raw_loss = torch.stack(batch_losses).mean()
            loss = raw_loss / gradient_accumulation_steps
            loss.backward()  # bf16 不需要 GradScaler

            metrics_tracker.update(
                kl_values=batch_kl_vals,
                n_correct=batch_n_correct,
                n_total=len(batch_prompts),
                n_explore=batch_n_explore,
            )

            global_step += 1
            is_last_batch = (i + rollout_batch_size) >= mistake_count
            if global_step % gradient_accumulation_steps == 0 or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

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
                    log_parts.append(
                        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    )
                logger.info(" | ".join(log_parts))
                metrics_tracker.reset_window()

        # ── Epoch 总结 ──────────────────────────────────────────────────
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
        logger.info(
            f"  phase1_acc={epoch_record['phase1_acc']:.4f}  mistakes={mistake_count}"
        )
        logger.info(f"  avg_kl={ep_stats['avg_kl']:.6f}")
        logger.info(
            f"  rollout_correct_rate={ep_stats['correct_rate']:.4f}  "
            f"({ep_stats['n_correct']}/{ep_stats['n_total']})"
        )
        logger.info(
            f"  explore_ratio={ep_stats['explore_ratio']:.4f}  "
            f"({ep_stats['n_explore']}/{ep_stats['n_total']})"
        )
        logger.info("=" * 60)

        # ── Checkpoint ──────────────────────────────────────────────────
        if save_total_limit > 0:
            save_interval = max(1, num_epochs // save_total_limit)
            if epoch % save_interval == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Checkpoint saved → {ckpt_dir}")

    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# API 包装
# ─────────────────────────────────────────────────────────────────────────────
def train_a_token_sd_api(
    questions,
    answers,
    epoch,
    output_dir=None,
    model_path_override=None,
    use_lora=True,
    learning_rate=1e-6,
    n_roll=8,
    max_prompt_length=1024,
    max_new_tokens=4096,
    vllm_gpu_memory_utilization=0.85,
    rollout_temperature=0.8,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    gradient_accumulation_steps=4,
    rollout_batch_size=8,
    log_interval=10,
    save_total_limit=10,
    vllm_tensor_parallel_size=1,
    gradient_checkpointing=True,
    kl_max=0.5,
    alpha=1.0,
    delta=1.0,
    device=None,
):
    if not model_path_override:
        raise ValueError(
            "train_a_token_sd_api: 必须通过 model_path_override 指定模型路径。"
        )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)
    train_samples = [
        {"question": str(q).strip(), "answer": str(a).strip()}
        for q, a in zip(questions, answers)
    ]
    os.makedirs(resolved_output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="a_token_sd_",
        dir=resolved_output_dir,
        delete=False,
    ) as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        temp_data_path = f.name
    try:
        train_a_token_sd(
            model_path=model_path_override,
            data_path=temp_data_path,
            output_dir=resolved_output_dir,
            num_epochs=epoch,
            learning_rate=learning_rate,
            n_roll=n_roll,
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
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            gradient_checkpointing=gradient_checkpointing,
            kl_max=kl_max,
            alpha=alpha,
            delta=delta,
            device=resolved_device,
        )
    finally:
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
    return {"output_dir": resolved_output_dir}


def _build_a_token_sd_output_dir(epoch: int) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        f"a_token_sd_{epoch}ep_{datetime.now().strftime('%m%d_%H%M')}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO-style A-Token 自蒸馏训练（vLLM rollout + 首 token KL loss）"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_roll", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--rollout_temperature", type=float, default=0.8)
    parser.add_argument("--rollout_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="正确 token 概率提升幅度"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="错误 token 压制幅度（1.0=清零鼓励探索，<1=软压制）",
    )
    parser.add_argument(
        "--kl_max", type=float, default=0.5, help="单题 KL 软截断上限"
    )
    args = parser.parse_args()

    train_a_token_sd(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        n_roll=args.n_roll,
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
        alpha=args.alpha,
        delta=args.delta,
        kl_max=args.kl_max,
    )

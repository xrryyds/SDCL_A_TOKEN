import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Sequence

# 必须在 import torch / vLLM 之前设置，PyTorch 在首次 CUDA 分配时读取此变量。
# expandable_segments:True 消除 reserved-but-unallocated 碎片导致的 OOM。
# 新版 PyTorch (2.x) 使用 PYTORCH_ALLOC_CONF，旧版使用 PYTORCH_CUDA_ALLOC_CONF，两个都设。
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_TqdmLoggingHandler()],
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = r"Please reason step by step and put your final answer within \boxed{}."


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
    step_log_file = os.path.join(output_dir, "step_metrics.jsonl")
    epoch_log_file = os.path.join(output_dir, "epoch_metrics.jsonl")
    return step_log_file, epoch_log_file


class StepMetricsTracker:
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


def _stringify_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_question_text(value) -> str:
    return _stringify_text(value).strip()


def extract_answer(text: str) -> str:
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


def _build_stop_token_ids(tokenizer) -> List[int]:
    ids = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        ids.update(eos)
    elif eos is not None:
        ids.add(eos)
    pad = tokenizer.pad_token_id
    if pad is not None:
        ids.add(pad)
    ids.add(151645)
    return list(ids)


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
    target_modules = [
        module_name for module_name in common_targets if module_name in available
    ]
    if not target_modules:
        raise ValueError("无法推断 LoRA 目标模块。")
    return target_modules


def _save_merged_lora_for_vllm(
    student_model: "PeftModel", tokenizer: "AutoTokenizer", tmp_dir: str
) -> None:
    logger.info("正在合并 LoRA 权重以供 vLLM 使用（保留原 PeftModel adapter）...")
    import copy as _copy

    cpu_copy = _copy.deepcopy(student_model)
    merged = cpu_copy.merge_and_unload()
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    del merged, cpu_copy
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
    device_map: Optional[dict] = None,
):
    student_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    if device_map is None:
        student_model = student_model.to(device)
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
        student_model = get_peft_model(student_model, peft_config)
        if device_map is None:
            student_model = student_model.to(device)
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
    max_len = input_ids.size(1)
    return input_ids, attn_mask, max_len


def _extract_first_token_from_solution(
    tokenizer: AutoTokenizer,
    solution_text: str,
) -> tuple[int, str]:
    """
    从 solution 文本中提取首个 token。

    返回:
        (first_token_id, first_token_text)
    """
    solution_text = _stringify_text(solution_text)
    token_ids = tokenizer(solution_text, add_special_tokens=False).input_ids
    if not token_ids:
        fallback_id = tokenizer.eos_token_id
        if fallback_id is None:
            fallback_id = tokenizer.pad_token_id
        if fallback_id is None:
            raise ValueError("tokenizer 必须提供 eos_token_id 或 pad_token_id。")
        return fallback_id, tokenizer.decode([fallback_id], skip_special_tokens=False)
    first_token_id = token_ids[0]
    first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=False)
    return first_token_id, first_token_text


def build_first_token_target_logprobs_fill(
    student_first_logits: torch.Tensor,
    solution_first_token_id: int,
    student_pred_first_token_id: int,
    is_fill_correct: bool,
    alpha: float,
    delta: float,
) -> torch.Tensor:
    """
    只对一个填充 token 做奖惩，并对原首 token 做相反操作。

    - 若填充后答对：提高 solution 首 token，惩罚模型原首 token（若两者不同）。
    - 若填充后答错：惩罚 solution 首 token，保留/不奖励其它 token。

    奖惩幅度与原算法保持一致：
      奖励: p <- p + (p_max - p) * alpha
      惩罚: p <- p * (1 - delta)

    注意：当 solution_first_token_id == student_pred_first_token_id 时，
    不做惩罚（因为奖励和惩罚会矛盾），只做奖励。
    """
    with torch.no_grad():
        probs = F.softmax(student_first_logits.float(), dim=-1).clone()
    p_max = probs.max().item()

    if is_fill_correct:
        # 奖励 solution 首 token
        p_cur = probs[0, solution_first_token_id].item()
        probs[0, solution_first_token_id] = p_cur + (p_max - p_cur) * alpha
        # 惩罚模型原首 token（仅当两者不同时）
        if student_pred_first_token_id != solution_first_token_id:
            probs[0, student_pred_first_token_id] = (
                probs[0, student_pred_first_token_id] * (1.0 - delta)
            )
    else:
        # 惩罚 solution 首 token
        probs[0, solution_first_token_id] = (
            probs[0, solution_first_token_id] * (1.0 - delta)
        )

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.log(probs.clamp(min=1e-12))


def vllm_eval_and_fill_test(
    model_path: str,
    tokenizer: "AutoTokenizer",
    all_prompts: List[str],
    all_answers: List[str],
    all_solution_first_token_texts: List[str],
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    tensor_parallel_size: int = 1,
):
    """
    在同一个 vLLM 实例内完成：
      1. greedy eval 识别错题
      2. 对错题使用 solution 首 token 作为 prefix-forcing 填充
      3. 测试填充后是否答对

    参考 exam_with_hints 的实现，但这里只填充 solution 的首个 token。
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

    eval_sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
    )
    eval_outputs = llm.generate(all_prompts, eval_sampling)
    base_predictions = [req.outputs[0].text for req in eval_outputs]

    mistake_indices = [
        i
        for i, (pred, ref) in enumerate(zip(base_predictions, all_answers))
        if not check_correctness(pred, ref)
    ]

    fill_predictions: List[str] = []
    fill_correct_flags: List[bool] = []
    if mistake_indices:
        # prefix-forcing：将 solution 首 token 拼到 prompt 后面，参考 exam_with_hints
        fill_prompts = [
            all_prompts[i] + all_solution_first_token_texts[i] for i in mistake_indices
        ]
        fill_outputs = llm.generate(fill_prompts, eval_sampling)
        # vLLM generate 返回的 text 是 prompt 之后新生成的部分（不含 prompt 和填充的 token），
        # 所以完整回答 = 填充 token + 生成文本，需要拼接后再检查答案。
        fill_predictions = [req.outputs[0].text for req in fill_outputs]
        fill_correct_flags = [
            check_correctness(
                all_solution_first_token_texts[idx] + pred,
                all_answers[idx],
            )
            for pred, idx in zip(fill_predictions, mistake_indices)
        ]

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_predictions, mistake_indices, fill_predictions, fill_correct_flags


def train_a_token_sd(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 1e-6,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 4096,
    vllm_gpu_memory_utilization: float = 0.85,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    gradient_accumulation_steps: int = 4,
    rollout_batch_size: int = 8,
    log_interval: int = 10,
    save_total_limit: int = 10,
    vllm_tensor_parallel_size: int = 4,
    gradient_checkpointing: bool = False,
    kl_max: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    A-Token 自蒸馏（fill 版本）。

    每个 epoch 流程：
      1. vLLM greedy eval 全部题目，识别错题。
      2. 不再做 GRPO roll n；改为将 solution 的首个 token 直接填充到 prompt 后。
      3. 参考 exam_with_hints 做 prefix-forcing，测试该 token 是否能帮助答对。
      4. 若能答对：奖励填充 token，惩罚模型原首 token。
      5. 若不能答对：惩罚填充 token。
      6. 仅在首 token 位置做 KL 自蒸馏。
    """
    if device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError(
            "train_a_token_sd 需要 CUDA 设备（vLLM 不支持 CPU）。"
            f" 当前 device={device!r}, cuda_available={torch.cuda.is_available()}"
        )

    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 4:
        raise RuntimeError(
            f"4 卡模式至少需要 4 张 CUDA 卡，当前仅检测到 {cuda_device_count} 张。"
        )
    if vllm_tensor_parallel_size != 4:
        logger.warning(
            "检测到 4 卡训练目标，自动将 vllm_tensor_parallel_size 从 %s 调整为 4。",
            vllm_tensor_parallel_size,
        )
        vllm_tensor_parallel_size = 4
    if gradient_checkpointing:
        logger.warning(
            "140GB * 4 环境默认关闭 gradient_checkpointing 以提升吞吐，已自动关闭。"
        )
        gradient_checkpointing = False

    train_device = "cuda:0"
    device_map = None
    os.makedirs(output_dir, exist_ok=True)
    step_log_file, epoch_log_file = setup_logging(output_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    student_model = _build_models(
        model_path,
        torch_dtype,
        train_device,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
        gradient_checkpointing=gradient_checkpointing,
        device_map=device_map,
    )

    torch.set_float32_matmul_precision("high")
    compiled_model = student_model
    _compile_ok = False
    logger.info("使用 eager 模式（仅首 token 蒸馏）。")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    use_amp = train_device != "cpu" and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    metrics_tracker = StepMetricsTracker()

    questions = [
        normalize_question_text(item.get("question", item.get("prompt", "")))
        for item in raw_data
    ]
    answers = [
        normalize_reference_answer(item.get("answer", item.get("ref_answer", "")))
        for item in raw_data
    ]
    solutions = [
        _stringify_text(item.get("solution", item.get("ref_solution", "")))
        for item in raw_data
    ]
    max_model_len = max_prompt_length + max_new_tokens
    global_step = 0

    solution_first_token_ids: List[int] = []
    solution_first_token_texts: List[str] = []
    for sol in solutions:
        tid, ttext = _extract_first_token_from_solution(tokenizer, sol)
        solution_first_token_ids.append(tid)
        solution_first_token_texts.append(ttext)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        metrics_tracker.reset_epoch()

        all_prompts_for_vllm = [build_prompt(tokenizer, q) for q in questions]
        _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_fill_vllm_")
        try:
            student_model.cpu()
            torch.cuda.empty_cache()
            if use_lora:
                _save_merged_lora_for_vllm(student_model, tokenizer, _vllm_tmp)
            else:
                student_model.save_pretrained(_vllm_tmp)
                tokenizer.save_pretrained(_vllm_tmp)

            (
                base_predictions,
                mistake_indices,
                fill_predictions,
                fill_correct_flags,
            ) = vllm_eval_and_fill_test(
                _vllm_tmp,
                tokenizer,
                all_prompts=all_prompts_for_vllm,
                all_answers=answers,
                all_solution_first_token_texts=solution_first_token_texts,
                max_new_tokens=max_new_tokens,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=vllm_tensor_parallel_size,
            )
        finally:
            shutil.rmtree(_vllm_tmp, ignore_errors=True)
            student_model.to(train_device)
            if _compile_ok:
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        n_correct_phase1 = sum(
            1 for p, a in zip(base_predictions, answers) if check_correctness(p, a)
        )
        mistake_count = len(mistake_indices)
        fill_success_count = sum(1 for x in fill_correct_flags if x)
        logger.info(
            f"Epoch {epoch} Eval: total={len(questions)}, correct={n_correct_phase1}, "
            f"mistakes={mistake_count}, acc={n_correct_phase1/max(len(questions),1):.4f}, "
            f"fill_success={fill_success_count}"
        )
        if mistake_count == 0:
            logger.info(f"Epoch {epoch}: no mistakes, skipping training.")
            continue

        mistake_prompts = [all_prompts_for_vllm[i] for i in mistake_indices]
        mistake_answers = [answers[i] for i in mistake_indices]
        mistake_base_predictions = [base_predictions[i] for i in mistake_indices]
        mistake_solution_token_ids = [solution_first_token_ids[i] for i in mistake_indices]
        mistake_fill_correct_flags = fill_correct_flags

        student_model.train()
        optimizer.zero_grad()

        mistake_progress = tqdm(
            range(0, mistake_count, rollout_batch_size),
            desc=f"Epoch {epoch} Train",
        )
        for i in mistake_progress:
            batch_slice = slice(i, i + rollout_batch_size)
            batch_prompts = mistake_prompts[batch_slice]
            batch_refs = mistake_answers[batch_slice]
            batch_base_predictions = mistake_base_predictions[batch_slice]
            batch_solution_token_ids = mistake_solution_token_ids[batch_slice]
            batch_fill_correct_flags = mistake_fill_correct_flags[batch_slice]

            input_ids, attn_mask, prompt_max_len = batch_prompt_ids(
                tokenizer, batch_prompts, train_device, max_prompt_length
            )
            last_idx = prompt_max_len - 1

            with autocast(enabled=use_amp):
                logits_train = compiled_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits
                batch_first_logits = logits_train[:, last_idx, :].float()
                batch_first_logprobs = F.log_softmax(batch_first_logits, dim=-1)

            batch_losses: List[torch.Tensor] = []
            batch_kl_vals: List[float] = []
            batch_n_correct = 0

            for (
                first_logits_j,
                first_logprobs_j,
                base_pred_j,
                solution_tid_j,
                is_fill_correct_j,
                ref_j,
            ) in zip(
                batch_first_logits,
                batch_first_logprobs,
                batch_base_predictions,
                batch_solution_token_ids,
                batch_fill_correct_flags,
                batch_refs,
            ):
                # 从 logits 直接取 argmax 作为模型原首 token，避免 decode→re-tokenize 的 BPE 不一致
                student_pred_first_token_id = first_logits_j.argmax(dim=-1).item()

                dyn_alpha = 1.0
                dyn_delta = 1.0
                target_logprobs = build_first_token_target_logprobs_fill(
                    first_logits_j.unsqueeze(0).detach(),
                    solution_first_token_id=solution_tid_j,
                    student_pred_first_token_id=student_pred_first_token_id,
                    is_fill_correct=is_fill_correct_j,
                    alpha=dyn_alpha,
                    delta=dyn_delta,
                )
                kl = F.kl_div(
                    first_logprobs_j.unsqueeze(0),
                    target_logprobs,
                    reduction="batchmean",
                    log_target=True,
                ).clamp(max=kl_max)
                batch_losses.append(kl)
                batch_kl_vals.append(kl.detach().item())
                if is_fill_correct_j and not check_correctness(base_pred_j, ref_j):
                    batch_n_correct += 1

            raw_loss = torch.stack(batch_losses).mean()
            loss = raw_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            metrics_tracker.update(
                loss=raw_loss.item(),
                kl_values=batch_kl_vals,
                n_correct=batch_n_correct,
                n_total=len(batch_prompts),
            )

            global_step += 1
            is_last_batch = (i + rollout_batch_size) >= mistake_count
            if global_step % gradient_accumulation_steps == 0 or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
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
                    _f.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                log_parts = [f"[Step {global_step}]", f"epoch={epoch}"]
                for k, v in win_stats.items():
                    log_parts.append(
                        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                    )
                logger.info(" | ".join(log_parts))
                metrics_tracker.reset_window()

        ep_stats = metrics_tracker.get_epoch_stats()
        epoch_record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "n_mistakes": mistake_count,
            "phase1_acc": n_correct_phase1 / max(len(questions), 1),
            "fill_success": fill_success_count,
            **ep_stats,
        }
        with open(epoch_log_file, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        logger.info("=" * 60)
        logger.info(f"*** EPOCH {epoch}/{num_epochs} FINISHED ***")
        logger.info(
            f"  phase1_acc={epoch_record['phase1_acc']:.4f}  mistakes={mistake_count}"
        )
        logger.info(f"  fill_success={fill_success_count}")
        logger.info(
            f"  avg_loss={ep_stats['avg_loss']:.6f}  avg_kl={ep_stats['avg_kl']:.6f}"
        )
        logger.info(
            f"  correct_rate={ep_stats['correct_rate']:.4f}  ({ep_stats['n_correct']}/{ep_stats['n_total']})"
        )
        logger.info("=" * 60)

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


def train_a_token_sd_api_4(
    questions,
    answers,
    solutions,
    epoch,
    output_dir=None,
    model_path_override=None,
    use_lora=True,
    learning_rate=1e-5,
    max_prompt_length=1024,
    max_new_tokens=4096,
    vllm_gpu_memory_utilization=0.85,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    gradient_accumulation_steps=4,
    rollout_batch_size=8,
    log_interval=10,
    save_total_limit=10,
    vllm_tensor_parallel_size=4,
    gradient_checkpointing=False,
    kl_max=0.5,
    device=None,
):
    if not model_path_override:
        raise ValueError(
            "train_a_token_sd_api: 必须通过 model_path_override 指定模型路径，"
            "当前值为 None 或空字符串。"
        )
    resolved_model_path = model_path_override
    resolved_device = device or "cuda:0"
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)
    train_samples = [
        {
            "question": str(q).strip(),
            "answer": str(a).strip(),
            "solution": str(s).strip(),
        }
        for q, a, s in zip(questions, answers, solutions)
    ]
    os.makedirs(resolved_output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="a_token_sd_fill_",
        dir=resolved_output_dir,
        delete=False,
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
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
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
        f"a_token_sd_fill_{epoch}ep_{datetime.now().strftime('%m%d_%H%M')}",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="A-Token 自蒸馏训练（solution 首 token fill + 首 token KL loss）"
    )
    parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument(
        "--data_path", type=str, required=True, help="训练数据 JSON 路径（需包含 solution 字段）"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=64,
        help="训练 batch size",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    train_a_token_sd(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        rollout_batch_size=args.rollout_batch_size,
        log_interval=args.log_interval,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        gradient_checkpointing=args.gradient_checkpointing,
    )

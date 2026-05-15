import copy
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Sequence

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

# 配置日志记录，同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        _TqdmLoggingHandler(),
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ],
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \boxed{}."

def _stringify_text(value) -> str:
    """将输入转换为字符串。"""
    if value is None: return ""
    if isinstance(value, str): return value
    return str(value)

def normalize_question_text(value) -> str:
    """规范化问题文本。"""
    return _stringify_text(value).strip()

def extract_answer(text: str) -> str:
    """从文本中提取 \boxed{} 中的答案。"""
    text = _stringify_text(text).strip()
    if "\boxed{" in text:
        start = text.rfind("\boxed{") + len("\boxed{")
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
    """将 LoRA 权重合并到基础模型并保存到临时目录。"""
    logger.info("正在合并 LoRA 权重以供 vLLM 使用...")
    merged = copy.deepcopy(student_model).merge_and_unload()
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    del merged
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _build_models(model_path: str, torch_dtype: torch.dtype, device: str, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    """构建学生模型。"""
    student_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    if use_lora:
        target_modules = _infer_lora_target_modules(student_model)
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, task_type="CAUSAL_LM", bias="none")
        student_model = get_peft_model(student_model, peft_config).to(device)
    student_model.config.use_cache = False
    return student_model

def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """构建聊天模板提示词。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": normalize_question_text(question)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def sample_first_tokens(logits: torch.Tensor, n_roll: int, temperature: float = 0.0, top_k: int = 50) -> List[int]:
    """确定性 top-k 采样首个 token。"""
    k = min(n_roll, logits.size(-1))
    return torch.topk(logits.float(), k=k, dim=-1).indices.tolist()

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

def batch_prompt_ids(tokenizer: AutoTokenizer, prompt_texts: List[str], device: str, max_prompt_length: int) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    """对多条 prompt 一次性构造左填充 batch。"""
    rows = [tokenize_prompt(tokenizer, text, device, max_prompt_length).squeeze(0) for text in prompt_texts]
    input_ids, attn_mask = pad_left_batch(rows, tokenizer.pad_token_id, device)
    row_lens = [row.size(0) for row in rows]
    return input_ids, attn_mask, row_lens

def generate_with_hints_batch(model: torch.nn.Module, tokenizer: AutoTokenizer, question: str, hint_token_ids_batch: Sequence[Sequence[int]], max_prompt_length: int, max_new_tokens: int, device: str) -> List[str]:
    """批量生成带提示的回答。"""
    prompt_text = build_prompt(tokenizer, question)
    input_id_rows = [torch.cat([tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length), torch.tensor([list(h)], dtype=torch.long, device=device)], dim=-1).squeeze(0) for h in hint_token_ids_batch]
    input_ids, attn_mask = pad_left_batch(input_id_rows, tokenizer.pad_token_id, device)
    max_input_len = input_ids.size(1)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(gen[max_input_len:], skip_special_tokens=True) for gen in outputs]

def generate_with_prompt_hints_batch(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt_texts: List[str], hint_token_ids_per_prompt: List[List[List[int]]], max_prompt_length: int, max_new_tokens: int, device: str) -> List[List[str]]:
    """将多题多候选 flatten 后一次 generate，提高吞吐。"""
    flat_rows: List[torch.Tensor] = []
    counts: List[int] = []
    for prompt_text, hint_groups in zip(prompt_texts, hint_token_ids_per_prompt):
        prompt_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
        count = 0
        for hint_ids in hint_groups:
            hint_tensor = torch.tensor([list(hint_ids)], dtype=torch.long, device=device)
            row = torch.cat([prompt_ids, hint_tensor], dim=-1)[:, -max_prompt_length:].squeeze(0)
            flat_rows.append(row)
            count += 1
        counts.append(count)

    if not flat_rows:
        return [[] for _ in prompt_texts]

    input_ids, attn_mask = pad_left_batch(flat_rows, tokenizer.pad_token_id, device)
    max_input_len = input_ids.size(1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    flat_answers = [tokenizer.decode(gen[max_input_len:], skip_special_tokens=True) for gen in outputs]
    grouped_answers: List[List[str]] = []
    idx = 0
    for count in counts:
        grouped_answers.append(flat_answers[idx: idx + count])
        idx += count
    return grouped_answers

def evaluate_questions(model: torch.nn.Module, tokenizer: AutoTokenizer, questions: List[str], max_prompt_length: int, max_new_tokens: int, device: str, batch_size: int = 8) -> List[str]:
    """批量评估问题。"""
    predictions = []
    for start_idx in tqdm(range(0, len(questions), batch_size), desc="Phase 1 Eval"):
        batch_q = questions[start_idx : start_idx + batch_size]
        encoded = [tokenize_prompt(tokenizer, build_prompt(tokenizer, q), device, max_prompt_length).squeeze(0) for q in batch_q]
        input_ids, attn_mask = pad_left_batch(encoded, tokenizer.pad_token_id, device)
        max_len = input_ids.size(1)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        predictions.extend([tokenizer.decode(gen[max_len:], skip_special_tokens=True) for gen in outputs])
    return predictions

def generate_rollouts_vllm(model_path: str, tokenizer: AutoTokenizer, questions: List[str], hint_token_ids_per_question: List[List[List[int]]], max_prompt_length: int, max_new_tokens: int, gpu_memory_utilization: float = 0.85) -> List[List[str]]:
    """使用 vLLM 批量生成 rollout。"""
    flat_token_ids = []
    rollout_counts = []
    for q, hints in zip(questions, hint_token_ids_per_question):
        prompt_ids = tokenize_prompt(tokenizer, build_prompt(tokenizer, q), "cpu", max_prompt_length).squeeze(0).tolist()
        for h in hints:
            flat_token_ids.append(prompt_ids[-max_prompt_length + len(h):] + h)
        rollout_counts.append(len(hints))
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_prompt_length + max_new_tokens, dtype="bfloat16")
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in flat_token_ids], SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=[tokenizer.eos_token_id]))
    flat_texts = [out.outputs[0].text for out in outputs]
    results, idx = [], 0
    for count in rollout_counts:
        results.append(flat_texts[idx : idx + count])
        idx += count
    return results

def evaluate_questions_vllm(model_path: str, tokenizer: AutoTokenizer, questions: List[str], max_prompt_length: int, max_new_tokens: int, batch_size: int = 8, gpu_memory_utilization: float = 0.9) -> List[str]:
    """使用 vLLM 批量评估。"""
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_prompt_length + max_new_tokens, dtype="bfloat16")
    outputs = llm.generate([build_prompt(tokenizer, q) for q in questions], SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop_token_ids=[tokenizer.eos_token_id]))
    return [out.outputs[0].text for out in outputs]

def build_first_token_target_logprobs(student_first_logits: torch.Tensor, sampled_token_ids: List[int], correct_token_ids: List[int], alpha: float, delta: float) -> torch.Tensor:
    """构建 KL 目标分布。"""
    with torch.no_grad():
        probs = F.softmax(student_first_logits.float(), dim=-1).clone()
    p_max = probs.max().item()
    correct_set = set(correct_token_ids)
    for tid in sampled_token_ids:
        if tid in correct_set:
            p_cur = probs[0, tid].item()
            probs[0, tid] = p_cur + (p_max - p_cur) * alpha
        else:
            probs[0, tid] *= (1.0 - delta)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.log(probs.clamp(min=1e-12))

def train_a_token_sd(model_path: str, data_path: str, output_dir: str, num_epochs: int = 3, learning_rate: float = 1e-3, n_roll: int = 8, alpha: float = 0.1, delta: float = 0.1, max_prompt_length: int = 1024, max_new_tokens: int = 2048, inference_batch_size: int = 4, eval_backend: str = "transformers", vllm_gpu_memory_utilization: float = 0.3, use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0, gradient_accumulation_steps: int = 4, rollout_batch_size: int = 8, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """主训练函数。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(data_path, "r", encoding="utf-8") as f: raw_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    student_model = _build_models(model_path, torch.bfloat16 if device != "cpu" else torch.float32, device, use_lora, lora_r, lora_alpha, lora_dropout)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    questions = [normalize_question_text(item.get("question", item.get("prompt", ""))) for item in raw_data]
    answers = [normalize_reference_answer(item.get("answer", item.get("ref_answer", ""))) for item in raw_data]
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        # 基础评估
        if eval_backend == "vllm":
            _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_vllm_")
            try:
                student_model.cpu()
                torch.cuda.empty_cache()
                if use_lora: _save_merged_lora_for_vllm(student_model, tokenizer, _vllm_tmp)
                else: student_model.save_pretrained(_vllm_tmp); tokenizer.save_pretrained(_vllm_tmp)
                base_predictions = evaluate_questions_vllm(_vllm_tmp, tokenizer, questions, max_prompt_length, max_new_tokens, inference_batch_size, vllm_gpu_memory_utilization)
            finally:
                shutil.rmtree(_vllm_tmp, ignore_errors=True)
                student_model.to(device)
        else: base_predictions = evaluate_questions(student_model, tokenizer, questions, max_prompt_length, max_new_tokens, device, inference_batch_size)
        
        mistakes = [raw_data[i] for i, pred in enumerate(base_predictions) if not check_correctness(pred, answers[i])]
        mistake_count = len(mistakes)
        if mistake_count == 0: continue
        
        student_model.train()
        epoch_loss, epoch_first_kl = 0.0, 0.0
        scaler = GradScaler()
        optimizer.zero_grad()
        
        mistake_progress = tqdm(range(0, mistake_count, rollout_batch_size), desc=f"Epoch {epoch} Train")
        for i in mistake_progress:
            batch_mistakes = mistakes[i : i + rollout_batch_size]
            batch_questions = [normalize_question_text(m.get("question", m.get("prompt", ""))) for m in batch_mistakes]
            batch_ref_answers = [normalize_reference_answer(m.get("answer", m.get("ref_answer", ""))) for m in batch_mistakes]
            batch_prompt_texts = [build_prompt(tokenizer, q) for q in batch_questions]
            
            # Phase A: 批量计算首 token logits 并选 top-n
            student_model.eval()
            with torch.no_grad():
                input_ids, attn_mask, row_lens = batch_prompt_ids(
                    tokenizer, batch_prompt_texts, device, max_prompt_length
                )
                logits = student_model(input_ids=input_ids, attention_mask=attn_mask).logits
                last_indices = torch.tensor([length - 1 for length in row_lens], device=device)
                batch_first_logits = logits[torch.arange(len(row_lens), device=device), last_indices, :]
                batch_sampled_ids = [
                    sample_first_tokens(row_logits, n_roll=n_roll, temperature=0.7, top_k=50)
                    for row_logits in batch_first_logits
                ]
            student_model.train()
            
            # Phase B: 将多题多候选合并成一个 generate batch
            batch_hint_ids = [[[tid] for tid in sampled] for sampled in batch_sampled_ids]
            batch_generated_answers = generate_with_prompt_hints_batch(
                student_model,
                tokenizer,
                batch_prompt_texts,
                batch_hint_ids,
                max_prompt_length,
                max_new_tokens,
                device,
            )
            
            # Phase C: 批量训练前向，再逐题构造 target 并累计 KL
            input_ids, attn_mask, row_lens = batch_prompt_ids(
                tokenizer, batch_prompt_texts, device, max_prompt_length
            )
            with autocast():
                out = student_model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits
                last_indices = torch.tensor([length - 1 for length in row_lens], device=device)
                batch_first_logits = logits[torch.arange(len(row_lens), device=device), last_indices, :]
                batch_first_logprobs = F.log_softmax(batch_first_logits, dim=-1)

                batch_losses = []
                batch_first_kl_values = []
                for idx, (ref, sampled, generated, hint_ids) in enumerate(
                    zip(batch_ref_answers, batch_sampled_ids, batch_generated_answers, batch_hint_ids)
                ):
                    correct_ids = [
                        tid
                        for tid, ans, h_id in zip(sampled, generated, hint_ids)
                        if check_correctness(
                            tokenizer.decode(h_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) + ans,
                            ref,
                        )
                    ]
                    target_logprobs = build_first_token_target_logprobs(
                        batch_first_logits[idx: idx + 1], sampled, correct_ids, alpha, delta
                    )
                    first_kl = F.kl_div(
                        batch_first_logprobs[idx: idx + 1],
                        target_logprobs,
                        reduction="batchmean",
                        log_target=True,
                    )
                    batch_losses.append(first_kl)
                    batch_first_kl_values.append(first_kl.detach())

                loss = torch.stack(batch_losses).sum() / gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            epoch_first_kl += sum(v.item() for v in batch_first_kl_values)
            
            if (i + rollout_batch_size) % (gradient_accumulation_steps * rollout_batch_size) == 0 or (i + rollout_batch_size) >= mistake_count:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        logger.info(
            f"Epoch {epoch} summary: avg_loss={epoch_loss/max(1, mistake_count):.6f}, "
            f"avg_first_kl={epoch_first_kl/max(1, mistake_count):.6f}"
        )
    
    student_model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")

def train_a_token_sd_api(questions, answers, epoch, output_dir=None, model_path_override=None, use_lora=True, learning_rate=1e-3, n_roll=8, alpha=0.1, delta=0.1, max_prompt_length=1024, max_new_tokens=2048, inference_batch_size=4, eval_backend="vllm", vllm_gpu_memory_utilization=0.85, lora_r=16, lora_alpha=32, lora_dropout=0.0, gradient_accumulation_steps=4, rollout_batch_size=8, device=None):
    """API 包装器。"""
    resolved_model_path = model_path_override
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)
    train_samples = [{"question": str(q).strip(), "answer": str(a).strip()} for q, a in zip(questions, answers)]
    os.makedirs(resolved_output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="a_token_sd_", dir=resolved_output_dir, delete=False) as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        temp_data_path = f.name
    train_a_token_sd(resolved_model_path, temp_data_path, resolved_output_dir, epoch, learning_rate, n_roll, alpha, delta, max_prompt_length, max_new_tokens, inference_batch_size, eval_backend, vllm_gpu_memory_utilization, use_lora, lora_r, lora_alpha, lora_dropout, gradient_accumulation_steps, rollout_batch_size, resolved_device)
    return {"output_dir": resolved_output_dir}

def _build_a_token_sd_output_dir(epoch: int) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"a_token_sd_{epoch}ep_{datetime.now().strftime('%m%d_%H%M')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train_a_token_sd(args.model_path, args.data_path, args.output_dir)

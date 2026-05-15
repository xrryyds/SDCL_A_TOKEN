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

def update_ema(teacher_model: torch.nn.Module, student_model: torch.nn.Module, decay: float = 0.99):
    """更新教师模型的 EMA 参数。"""
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)

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
    """构建学生和教师模型。"""
    student_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    teacher_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    if use_lora:
        target_modules = _infer_lora_target_modules(student_model)
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout, task_type="CAUSAL_LM", bias="none")
        student_model = get_peft_model(student_model, peft_config).to(device)
        teacher_model = get_peft_model(teacher_model, peft_config).to(device)
        teacher_model.load_state_dict(student_model.state_dict(), strict=False)
    teacher_model.eval()
    student_model.config.use_cache = False
    teacher_model.config.use_cache = False
    return student_model, teacher_model

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

def generate_with_hints_batch(model: torch.nn.Module, tokenizer: AutoTokenizer, question: str, hint_token_ids_batch: Sequence[Sequence[int]], max_prompt_length: int, max_new_tokens: int, device: str) -> List[str]:
    """批量生成带提示的回答。"""
    prompt_text = build_prompt(tokenizer, question)
    input_id_rows = [torch.cat([tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length), torch.tensor([list(h)], dtype=torch.long, device=device)], dim=-1).squeeze(0) for h in hint_token_ids_batch]
    max_input_len = max(row.size(0) for row in input_id_rows)
    input_ids = torch.full((len(input_id_rows), max_input_len), tokenizer.pad_token_id, dtype=input_id_rows[0].dtype, device=device)
    attn_mask = torch.zeros((len(input_id_rows), max_input_len), dtype=torch.long, device=device)
    for i, row in enumerate(input_id_rows):
        input_ids[i, -row.size(0):] = row
        attn_mask[i, -row.size(0):] = 1
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(gen[max_input_len:], skip_special_tokens=True) for gen in outputs]

def evaluate_questions(model: torch.nn.Module, tokenizer: AutoTokenizer, questions: List[str], max_prompt_length: int, max_new_tokens: int, device: str, batch_size: int = 8) -> List[str]:
    """批量评估问题。"""
    predictions = []
    for start_idx in tqdm(range(0, len(questions), batch_size), desc="Phase 1 Eval"):
        batch_q = questions[start_idx : start_idx + batch_size]
        encoded = [tokenize_prompt(tokenizer, build_prompt(tokenizer, q), device, max_prompt_length).squeeze(0) for q in batch_q]
        max_len = max(t.size(0) for t in encoded)
        input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=encoded[0].dtype, device=device)
        attn_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            input_ids[i, -row.size(0):] = row
            attn_mask[i, -row.size(0):] = 1
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

def build_rollout_weights(sampled_token_ids: List[int], correct_token_ids: List[int], alpha: float, delta: float, device: str) -> torch.Tensor:
    """构建 rollout 权重。"""
    weights = [max(0.0, 1.0 + alpha) if tid in set(correct_token_ids) else max(0.0, 1.0 - delta) for tid in sampled_token_ids]
    if not weights: return torch.tensor([], dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.sum() if w.sum() > 0 else torch.full_like(w, 1.0 / len(weights))

def compute_rest_trajectory_kl_batch(prompt_text: str, hint_token_ids_list: List[List[int]], generated_answers: List[str], teacher_model: torch.nn.Module, student_model: torch.nn.Module, tokenizer: AutoTokenizer, device: str, max_prompt_length: int) -> List[torch.Tensor]:
    """批量计算剩余轨迹的 KL 散度。"""
    results = []
    non_empty = [(i, ans) for i, ans in enumerate(generated_answers) if ans.strip()]
    if not non_empty: return [torch.tensor(0.0, device=device) for _ in generated_answers]
    
    prompt_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
    rows, p_lens, a_lens = [], [], []
    for i, ans in non_empty:
        prefix = torch.cat([prompt_ids, torch.tensor([hint_token_ids_list[i]], device=device)], dim=-1)[:, -max_prompt_length:]
        ans_ids = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        rows.append(torch.cat([prefix, ans_ids], dim=-1).squeeze(0))
        p_lens.append(prefix.size(-1)); a_lens.append(ans_ids.size(-1))
    
    max_len = max(r.size(0) for r in rows)
    batch_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=rows[0].dtype, device=device)
    attn_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for b, r in enumerate(rows):
        batch_ids[b, -r.size(0):] = r
        attn_mask[b, -r.size(0):] = 1
        
    with torch.no_grad():
        t_logprobs = F.log_softmax(teacher_model(batch_ids, attention_mask=attn_mask).logits[:, :-1, :], dim=-1)
    s_logprobs = F.log_softmax(student_model(batch_ids, attention_mask=attn_mask).logits[:, :-1, :], dim=-1)
    
    final_results = [torch.tensor(0.0, device=device) for _ in generated_answers]
    for b, (orig_idx, pl, al) in enumerate(zip([i for i, _ in non_empty], p_lens, a_lens)):
        start = max_len - rows[b].size(0) + max(pl - 1, 0)
        final_results[orig_idx] = F.kl_div(s_logprobs[b, start:start+al], t_logprobs[b, start:start+al], reduction="batchmean", log_target=True)
    return final_results

def train_a_token_sd(model_path: str, data_path: str, output_dir: str, num_epochs: int = 3, learning_rate: float = 1e-3, n_roll: int = 8, alpha: float = 0.1, delta: float = 0.1, w_tail: float = 1.0, ema_decay: float = 0.99, max_prompt_length: int = 1024, max_new_tokens: int = 2048, inference_batch_size: int = 4, eval_backend: str = "transformers", vllm_gpu_memory_utilization: float = 0.3, use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0, gradient_accumulation_steps: int = 4, use_ema: bool = False, use_rest_kl: bool = False, rollout_batch_size: int = 2, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """主训练函数。"""
    os.makedirs(output_dir, exist_ok=True)
    with open(data_path, "r", encoding="utf-8") as f: raw_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    student_model, teacher_model = _build_models(model_path, torch.bfloat16 if device != "cpu" else torch.float32, device, use_lora, lora_r, lora_alpha, lora_dropout)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
    questions = [normalize_question_text(item.get("question", item.get("prompt", ""))) for item in raw_data]
    answers = [normalize_reference_answer(item.get("answer", item.get("ref_answer", ""))) for item in raw_data]
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        # 基础评估
        if eval_backend == "vllm":
            _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_vllm_")
            try:
                student_model.cpu(); teacher_model.cpu(); torch.cuda.empty_cache()
                if use_lora: _save_merged_lora_for_vllm(student_model, tokenizer, _vllm_tmp)
                else: student_model.save_pretrained(_vllm_tmp); tokenizer.save_pretrained(_vllm_tmp)
                base_predictions = evaluate_questions_vllm(_vllm_tmp, tokenizer, questions, max_prompt_length, max_new_tokens, inference_batch_size, vllm_gpu_memory_utilization)
            finally: shutil.rmtree(_vllm_tmp, ignore_errors=True); student_model.to(device); teacher_model.to(device)
        else: base_predictions = evaluate_questions(student_model, tokenizer, questions, max_prompt_length, max_new_tokens, device, inference_batch_size)
        
        mistakes = [raw_data[i] for i, pred in enumerate(base_predictions) if not check_correctness(pred, answers[i])]
        mistake_count = len(mistakes)
        if mistake_count == 0: continue
        
        student_model.train()
        epoch_loss, epoch_first_kl, epoch_rest_kl = 0.0, 0.0, 0.0
        scaler = GradScaler()
        optimizer.zero_grad()
        
        mistake_progress = tqdm(range(0, mistake_count, rollout_batch_size), desc=f"Epoch {epoch} Train")
        for i in mistake_progress:
            batch_mistakes = mistakes[i : i + rollout_batch_size]
            batch_questions = [normalize_question_text(m.get("question", m.get("prompt", ""))) for m in batch_mistakes]
            batch_ref_answers = [normalize_reference_answer(m.get("answer", m.get("ref_answer", ""))) for m in batch_mistakes]
            
            # Phase A: 采样首 token
            batch_sampled_ids = []
            student_model.eval()
            with torch.no_grad():
                for q in batch_questions:
                    pt = build_prompt(tokenizer, q)
                    ids = tokenize_prompt(tokenizer, pt, device, max_prompt_length)
                    logits = student_model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1, :]
                    batch_sampled_ids.append(sample_first_tokens(logits[0], n_roll=n_roll, temperature=0.7, top_k=50))
            student_model.train()
            
            # Phase B: 生成 rollout
            batch_generated_answers = []
            if eval_backend == "vllm" and _is_vllm_available():
                _roll_tmp = tempfile.mkdtemp(prefix="a_token_sd_roll_")
                try:
                    student_model.cpu(); teacher_model.cpu(); torch.cuda.empty_cache()
                    if use_lora: _save_merged_lora_for_vllm(student_model, tokenizer, _roll_tmp)
                    else: student_model.save_pretrained(_roll_tmp); tokenizer.save_pretrained(_roll_tmp)
                    batch_generated_answers = generate_rollouts_vllm(_roll_tmp, tokenizer, batch_questions, [[[tid] for tid in s] for s in batch_sampled_ids], max_prompt_length + n_roll, max_new_tokens, vllm_gpu_memory_utilization)
                finally: shutil.rmtree(_roll_tmp, ignore_errors=True); student_model.to(device); teacher_model.to(device); torch.cuda.empty_cache()
            else:
                for q, sampled in zip(batch_questions, batch_sampled_ids):
                    batch_generated_answers.append(generate_with_hints_batch(student_model, tokenizer, q, [[tid] for tid in sampled], max_prompt_length, max_new_tokens, device))
            
            # Phase C: KL 损失计算与反向传播
            for q, ref, sampled, generated, hint_ids in zip(batch_questions, batch_ref_answers, batch_sampled_ids, batch_generated_answers, [[ [tid] for tid in s] for s in batch_sampled_ids]):
                pt = build_prompt(tokenizer, q)
                ids = tokenize_prompt(tokenizer, pt, device, max_prompt_length)
                with autocast():
                    out = student_model(input_ids=ids, attention_mask=torch.ones_like(ids))
                    first_logits = out.logits[:, -1, :]
                    first_logprobs = F.log_softmax(first_logits, dim=-1)
                    correct_ids = [tid for tid, ans, h_id in zip(sampled, generated, hint_ids) if check_correctness(tokenizer.decode(h_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) + ans, ref)]
                    target_logprobs = build_first_token_target_logprobs(first_logits, sampled, correct_ids, alpha, delta)
                    weights = build_rollout_weights(sampled, correct_ids, alpha, delta, device)
                    first_kl = F.kl_div(first_logprobs, target_logprobs, reduction="batchmean", log_target=True)
                    rest_kl = torch.sum(weights * torch.stack(compute_rest_trajectory_kl_batch(pt, hint_ids, generated, teacher_model, student_model, tokenizer, device, max_prompt_length))) if use_rest_kl else 0.0
                    loss = (first_kl + w_tail * rest_kl) / gradient_accumulation_steps
                scaler.scale(loss).backward()
                epoch_loss += loss.item(); epoch_first_kl += first_kl.item(); epoch_rest_kl += rest_kl if isinstance(rest_kl, (float, int)) else rest_kl.item()
            
            if (i + rollout_batch_size) % (gradient_accumulation_steps * rollout_batch_size) == 0 or (i + rollout_batch_size) >= mistake_count:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        
        if use_ema: update_ema(teacher_model, student_model, decay=ema_decay)
        logger.info(f"Epoch {epoch} summary: avg_loss={epoch_loss/max(1, mistake_count):.6f}")
    
    student_model.save_pretrained(output_dir); tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")

def train_a_token_sd_api(questions, answers, epoch, output_dir=None, model_path_override=None, use_lora=True, learning_rate=1e-3, n_roll=8, alpha=0.1, delta=0.1, w_tail=1.0, ema_decay=0.99, max_prompt_length=1024, max_new_tokens=2048, inference_batch_size=4, eval_backend="vllm", vllm_gpu_memory_utilization=0.85, lora_r=16, lora_alpha=32, lora_dropout=0.0, gradient_accumulation_steps=4, use_ema=False, use_rest_kl=False, rollout_batch_size=2, device=None):
    """API 包装器。"""
    resolved_model_path = model_path_override
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)
    train_samples = [{"question": str(q).strip(), "answer": str(a).strip()} for q, a in zip(questions, answers)]
    os.makedirs(resolved_output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="a_token_sd_", dir=resolved_output_dir, delete=False) as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
        temp_data_path = f.name
    train_a_token_sd(resolved_model_path, temp_data_path, resolved_output_dir, epoch, learning_rate, n_roll, alpha, delta, w_tail, ema_decay, max_prompt_length, max_new_tokens, inference_batch_size, eval_backend, vllm_gpu_memory_utilization, use_lora, lora_r, lora_alpha, lora_dropout, gradient_accumulation_steps, use_ema, use_rest_kl, rollout_batch_size, resolved_device)
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

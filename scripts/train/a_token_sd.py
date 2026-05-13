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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import-time environment specific
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = exc

class _TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write() so progress bars are not clobbered."""

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

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."


def _stringify_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_question_text(value) -> str:
    text = _stringify_text(value).strip()
    return text


def extract_answer(text: str) -> str:
    text = _stringify_text(text).strip()
    if "\\boxed{" in text:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()


def normalize_reference_answer(value) -> str:
    return extract_answer(value)


def check_correctness(pred: str, ref: str) -> bool:
    return extract_answer(pred) == extract_answer(ref)


def update_ema(teacher_model: torch.nn.Module, student_model: torch.nn.Module, decay: float = 0.99):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)


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
    target_modules = [module_name for module_name in common_targets if module_name in available]
    if not target_modules:
        raise ValueError(
            "Unable to infer LoRA target modules for this model architecture. "
            "Please provide explicit target modules in code."
        )
    return target_modules


def _save_merged_lora_for_vllm(
    student_model: "PeftModel",
    tokenizer: "AutoTokenizer",
    tmp_dir: str,
) -> None:
    """Merge LoRA weights into a deep-copied base model and save to *tmp_dir*.

    The original *student_model* (PeftModel) is **not** modified.  The merged
    copy is deleted from GPU memory before this function returns so that the
    caller can immediately launch vLLM without running out of VRAM.

    Args:
        student_model: A ``PeftModel`` wrapping an ``AutoModelForCausalLM``.
        tokenizer:     The tokenizer to save alongside the merged weights.
        tmp_dir:       Directory that will receive the merged model files.
    """
    logger.info("Merging LoRA weights into a temporary copy for vLLM eval …")
    # Deep-copy keeps the original student_model intact.
    merged = copy.deepcopy(student_model)
    # merge_and_unload() folds adapter weights into the base model in-place
    # and returns a plain AutoModelForCausalLM.
    merged = merged.merge_and_unload()
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)
    # Free the merged copy immediately so vLLM can use the VRAM.
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Merged model saved to {tmp_dir}")


def _build_models(
    model_path: str,
    torch_dtype: torch.dtype,
    device: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    student_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    if use_lora:
        target_modules = _infer_lora_target_modules(student_model)
        logger.info(f"Using LoRA target modules: {target_modules}")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
        )
        student_model = get_peft_model(student_model, peft_config).to(device)
        teacher_model = get_peft_model(teacher_model, peft_config).to(device)
        teacher_model.load_state_dict(student_model.state_dict(), strict=False)

    teacher_model.eval()
    student_model.config.use_cache = False
    teacher_model.config.use_cache = False
    return student_model, teacher_model


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def sample_first_tokens(logits: torch.Tensor, n_roll: int, temperature: float = 0.0, top_k: int = 50) -> List[int]:
    """Return the top-n_roll token ids by logit value (deterministic top-k).

    Using deterministic top-k ensures the sampled tokens have the highest
    probability mass, so alpha/delta adjustments produce meaningful KL signal.
    Stochastic sampling with replacement + unique() tends to collapse to 1-2
    tokens and may pick low-probability tokens, making the KL near zero.
    """
    k = min(n_roll, logits.size(-1))
    top_indices = torch.topk(logits.float(), k=k, dim=-1).indices
    return top_indices.tolist()


def tokenize_prompt(
    tokenizer: AutoTokenizer,
    text: str,
    device: str,
    max_prompt_length: int,
) -> torch.Tensor:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    ).input_ids.to(device)


def build_generation_input_ids(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    hint_token_ids: Sequence[int],
    device: str,
    max_prompt_length: int,
) -> torch.Tensor:
    prompt_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
    if not hint_token_ids:
        return prompt_ids

    hint_ids = torch.tensor([list(hint_token_ids)], dtype=prompt_ids.dtype, device=device)
    input_ids = torch.cat([prompt_ids, hint_ids], dim=-1)
    return input_ids[:, -max_prompt_length:]


def generate_with_hints_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    hint_token_ids_batch: Sequence[Sequence[int]],
    max_prompt_length: int,
    max_new_tokens: int,
    device: str,
) -> List[str]:
    prompt_text = build_prompt(tokenizer, question)
    input_id_rows = [
        build_generation_input_ids(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            hint_token_ids=hint_token_ids,
            device=device,
            max_prompt_length=max_prompt_length,
        ).squeeze(0)
        for hint_token_ids in hint_token_ids_batch
    ]
    if not input_id_rows:
        return []

    max_input_len = max(row.size(0) for row in input_id_rows)
    batch_size = len(input_id_rows)
    input_ids = torch.full(
        (batch_size, max_input_len),
        fill_value=tokenizer.pad_token_id,
        dtype=input_id_rows[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long, device=device)

    for row_idx, row in enumerate(input_id_rows):
        row_len = row.size(0)
        input_ids[row_idx, -row_len:] = row
        attention_mask[row_idx, -row_len:] = 1

    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if was_training:
        model.train()

    generated_ids_batch = outputs[:, max_input_len:]
    return [
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for generated_ids in generated_ids_batch
    ]


def generate_with_hint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    question: str,
    hint_token_ids: List[int],
    max_prompt_length: int,
    max_new_tokens: int,
    device: str,
) -> str:
    return generate_with_hints_batch(
        model=model,
        tokenizer=tokenizer,
        question=question,
        hint_token_ids_batch=[hint_token_ids],
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        device=device,
    )[0]


def evaluate_questions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    questions: List[str],
    max_prompt_length: int,
    max_new_tokens: int,
    device: str,
    batch_size: int = 8,
) -> List[str]:
    """Batch-evaluate questions using true GPU batching for speed."""
    predictions: List[str] = []
    step = max(1, batch_size)
    total_batches = (len(questions) + step - 1) // step

    was_training = model.training
    model.eval()

    for start_idx in tqdm(
        range(0, len(questions), step),
        total=total_batches,
        desc="Phase 1 Eval",
        leave=True,
    ):
        batch_questions = questions[start_idx : start_idx + step]
        # Build prompt texts and tokenize each into a 1-D tensor
        prompt_texts = [build_prompt(tokenizer, q) for q in batch_questions]
        encoded = [
            tokenizer(
                pt,
                return_tensors="pt",
                truncation=True,
                max_length=max_prompt_length,
                add_special_tokens=False,
            ).input_ids.squeeze(0)
            for pt in prompt_texts
        ]
        # Left-pad to the same length
        max_len = max(t.size(0) for t in encoded)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full(
            (len(encoded), max_len),
            fill_value=pad_id,
            dtype=encoded[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(encoded), max_len), dtype=torch.long, device=device
        )
        for i, row in enumerate(encoded):
            row_len = row.size(0)
            input_ids[i, -row_len:] = row.to(device)
            attention_mask[i, -row_len:] = 1

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        generated_ids = outputs[:, max_len:]
        for gen in generated_ids:
            predictions.append(
                tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )

    if was_training:
        model.train()

    return predictions


def generate_rollouts_vllm(
    model_path: str,
    tokenizer: "AutoTokenizer",
    questions: List[str],
    hint_token_ids_per_question: List[List[List[int]]],
    max_prompt_length: int,
    max_new_tokens: int,
    gpu_memory_utilization: float = 0.85,
) -> List[List[str]]:
    """Use vLLM to batch-generate rollout continuations for all mistakes.

    For each question[i] there are ``len(hint_token_ids_per_question[i])``
    hint prefixes (one per sampled first-token).  vLLM generates one
    continuation per (question, hint) pair.

    Critically, we pass ``prompt_token_ids`` (a list of token-id integers)
    directly to vLLM instead of decoding the hint tokens to text and
    re-tokenizing.  The decode→re-tokenize roundtrip is NOT lossless for
    BPE/SentencePiece models: a single hint token_id may decode to a string
    that, when appended to the prompt text and re-tokenized, produces a
    different (or split) token sequence.  Using ``prompt_token_ids`` bypasses
    the tokenizer entirely and guarantees that the prefix seen by the model is
    exactly ``[prompt_token_ids..., hint_token_id]`` — identical to the HF
    path ``torch.cat([prompt_ids, hint_ids], dim=-1)``.

    Args:
        model_path:                  Path to the merged (non-LoRA) model.
        tokenizer:                   Tokenizer (used to build prompt token ids).
        questions:                   List of question strings (one per mistake).
        hint_token_ids_per_question: ``questions[i]`` → list of hint-token-id
                                     lists, one per rollout.
        max_prompt_length:           Maximum prompt token length passed to vLLM.
        max_new_tokens:              Maximum new tokens to generate.
        gpu_memory_utilization:      Fraction of GPU memory for vLLM.

    Returns:
        List of length ``len(questions)``.  Element ``i`` is a list of
        ``len(hint_token_ids_per_question[i])`` generated answer strings
        (continuation *after* the hint prefix).
    """
    if not _is_vllm_available():
        raise RuntimeError(
            "vLLM backend requested but unavailable. "
            f"Import error: {_VLLM_IMPORT_ERROR!r}"
        )

    # Build one token-id list per (question, hint) pair.
    # Using prompt_token_ids avoids the decode→re-tokenize roundtrip that
    # would break token-level prefix forcing for BPE/SentencePiece models.
    flat_token_id_lists: List[List[int]] = []
    rollout_counts: List[int] = []
    for question, hint_ids_list in zip(questions, hint_token_ids_per_question):
        prompt_text = build_prompt(tokenizer, question)
        prompt_ids: List[int] = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        ).input_ids
        count = 0
        for hint_token_ids in hint_ids_list:
            # Truncate prompt so that prompt + hint fits within max_prompt_length
            if hint_token_ids:
                allowed = max_prompt_length - len(hint_token_ids)
                prefix_ids = prompt_ids[-allowed:] if allowed < len(prompt_ids) else prompt_ids
                full_ids = prefix_ids + list(hint_token_ids)
            else:
                full_ids = prompt_ids
            flat_token_id_lists.append(full_ids)
            count += 1
        rollout_counts.append(count)

    if not flat_token_id_lists:
        return [[] for _ in questions]

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_prompt_length + max_new_tokens,
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,   # greedy — matches generate_with_hints_batch(do_sample=False)
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    logger.info(f"vLLM rollout generation: {len(flat_token_id_lists)} prompts total")
    # Pass token-id lists via the `inputs` parameter (dict form) which is
    # supported across vLLM versions.  The older `prompt_token_ids` kwarg was
    # removed in newer vLLM releases.
    vllm_inputs = [{"prompt_token_ids": ids} for ids in flat_token_id_lists]
    outputs = llm.generate(
        vllm_inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    flat_texts = [out.outputs[0].text for out in outputs]

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Re-group flat results back into per-question lists.
    results: List[List[str]] = []
    idx = 0
    for count in rollout_counts:
        results.append(flat_texts[idx : idx + count])
        idx += count
    return results


def evaluate_questions_vllm(
    model_path: str,
    tokenizer: AutoTokenizer,
    questions: List[str],
    max_prompt_length: int,
    max_new_tokens: int,
    batch_size: int = 8,
    gpu_memory_utilization: float = 0.9,
) -> List[str]:
    if not _is_vllm_available():
        raise RuntimeError(
            "vLLM backend requested but unavailable. "
            f"Import error: {_VLLM_IMPORT_ERROR!r}"
        )

    prompts = [build_prompt(tokenizer, question) for question in questions]
    stop_token_ids = [tokenizer.eos_token_id]

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_prompt_length,
        enforce_eager=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
    )

    predictions: List[str] = []
    step = max(1, batch_size)
    total_batches = (len(prompts) + step - 1) // step
    for start_idx in tqdm(
        range(0, len(prompts), step),
        total=total_batches,
        desc="Phase 1 Eval vLLM",
        leave=False,
    ):
        batch_prompts = prompts[start_idx : start_idx + max(1, batch_size)]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        predictions.extend(output.outputs[0].text for output in outputs)

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def build_first_token_target_logprobs(
    student_first_logits: torch.Tensor,
    sampled_token_ids: List[int],
    correct_token_ids: List[int],
    alpha: float,
    delta: float,
) -> torch.Tensor:
    """Build the KL target distribution by multiplicatively scaling logits.

    Correct tokens:  logit *= (1 + alpha)   — boost by alpha fraction
    Wrong   tokens:  logit *= (1 - delta)   — suppress by delta fraction

    Multiplicative scaling is proportional to the logit magnitude, so the
    KL signal is meaningful even when the model is very confident (large
    logit gap).  With additive ±const the adjustment is negligible compared
    to logit differences of 15-25 typical in confident LLMs.

    Example (delta=0.1, logit=20):  20 * 0.9 = 18  →  prob drops ~85%→~50%
    Example (delta=0.9, logit=20):  20 * 0.1 =  2  →  prob drops ~85%→~1%
    """
    adjusted_logits = student_first_logits.detach().clone().float()
    correct_token_set = set(correct_token_ids)

    for token_id in sampled_token_ids:
        if token_id in correct_token_set:
            adjusted_logits[0, token_id] *= (1.0 + alpha)
        else:
            adjusted_logits[0, token_id] *= (1.0 - delta)

    return F.log_softmax(adjusted_logits, dim=-1)


def build_rollout_weights(
    sampled_token_ids: List[int],
    correct_token_ids: List[int],
    alpha: float,
    delta: float,
    device: str,
) -> torch.Tensor:
    weights = []
    correct_token_set = set(correct_token_ids)

    for token_id in sampled_token_ids:
        if token_id in correct_token_set:
            weights.append(max(0.0, 1.0 + alpha))
        else:
            weights.append(max(0.0, 1.0 - delta))

    if not weights:
        return torch.tensor([], dtype=torch.float32, device=device)

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    total = weight_tensor.sum()
    if total <= 0:
        return torch.full_like(weight_tensor, 1.0 / len(weights))
    return weight_tensor / total


def compute_rest_trajectory_kl(
    prompt_text: str,
    hint_token_ids: List[int],
    generated_answer: str,
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    max_prompt_length: int,
) -> torch.Tensor:
    if not generated_answer.strip():
        return torch.tensor(0.0, device=device)

    prompt_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
    if hint_token_ids:
        hint_ids = torch.tensor([hint_token_ids], dtype=prompt_ids.dtype, device=device)
        prefix_ids = torch.cat([prompt_ids, hint_ids], dim=-1)
        prefix_ids = prefix_ids[:, -max_prompt_length:]
    else:
        prefix_ids = prompt_ids
    answer_ids = tokenizer(generated_answer, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    if answer_ids.numel() == 0:
        return torch.tensor(0.0, device=device)

    full_ids = torch.cat([prefix_ids, answer_ids], dim=-1)

    with torch.no_grad():
        teacher_outputs = teacher_model(full_ids)
        teacher_logprobs = F.log_softmax(teacher_outputs.logits[:, :-1, :], dim=-1)

    student_outputs = student_model(full_ids)
    student_logprobs = F.log_softmax(student_outputs.logits[:, :-1, :], dim=-1)

    prefix_len = prefix_ids.size(-1)
    answer_len = answer_ids.size(-1)
    start = max(prefix_len - 1, 0)
    end = start + answer_len

    teacher_slice = teacher_logprobs[:, start:end, :]
    student_slice = student_logprobs[:, start:end, :]

    return F.kl_div(
        student_slice.reshape(-1, student_slice.size(-1)),
        teacher_slice.reshape(-1, teacher_slice.size(-1)),
        reduction="batchmean",
        log_target=True,
    )


def compute_rest_trajectory_kl_batch(
    prompt_text: str,
    hint_token_ids_list: List[List[int]],
    generated_answers: List[str],
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    max_prompt_length: int,
) -> List[torch.Tensor]:
    """Batch version of compute_rest_trajectory_kl.

    Runs teacher and student forward passes once per non-empty rollout batch
    instead of once per rollout, reducing GPU kernel launches by n_roll×.
    The KL is computed per-rollout (same semantics as the scalar version) and
    returned as a list so the caller can stack / weight them identically.
    """
    results: List[torch.Tensor] = []

    # Separate empty answers (return 0 immediately) from non-empty ones
    non_empty_indices = []
    for i, ans in enumerate(generated_answers):
        if not ans.strip():
            results.append(torch.tensor(0.0, device=device))
        else:
            non_empty_indices.append(i)
            results.append(None)  # placeholder

    if not non_empty_indices:
        return results

    prompt_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)

    # Build full_ids for each non-empty rollout
    full_id_rows: List[torch.Tensor] = []
    prefix_lens: List[int] = []
    answer_lens: List[int] = []

    for i in non_empty_indices:
        hint_token_ids = hint_token_ids_list[i]
        generated_answer = generated_answers[i]

        if hint_token_ids:
            hint_ids = torch.tensor([hint_token_ids], dtype=prompt_ids.dtype, device=device)
            prefix_ids = torch.cat([prompt_ids, hint_ids], dim=-1)
            prefix_ids = prefix_ids[:, -max_prompt_length:]
        else:
            prefix_ids = prompt_ids

        answer_ids = tokenizer(
            generated_answer, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        if answer_ids.numel() == 0:
            # Treat as empty — overwrite placeholder with 0
            results[i] = torch.tensor(0.0, device=device)
            full_id_rows.append(None)
            prefix_lens.append(0)
            answer_lens.append(0)
            continue

        full_ids = torch.cat([prefix_ids, answer_ids], dim=-1)
        full_id_rows.append(full_ids.squeeze(0))
        prefix_lens.append(prefix_ids.size(-1))
        answer_lens.append(answer_ids.size(-1))

    # Filter out the None placeholders (zero-answer cases handled above)
    valid = [
        (idx, row, pl, al)
        for idx, row, pl, al in zip(non_empty_indices, full_id_rows, prefix_lens, answer_lens)
        if row is not None
    ]
    if not valid:
        return results

    valid_indices, rows, p_lens, a_lens = zip(*valid)

    # Pad rows to the same length for a single batch forward (left-pad)
    max_seq_len = max(r.size(0) for r in rows)
    batch_size = len(rows)
    pad_id = tokenizer.pad_token_id
    batch_ids = torch.full(
        (batch_size, max_seq_len), fill_value=pad_id, dtype=rows[0].dtype, device=device
    )
    batch_attn_mask = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.long, device=device
    )
    for b, row in enumerate(rows):
        row_len = row.size(0)
        batch_ids[b, -row_len:] = row
        batch_attn_mask[b, -row_len:] = 1

    # Single teacher forward (no grad)
    with torch.no_grad():
        teacher_logprobs = F.log_softmax(
            teacher_model(batch_ids, attention_mask=batch_attn_mask).logits[:, :-1, :], dim=-1
        )

    # Single student forward (with grad for backprop)
    student_logprobs = F.log_softmax(
        student_model(batch_ids, attention_mask=batch_attn_mask).logits[:, :-1, :], dim=-1
    )

    # Compute per-rollout KL from the shared batch tensors
    for b, (orig_idx, pl, al) in enumerate(zip(valid_indices, p_lens, a_lens)):
        # Adjust for left-padding offset
        pad_offset = max_seq_len - rows[b].size(0)
        start = pad_offset + max(pl - 1, 0)
        end = start + al

        t_slice = teacher_logprobs[b, start:end, :]
        s_slice = student_logprobs[b, start:end, :]

        if t_slice.numel() == 0:
            results[orig_idx] = torch.tensor(0.0, device=device)
        else:
            results[orig_idx] = F.kl_div(
                s_slice.reshape(-1, s_slice.size(-1)),
                t_slice.reshape(-1, t_slice.size(-1)),
                reduction="batchmean",
                log_target=True,
            )

    return results


def train_a_token_sd(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    n_roll: int = 8,
    alpha: float = 0.0,
    delta: float = 0.1,
    w_tail: float = 1.0,
    ema_decay: float = 0.99,
    max_prompt_length: int = 3072,
    max_new_tokens: int = 2048,
    inference_batch_size: int = 4,
    eval_backend: str = "transformers",
    vllm_gpu_memory_utilization: float = 0.3,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    gradient_accumulation_steps: int = 4,
    use_ema: bool = False,
    use_rest_kl: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if not isinstance(raw_data, list):
        raise ValueError("Training data must be a list of JSON objects")

    logger.info(f"Loaded {len(raw_data)} samples from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
    student_model, teacher_model = _build_models(
        model_path,
        torch_dtype,
        device,
        use_lora,
        lora_r,
        lora_alpha,
        lora_dropout,
    )

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)

    questions = [normalize_question_text(item.get("question", item.get("prompt", ""))) for item in raw_data]
    answers = [
        normalize_reference_answer(item.get("answer", item.get("ref_answer", "")))
        for item in raw_data
    ]

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="A-Token-SD Epochs")
    for epoch in epoch_progress:
        epoch_progress.set_postfix({"samples": len(raw_data)})
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        logger.info("Phase 1: Base test (student_model)")

        if eval_backend == "vllm":
            if device == "cpu":
                raise ValueError("vLLM backend requires CUDA; please use --eval_backend transformers on CPU.")
            # For both LoRA and non-LoRA we save a merged/full model to a temp
            # directory, run vLLM inference, then delete the temp directory.
            # When use_lora=True we deep-copy the PeftModel, merge the adapter
            # into the copy, and save it — the live student_model is untouched.
            _vllm_tmp = tempfile.mkdtemp(prefix="a_token_sd_vllm_")
            try:
                if use_lora:
                    # Offload student to CPU while vLLM occupies the GPU so
                    # that the merged copy + vLLM engine fit in VRAM together.
                    student_model.cpu()
                    teacher_model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _save_merged_lora_for_vllm(student_model, tokenizer, _vllm_tmp)
                else:
                    logger.info(f"Saving student snapshot for vLLM eval to {_vllm_tmp}")
                    student_model.cpu()
                    teacher_model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    student_model.save_pretrained(_vllm_tmp)
                    tokenizer.save_pretrained(_vllm_tmp)

                base_predictions = evaluate_questions_vllm(
                    model_path=_vllm_tmp,
                    tokenizer=tokenizer,
                    questions=questions,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=inference_batch_size,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                )
            finally:
                shutil.rmtree(_vllm_tmp, ignore_errors=True)
                # Restore models to GPU for the training phases (both LoRA and non-LoRA paths offload above).
                student_model.to(device)
                teacher_model.to(device)
        else:
            base_predictions = evaluate_questions(
                student_model,
                tokenizer,
                questions,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                device=device,
                batch_size=inference_batch_size,
            )

        mistakes = []
        for idx, prediction in enumerate(base_predictions):
            if not check_correctness(prediction, answers[idx]):
                mistakes.append(raw_data[idx])

        mistake_count = len(mistakes)
        logger.info(f"Epoch {epoch}: mistake count = {mistake_count}/{len(raw_data)}")

        if mistake_count == 0:
            epoch_progress.set_postfix({"samples": len(raw_data), "mistakes": 0})
            continue

        student_model.train()
        epoch_loss = 0.0
        epoch_first_kl = 0.0
        epoch_rest_kl = 0.0
        grad_accum_steps = max(1, gradient_accumulation_steps)
        optimizer.zero_grad()

        # ── Phase A: sample first tokens for every mistake (pure forward, no generate) ──
        # We only need sampled_token_ids here; the actual gradient-carrying forward
        # is done again in Phase C so that first_token_kl can backprop.
        mistake_questions:     List[str]        = []
        mistake_ref_answers:   List[str]        = []
        mistake_prompt_texts:  List[str]        = []
        all_sampled_token_ids: List[List[int]]  = []

        student_model.eval()
        with torch.no_grad():
            for mistake in tqdm(mistakes, desc=f"Epoch {epoch} Phase-A (first-token)", leave=True):
                q   = normalize_question_text(mistake.get("question", mistake.get("prompt", "")))
                ref = normalize_reference_answer(mistake.get("answer", mistake.get("ref_answer", "")))
                pt  = build_prompt(tokenizer, q)
                ids = tokenize_prompt(tokenizer, pt, device, max_prompt_length)
                attn = torch.ones_like(ids, device=device)
                out  = student_model(input_ids=ids, attention_mask=attn)
                logits = out.logits[:, -1, :]
                sampled = sample_first_tokens(logits[0], n_roll=n_roll, temperature=0.7, top_k=50)
                if not sampled:
                    continue
                mistake_questions.append(q)
                mistake_ref_answers.append(ref)
                mistake_prompt_texts.append(pt)
                all_sampled_token_ids.append(sampled)
        student_model.train()

        # ── Phase B: batch-generate rollout continuations ──
        all_hint_ids: List[List[List[int]]] = [
            [[tid] for tid in sampled] for sampled in all_sampled_token_ids
        ]

        if eval_backend == "vllm" and _is_vllm_available():
            logger.info("Phase 2/3: generating rollouts with vLLM …")
            _roll_tmp = tempfile.mkdtemp(prefix="a_token_sd_roll_")
            try:
                student_model.cpu()
                teacher_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if use_lora:
                    _save_merged_lora_for_vllm(student_model, tokenizer, _roll_tmp)
                else:
                    student_model.save_pretrained(_roll_tmp)
                    tokenizer.save_pretrained(_roll_tmp)
                    # non-LoRA path: student/teacher already on CPU from the lines above
                all_generated_answers = generate_rollouts_vllm(
                    model_path=_roll_tmp,
                    tokenizer=tokenizer,
                    questions=mistake_questions,
                    hint_token_ids_per_question=all_hint_ids,
                    max_prompt_length=max_prompt_length + n_roll,
                    max_new_tokens=max_new_tokens,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                )
            finally:
                shutil.rmtree(_roll_tmp, ignore_errors=True)
                # vLLM runs in a subprocess; after it exits the GPU memory is
                # released by the OS, but the CUDA allocator cache may still
                # hold pages.  Calling empty_cache() here gives PyTorch a
                # chance to reclaim that memory before we move the models back.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                student_model.to(device)
                teacher_model.to(device)
        else:
            all_generated_answers = []
            for q, hint_ids_list in tqdm(
                zip(mistake_questions, all_hint_ids),
                total=len(mistake_questions),
                desc=f"Epoch {epoch} Phase-B (rollouts)",
                leave=True,
            ):
                answers = generate_with_hints_batch(
                    student_model, tokenizer, q, hint_ids_list,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                all_generated_answers.append(answers)

        # ── Phase C: KL loss backward ──
        active_mistake_count = len(mistake_questions)
        mistake_progress = tqdm(
            range(active_mistake_count),
            total=active_mistake_count,
            desc=f"Epoch {epoch} Train",
            leave=True,
        )
        for step in mistake_progress:
            question          = mistake_questions[step]
            reference_answer  = mistake_ref_answers[step]
            prompt_text       = mistake_prompt_texts[step]
            sampled_token_ids = all_sampled_token_ids[step]
            generated_answers = all_generated_answers[step]
            rollout_hint_token_ids = all_hint_ids[step]

            # Re-run student forward WITH gradient so first_token_kl can backprop.
            # Phase A used no_grad; we need a fresh forward here for the loss graph.
            _ids  = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
            _attn = torch.ones_like(_ids, device=device)
            _out  = student_model(input_ids=_ids, attention_mask=_attn)
            student_first_logits   = _out.logits[:, -1, :]          # [1, vocab], has grad
            student_first_logprobs = F.log_softmax(student_first_logits, dim=-1)

            correct_token_ids: List[int] = []

            # Phase 4: correctness check (determines correct_token_ids)
            for token_id, hint_token_ids, generated_answer in zip(
                sampled_token_ids, rollout_hint_token_ids, generated_answers
            ):
                forced_prefix = tokenizer.decode(
                    hint_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                full_generated_answer = forced_prefix + generated_answer

                if check_correctness(full_generated_answer, reference_answer):
                    correct_token_ids.append(token_id)

            # Phase 5 (tail KL): only computed when use_rest_kl=True.
            # When disabled, rest_kl acts as a pure zero — no teacher/student
            # forward for the rollout sequences, saving the bulk of compute.
            if use_rest_kl:
                rest_kls = compute_rest_trajectory_kl_batch(
                    prompt_text,
                    rollout_hint_token_ids,
                    generated_answers,
                    teacher_model,
                    student_model,
                    tokenizer,
                    device,
                    max_prompt_length,
                )
            else:
                rest_kls = []

            first_token_target_logprobs = build_first_token_target_logprobs(
                student_first_logits,
                sampled_token_ids,
                correct_token_ids,
                alpha=alpha,
                delta=delta,
            )
            rollout_weights = build_rollout_weights(
                sampled_token_ids,
                correct_token_ids,
                alpha=alpha,
                delta=delta,
                device=device,
            )
            first_token_kl = F.kl_div(
                student_first_logprobs,
                first_token_target_logprobs,
                reduction="batchmean",
                log_target=True,
            )

            if use_rest_kl and rest_kls:
                rest_kl_tensor = torch.stack(rest_kls)
                avg_rest_kl = torch.sum(rollout_weights * rest_kl_tensor)
            else:
                avg_rest_kl = torch.tensor(0.0, device=device)
            step_loss = (first_token_kl + w_tail * avg_rest_kl) / grad_accum_steps

            step_loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == active_mistake_count:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += step_loss.item()
            epoch_first_kl += first_token_kl.item()
            epoch_rest_kl += avg_rest_kl.item()
            mistake_progress.set_postfix(
                {
                    "correct_rollouts": f"{len(correct_token_ids)}/{len(sampled_token_ids)}",
                    "loss": f"{step_loss.item():.4f}",
                }
            )

            # ── diagnostic: show how alpha/delta shifted the first-token probs ──
            with torch.no_grad():
                orig_probs = student_first_logprobs.exp()
                tgt_probs  = first_token_target_logprobs.exp()
                _corr_orig = sum(orig_probs[0, tid].item() for tid in correct_token_ids) if correct_token_ids else 0.0
                _corr_tgt  = sum(tgt_probs[0, tid].item()  for tid in correct_token_ids) if correct_token_ids else 0.0
                _err_orig  = sum(orig_probs[0, tid].item() for tid in sampled_token_ids if tid not in set(correct_token_ids))
                _err_tgt   = sum(tgt_probs[0, tid].item()  for tid in sampled_token_ids if tid not in set(correct_token_ids))
            logger.info(
                f"Epoch {epoch} Step {step}: correct_rollouts={len(correct_token_ids)}/{len(sampled_token_ids)}, "
                f"first_token_kl={first_token_kl.item():.6f}, rest_kl={avg_rest_kl.item():.6f}, "
                f"step_loss={step_loss.item():.6f} | "
                f"corr_prob {_corr_orig:.4f}→{_corr_tgt:.4f}  "
                f"err_prob {_err_orig:.4f}→{_err_tgt:.4f}"
            )

        if use_ema:
            update_ema(teacher_model, student_model, decay=ema_decay)
        else:
            logger.debug("EMA update skipped (use_ema=False)")

        avg_first_kl = epoch_first_kl / max(1, active_mistake_count)
        avg_rest_kl  = epoch_rest_kl  / max(1, active_mistake_count)
        avg_loss     = epoch_loss     / max(1, active_mistake_count)
        epoch_progress.set_postfix(
            {
                "samples":  len(raw_data),
                "mistakes": mistake_count,
                "active":   active_mistake_count,
                "avg_loss": f"{avg_loss:.4f}",
            }
        )
        logger.info(
            f"Epoch {epoch} summary: mistake_count={mistake_count} (active={active_mistake_count}), "
            f"avg_first_token_kl={avg_first_kl:.6f}, avg_rest_kl={avg_rest_kl:.6f}, avg_loss={avg_loss:.6f}"
        )

    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")


def _build_a_token_sd_output_dir(epoch: int) -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        project_root,
        "outputs",
        f"a_token_sd_{epoch}ep_{datetime.now().strftime('%m%d_%H%M')}",
    )


def train_a_token_sd_api(
    questions,
    answers,
    epoch,
    output_dir=None,
    model_path_override=None,
    use_lora=True,
    learning_rate=5e-5,
    n_roll=8,
    alpha=0.0,
    delta=0.1,
    w_tail=1.0,
    ema_decay=0.99,
    max_prompt_length=3072,
    max_new_tokens=2048,
    inference_batch_size=4,
    eval_backend="vllm",
    vllm_gpu_memory_utilization=0.85,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    gradient_accumulation_steps=4,
    use_ema=False,
    use_rest_kl=False,
    device=None,
):
    """External wrapper for batch A-Token-SD training.

    Args:
        questions: List of question strings.
        answers: List of reference final answer texts.
        epoch: Number of training epochs.
        output_dir: Directory to save the trained model or LoRA adapter.
        model_path_override: Path to the base model. Required.
    """
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise TypeError("questions and answers must both be lists")
    if len(questions) != len(answers):
        raise ValueError(
            f"questions and answers must have the same length, got {len(questions)} and {len(answers)}"
        )
    if model_path_override is None:
        raise ValueError("model_path_override must be provided")

    resolved_model_path = model_path_override
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_output_dir = output_dir or _build_a_token_sd_output_dir(epoch)

    train_samples = [
        {
            "question": "" if question is None else str(question).strip(),
            "answer": "" if answer is None else str(answer).strip(),
        }
        for question, answer in zip(questions, answers)
    ]

    os.makedirs(resolved_output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="a_token_sd_",
        dir=resolved_output_dir,
        delete=False,
    ) as temp_file:
        json.dump(train_samples, temp_file, ensure_ascii=False, indent=2)
        temp_data_path = temp_file.name

    logger.info(
        "Starting A-Token-SD API training with samples=%s, epochs=%s, use_lora=%s, output_dir=%s",
        len(train_samples),
        epoch,
        use_lora,
        resolved_output_dir,
    )

    train_a_token_sd(
        model_path=resolved_model_path,
        data_path=temp_data_path,
        output_dir=resolved_output_dir,
        num_epochs=epoch,
        learning_rate=learning_rate,
        n_roll=n_roll,
        alpha=alpha,
        delta=delta,
        w_tail=w_tail,
        ema_decay=ema_decay,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        inference_batch_size=inference_batch_size,
        eval_backend=eval_backend,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_ema=use_ema,
        use_rest_kl=use_rest_kl,
        device=resolved_device,
    )

    return {
        "sample_count": len(train_samples),
        "epochs": epoch,
        "use_lora": use_lora,
        "model_path": resolved_model_path,
        "data_path": temp_data_path,
        "output_dir": resolved_output_dir,
        "device": resolved_device,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tail Token Exploration Training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--n_roll", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--w_tail", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--inference_batch_size", type=int, default=4)
    parser.add_argument("--eval_backend", type=str, choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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
        w_tail=args.w_tail,
        ema_decay=args.ema_decay,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        inference_batch_size=args.inference_batch_size,
        eval_backend=args.eval_backend,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_ema=args.use_ema,
        device=args.device,
    )

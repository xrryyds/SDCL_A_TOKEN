import json
import logging
import os
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import-time environment specific
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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


def sample_first_tokens(logits: torch.Tensor, n_roll: int, temperature: float = 0.7, top_k: int = 50) -> List[int]:
    if temperature <= 0:
        top_indices = torch.topk(logits, k=min(n_roll, logits.size(-1)), dim=-1).indices
        return top_indices.tolist()

    distribution = logits.clone().float()
    if top_k is not None and top_k > 0:
        topk_values, topk_indices = torch.topk(distribution, top_k, dim=-1)
        mask = torch.full_like(distribution, float("-inf"))
        mask.scatter_(-1, topk_indices, topk_values)
        distribution = mask

    distribution = distribution / temperature
    probs = F.softmax(distribution, dim=-1)
    sampled = torch.multinomial(probs, num_samples=n_roll, replacement=True)
    return sampled.unique().tolist()


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
    predictions: List[str] = []
    for start_idx in range(0, len(questions), max(1, batch_size)):
        batch_questions = questions[start_idx : start_idx + max(1, batch_size)]
        for question in batch_questions:
            predictions.extend(
                generate_with_hints_batch(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    hint_token_ids_batch=[[]],
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            )
    return predictions


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
    for start_idx in range(0, len(prompts), max(1, batch_size)):
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
    adjusted_logits = student_first_logits.detach().clone()
    correct_token_set = set(correct_token_ids)

    for token_id in sampled_token_ids:
        if token_id in correct_token_set:
            adjusted_logits[0, token_id] += alpha
        else:
            adjusted_logits[0, token_id] -= delta

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

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        logger.info("Phase 1: Base test (student_model)")

        if eval_backend == "vllm":
            if device == "cpu":
                raise ValueError("vLLM backend requires CUDA; please use --eval_backend transformers on CPU.")
            if use_lora:
                logger.warning(
                    "LoRA mode with eval_backend=vllm is not supported in-place yet; "
                    "falling back to transformers backend for Phase 1 this epoch."
                )
                base_predictions = evaluate_questions(
                    student_model,
                    tokenizer,
                    questions,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    batch_size=inference_batch_size,
                )
            else:
                snapshot_dir = os.path.join(output_dir, "_vllm_eval_snapshot")
                os.makedirs(snapshot_dir, exist_ok=True)
                logger.info(f"Saving student snapshot for vLLM eval to {snapshot_dir}")
                student_model.save_pretrained(snapshot_dir)
                tokenizer.save_pretrained(snapshot_dir)
                base_predictions = evaluate_questions_vllm(
                    model_path=snapshot_dir,
                    tokenizer=tokenizer,
                    questions=questions,
                    max_prompt_length=max_prompt_length,
                    max_new_tokens=max_new_tokens,
                    batch_size=inference_batch_size,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                )
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
            continue

        student_model.train()
        epoch_loss = 0.0
        epoch_first_kl = 0.0
        epoch_rest_kl = 0.0

        for step, mistake in enumerate(mistakes):
            question = normalize_question_text(mistake.get("question", mistake.get("prompt", "")))
            reference_answer = normalize_reference_answer(
                mistake.get("answer", mistake.get("ref_answer", ""))
            )

            prompt_text = build_prompt(tokenizer, question)
            input_ids = tokenize_prompt(tokenizer, prompt_text, device, max_prompt_length)
            attention_mask = torch.ones_like(input_ids, device=device)

            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_first_logits = student_outputs.logits[:, -1, :]
            student_first_logprobs = F.log_softmax(student_first_logits, dim=-1)

            sampled_token_ids = sample_first_tokens(
                student_first_logits[0], n_roll=n_roll, temperature=0.7, top_k=50
            )
            if not sampled_token_ids:
                continue

            correct_token_ids: List[int] = []
            rest_kls: List[torch.Tensor] = []
            rollout_hint_token_ids = [[token_id] for token_id in sampled_token_ids]
            generated_answers = generate_with_hints_batch(
                student_model,
                tokenizer,
                question,
                rollout_hint_token_ids,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                device=device,
            )

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

                rest_kls.append(
                    compute_rest_trajectory_kl(
                        prompt_text,
                        hint_token_ids,
                        generated_answer,
                        teacher_model,
                        student_model,
                        tokenizer,
                        device,
                        max_prompt_length,
                    )
                )

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

            if rest_kls:
                rest_kl_tensor = torch.stack(rest_kls)
                avg_rest_kl = torch.sum(rollout_weights * rest_kl_tensor)
            else:
                avg_rest_kl = torch.tensor(0.0, device=device)
            step_loss = first_token_kl + w_tail * avg_rest_kl

            optimizer.zero_grad()
            step_loss.backward()
            optimizer.step()

            epoch_loss += step_loss.item()
            epoch_first_kl += first_token_kl.item()
            epoch_rest_kl += avg_rest_kl.item()

            logger.info(
                f"Epoch {epoch} Step {step}: correct_rollouts={len(correct_token_ids)}/{len(sampled_token_ids)}, "
                f"first_token_kl = {first_token_kl.item():.6f}, "
                f"rest_kl = {avg_rest_kl.item():.6f}, "
                f"step_loss = {step_loss.item():.6f}"
            )

        update_ema(teacher_model, student_model, decay=ema_decay)

        avg_first_kl = epoch_first_kl / max(1, mistake_count)
        avg_rest_kl = epoch_rest_kl / max(1, mistake_count)
        avg_loss = epoch_loss / max(1, mistake_count)
        logger.info(
            f"Epoch {epoch} summary: mistake_count={mistake_count}, "
            f"avg_first_token_kl={avg_first_kl:.6f}, avg_rest_kl={avg_rest_kl:.6f}, avg_loss={avg_loss:.6f}"
        )

    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training finished and saved to {output_dir}")


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
    parser.add_argument("--delta", type=float, default=0.1)
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
        device=args.device,
    )

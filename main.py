import os
import json
import gc
import random
import tempfile
from datetime import datetime
import torch
import numpy as np
import logging
from tqdm import tqdm
from scripts.train.extract_first_tokens import extract_and_save_first_tokens
from scripts.train.a_token_sdcl import (
    generate_fill_correct,
    merge_to_train_data,
)
from scripts.train.a_token_sdcl_train import train_a_token_sdcl
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel


from scripts import TakeExam, TeacherCorrecter
from utils import (
    FileIOUtils,
    remove_null_hints,
    filter_json_by_question_idx,
    generate_irdcl_dataset,
    generate_irdcl_datase_v2,
    remove_null_hints,
    merge_lora_to_base_model,
    generate_sft_data,
)
from data_math import (
    Math_500,
    GSM8K,
    AIME,
    Math_All,
    Math_Subset,
    LiveMathBench,
    AIME_1983_2024,
    DeepMath_103K,
)


# =====================================================
# Logger Setup
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# Global Config
# =====================================================
exam_paper = FileIOUtils()
_tokenizer_cache = None


def _get_tokenizer():
    """Lazy-load tokenizer for hint truncation."""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    return _tokenizer_cache


def truncate_hints_by_tokens(hints_list: list, max_tokens: int) -> list:
    """Truncate each hint string in hints_list to at most max_tokens tokens.

    Args:
        hints_list: List of hint strings.
        max_tokens: Maximum number of tokens to keep for each hint.

    Returns:
        List of truncated hint strings.
    """
    if max_tokens is None:
        return hints_list
    tokenizer = _get_tokenizer()
    truncated = []
    for hint in hints_list:
        if hint is None or hint == "":
            truncated.append(hint)
            continue
        token_ids = tokenizer.encode(hint, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            hint = tokenizer.decode(token_ids, skip_special_tokens=True)
        truncated.append(hint)
    return truncated


model_path = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"


def exam_roll_recheck_hints(
    lora_path: str = None, max_token: int = 4096, hint_token_limit: int = None
):
    try:
        logger.info("Step 1: Loading Dataset...")
        with open(exam_paper.disadv_hints_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        (
            question_idx,
            question,
            question_with_hint,
            ref_solution,
            ref_answer,
            _,
            hints,
            from_entropy,
        ) = exam_paper.parse_hints_exam(data)

        # Truncate hints if hint_token_limit is specified
        if hint_token_limit is not None:
            logger.info(f"Truncating hints to {hint_token_limit} tokens...")
            hints = truncate_hints_by_tokens(hints, hint_token_limit)

        # Key: question_idx, Value: {hints, entropy}
        meta_map = {}
        for q_id, h, ent in zip(question_idx, hints, from_entropy):
            meta_map[q_id] = {"hints": h, "orig_ent": ent}

        logger.info("Step 2: Student Rolling Exam...")
        if lora_path:
            student_exam = TakeExam(
                model_path=model_path,
                use_lora=True,
                adapter_path=lora_path,
                max_seq_length=max_token,
            )
        else:
            student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
        student_exam.exam_roll_k_with_hints(
            question=question,
            solution=ref_solution,
            answer=ref_answer,
            question_idx=question_idx,
            hints=hints,
        )

        logger.info("Step 3: Teacher Grading...")
        teacher = TeacherCorrecter()
        _, correct_data = teacher.teacher_mark_paper(True)

        c_ids, c_qs, c_ans, c_sols, c_refs, c_ents = correct_data

        best_candidates = {}

        for i in range(len(c_ids)):
            qid = c_ids[i]
            curr_ans = c_ans[i]

            item = {
                "question_idx": qid,
                "question": c_qs[i],
                "hints": meta_map.get(qid, {}).get("hints", []),
                "student_answer": curr_ans,
                "ref_solution": c_sols[i],
                "ref_answer": c_refs[i],
                "entropy_original": meta_map.get(qid, {}).get("orig_ent", 0.0),
                "entropy_with_hints": c_ents[i],
                "success": True,
            }

            if qid not in best_candidates:
                best_candidates[qid] = item
            else:
                prev_len = len(best_candidates[qid]["student_answer"])
                curr_len = len(curr_ans)
                if curr_len < prev_len:
                    best_candidates[qid] = item

        new_data_to_append = list(best_candidates.values())
        logger.info(
            f"Filtered {len(c_ids)} correct samples down to {len(new_data_to_append)} unique items (shortest answer strategy)."
        )

        target_path = exam_paper.adv_hints_dataset_path
        existing_data = []

        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        logger.warning(
                            f"Existing file {target_path} is not a list. Overwriting."
                        )
                        existing_data = []
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode {target_path}. Starting with empty list."
                )
                existing_data = []

        final_data = existing_data + new_data_to_append

        exam_paper.save_results_to_json(final_data, exam_paper.adv_hints_dataset_path)

        logger.info(
            f"Successfully appended {len(new_data_to_append)} items to {target_path}. Total items: {len(final_data)}"
        )

        return {"success": True, "processed_count": len(new_data_to_append)}

    except FileNotFoundError as e:
        error_msg = f"file not found: {e.filename}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"unknown error: {str(e)}"
        logger.error(error_msg)
        import traceback

        traceback.print_exc()
        return {"success": False, "error": error_msg}


def process_exam_file_batch(file_path, lora_path: str = None, max_token: int = 4096):
    """
    JSON student_exam.exam
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item.get("question", "") for item in data]

        solutions = [item.get("ref_solution", "") for item in data]

        answers = [item.get("ref_answer", "") for item in data]

        indices = [item.get("question_idx", 0) for item in data]

        student_exam = None

        if lora_path:
            student_exam = TakeExam(
                model_path=model_path,
                use_lora=True,
                adapter_path=lora_path,
                max_seq_length=max_token,
            )
        else:
            student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
        student_exam.exam(
            question=questions, solution=solutions, answer=answers, question_idx=indices
        )

        print(f" {len(data)} ")

    except FileNotFoundError:
        print(f" {file_path}")
    except json.JSONDecodeError:
        print("JSON ")
    except Exception as e:
        print(f"{e}")


def student_correct(
    lora_path: str = None, max_token: int = 4096, hint_token_limit: int = None
):
    logger.info("Step 1: Loading Dataset...")
    exam_paper.load_question_with_hints()
    (
        question_idx,
        question,
        question_with_hint,
        ref_solution,
        ref_answer,
        _,
        hints,
        from_entropy,
    ) = exam_paper.parse_hints_exam(exam_paper.question_with_hints)

    # Truncate hints if hint_token_limit is specified
    if hint_token_limit is not None:
        logger.info(f"Truncating hints to {hint_token_limit} tokens...")
        hints = truncate_hints_by_tokens(hints, hint_token_limit)

    logger.info("Step 2: Student Taking Exam...")
    if lora_path:
        student_exam = TakeExam(
            model_path=model_path,
            use_lora=True,
            adapter_path=lora_path,
            max_seq_length=max_token,
        )
    else:
        student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    student_exam.exam_with_hints(
        question=question,
        solution=ref_solution,
        answer=ref_answer,
        question_idx=question_idx,
        hints=hints,
    )

    logger.info("Step 3: Teacher Grading...")
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()

    err_question_idx, _, err_answers, _, _, err_entropies = incorrect_data
    correct_question_idx, _, correct_answers, _, _, correct_entropies = correct_data

    results_map = {}

    for q_id, s_ans, s_ent in zip(err_question_idx, err_answers, err_entropies):
        results_map[str(q_id)] = {"answer": s_ans, "entropy": s_ent}

    for q_id, s_ans, s_ent in zip(
        correct_question_idx, correct_answers, correct_entropies
    ):
        results_map[str(q_id)] = {"answer": s_ans, "entropy": s_ent}

    err_ids_set = set(str(x) for x in err_question_idx)
    processed_ids_set = set(results_map.keys())

    correct_group = []
    incorrect_group = []

    total_data = zip(
        question_idx, question, hints, ref_solution, ref_answer, from_entropy
    )

    for q_id, q, q_hints, r_sol, r_ans, orig_ent in total_data:
        str_qid = str(q_id)

        if str_qid not in processed_ids_set:
            logger.warning(f"Question ID {q_id} missing from exam results. Skipping.")
            continue

        exam_res = results_map[str_qid]
        s_ans = exam_res["answer"]
        hint_ent = exam_res["entropy"]

        item = {
            "question_idx": q_id,
            "question": q,
            "hints": q_hints,
            "student_answer": s_ans,
            "ref_solution": r_sol,
            "ref_answer": r_ans,
            "entropy_original": orig_ent,
            "entropy_with_hints": hint_ent,
        }

        if str_qid in err_ids_set:
            incorrect_group.append(item)
        else:
            correct_group.append(item)

    logger.info(
        f"Classification Done. Correct: {len(correct_group)}, Incorrect: {len(incorrect_group)}"
    )

    data_for_teacher_grpo = []
    for item in correct_group:
        data_for_teacher_grpo.append({**item, "success": True})
    for item in incorrect_group:
        data_for_teacher_grpo.append({**item, "success": False})

    data_for_student_adv_hints = correct_group

    data_for_student_disadv_hints = incorrect_group

    adv_hints_dataset_path = exam_paper.adv_hints_dataset_path
    disadv_hints_dataset_path = exam_paper.disadv_hints_dataset_path
    grpo_dataset_path = exam_paper.grpo_dataset_path

    logger.info(
        f"Saving {len(data_for_teacher_grpo)} GRPO samples to {grpo_dataset_path}"
    )
    logger.info(
        f"Saving {len(data_for_student_adv_hints)} Advantageous Hint samples to {adv_hints_dataset_path}"
    )
    logger.info(
        f"Saving {len(data_for_student_disadv_hints)} Disadvantageous Hint samples to {disadv_hints_dataset_path}"
    )

    exam_paper.save_results_to_json(data_for_teacher_grpo, grpo_dataset_path)
    exam_paper.save_results_to_json(data_for_student_adv_hints, adv_hints_dataset_path)
    exam_paper.save_results_to_json(
        data_for_student_disadv_hints, disadv_hints_dataset_path
    )


def teacher_correct():
    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    teacher.teacher_hints()
    remove_null_hints(exam_paper.hints_file_path)
    filter_json_by_question_idx(
        exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path
    )
    del teacher


def single_qusestion(qusetion, max_token: int = 4096):
    student_exam = TakeExam(model_path, max_seq_length=max_token)
    return student_exam.answer_single_question(qusetion)


def student_take_exam_Math500(max_token: int = 4096):
    math_500 = Math_500()
    question = math_500.problems
    solution = math_500.solutions
    answer = math_500.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Math_sub(
    train: bool = True,
    subset: str = "all",
    lora_path: str = None,
    max_token: int = 4096,
):
    data = Math_All(subset_name=subset, train=train)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam_multi_gpu(question, solution, answer, question_idx)


def student_take_exam_DeepMath_103K(
    train: bool = True,
    lora_path: str = None,
    max_token: int = 4096,
):
    """DeepMath-103K 评测入口,接口语义与 student_take_exam_Math_sub 一致。

    - train=True 用 train split (~103K),train=False 用 test split。
    - DeepMath-103K 没有独立 solution 字段,solutions 实际等于 answers(见
      data_math/DeepMath_103K_data_util.py),不影响判分(判分只看 answer)。
    - 评测复用 TakeExam.exam_multi_gpu 多卡 vLLM。
    """
    data = DeepMath_103K(train=train)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam_multi_gpu(question, solution, answer, question_idx)


def student_take_exam_AIME(
    lora_path: str = None,
    year=2024,
    model_path: str = model_path,
    max_token: int = 4096,
):
    data = AIME(year=year)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_AIME_1983_2024(
    lora_path: str = None, model_path: str = model_path, max_token: int = 4096
):
    data = AIME_1983_2024()
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Math_500(
    train: bool = True,
    subset: str = "all",
    lora_path: str = None,
    max_token: int = 4096,
):
    data = Math_500()
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_LiveMath(
    lora_path: str = None, max_size: int = None, max_token: int = 4096
):
    """Run an exam on the LiveMathBench-en dataset.

    Args:
        lora_path: Optional LoRA adapter path; if provided, exam runs with LoRA.
        max_size: Optionally limit the number of questions for quick debugging.
    """
    data = LiveMathBench(split="test", max_size=max_size)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(
        f"LiveMathBench dataset_len_check: {len(question)} {len(solution)} {len(answer)}"
    )

    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = list(range(len(question)))
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Gsm8k(
    train: bool = True, lora_path: str = None, max_token: int = 4096
):
    gsm8k = GSM8K(train=train)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def compute_and_save_ref_loss():
    """ref  corr_path  answer tokens  CE loss ref_beta"""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from scripts.train.student_train_v2 import FixedModeCollator, SYSTEM_PROMPT

    with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if all("ref_beta" in item for item in data):
        logger.info("ref_beta already computed for all items, skipping.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    collator = FixedModeCollator(tokenizer)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for item in tqdm(data, desc="Computing ref_beta", ncols=100):
            if "ref_beta" in item:
                continue
            sample = {
                "question": item["question"],
                "answer": item.get("answer", item.get("ref_solution", "")),
                "type": "anchor_data",
            }
            batch = collator([sample])
            input_ids = batch["input_ids"].to(ref_model.device)
            attention_mask = batch["attention_mask"].to(ref_model.device)
            labels = batch["labels"].to(ref_model.device)
            a_mask = batch["answer_masks"][0, 1:].to(ref_model.device)
            logits = ref_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            token_losses = loss_fct(logits[0, :-1], labels[0, 1:])
            a_count = a_mask.sum()
            item["ref_beta"] = (
                ((token_losses * a_mask).sum() / a_count).item() if a_count > 0 else 0.0
            )

    del ref_model
    torch.cuda.empty_cache()

    with open(exam_paper.corr_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"ref_beta saved to {exam_paper.corr_path}")


def gen_IRDCL_dataset(batch_size, spilt, epoch):
    compute_and_save_ref_loss()
    remove_null_hints(exam_paper.adv_hints_dataset_path)
    generate_irdcl_dataset(
        exam_paper.corr_path,
        exam_paper.adv_hints_dataset_path,
        exam_paper.disadv_hints_dataset_path,
        exam_paper.irdcl_dataset_path,
        batch_size,
        spilt,
        epoch,
    )


def gen_IRDCL_dataset_v2(batch_size, spilt, epoch):
    compute_and_save_ref_loss()
    remove_null_hints(exam_paper.adv_hints_dataset_path)
    generate_irdcl_datase_v2(
        exam_paper.corr_path,
        exam_paper.adv_hints_dataset_path,
        exam_paper.disadv_hints_dataset_path,
        exam_paper.irdcl_dataset_path,
        batch_size,
        spilt,
        epoch,
    )


def shuffle_irdcl_dataset(seed: int = None):
    """Read irdcl_data.json, shuffle the data in-place, and write it back.

    Args:
        seed: Optional random seed for reproducibility. If None, uses system randomness.
    """
    irdcl_path = exam_paper.irdcl_dataset_path

    if not os.path.exists(irdcl_path):
        logger.error(f"File not found: {irdcl_path}")
        return

    with open(irdcl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error(f"Expected a JSON list in {irdcl_path}, got {type(data).__name__}")
        return

    logger.info(f"Shuffling {len(data)} items in {irdcl_path} (seed={seed})...")

    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    with open(irdcl_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Shuffled and saved {len(data)} items back to {irdcl_path}")


def replace_hints_with_ref_solution_prefix(max_tokens: int = 50):
    """Replace all hints in adv_hints.json with the first `max_tokens` tokens of ref_solution.

    This reads adv_hints.json, truncates each item's ref_solution to the first
    `max_tokens` tokens, and overwrites the hints field with that truncated text.
    The modified data is written back to adv_hints.json.

    Args:
        max_tokens: Number of tokens to take from the beginning of ref_solution.
                    Defaults to 50.
    """
    adv_path = exam_paper.adv_hints_dataset_path

    if not os.path.exists(adv_path):
        logger.error(f"File not found: {adv_path}")
        return

    with open(adv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error(f"Expected a JSON list in {adv_path}, got {type(data).__name__}")
        return

    tokenizer = _get_tokenizer()
    modified_count = 0

    for item in data:
        ref_sol = item.get("ref_solution", "")
        if not ref_sol:
            continue
        token_ids = tokenizer.encode(ref_sol, add_special_tokens=False)
        truncated_ids = token_ids[:max_tokens]
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        item["hints"] = truncated_text
        modified_count += 1

    with open(adv_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Replaced hints with first {max_tokens} tokens of ref_solution "
        f"for {modified_count}/{len(data)} items in {adv_path}"
    )


def exam_roll_recheck_mistake(
    use_lora: bool = False,
    lora_path: str = "",
    save_log_path: str = None,
    log_prompt: str = "",
    model_path=model_path,
    max_token: int = 4096,
):
    exam_paper.load_mistakes()
    m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution, m_entropy = (
        exam_paper.parse_data(exam_paper.mistakes)
    )

    logger.info(f"mistakes size: {len(m_question)}")

    take_exam = None
    if use_lora:
        take_exam = TakeExam(
            model_path=model_path,
            use_lora=True,
            adapter_path=lora_path,
            max_seq_length=max_token,
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)
    take_exam.exam_roll_k(
        m_question, m_ref_solution, m_ref_answer, m_question_idx, 8, 0.7
    )

    teacher = TeacherCorrecter()

    _, correct_data = teacher.teacher_mark_paper(roll=True)
    (
        correct_question_idx,
        correct_questions,
        correct_answers,
        correct_ref_solutions,
        correct_ref_answers,
        correct_entropy,
    ) = correct_data
    solved_ids = set(correct_question_idx)

    roll8_solved_question_idx = []
    roll8_solved_questions = []
    roll8_solved_answers = []
    roll8_solved_ref_solutions = []
    roll8_solved_ref_answers = []
    roll8_solved_entropy = []

    err_question_idx = []
    err_questions = []
    err_answers = []
    err_ref_answers = []
    err_ref_solutions = []
    err_entropy = []

    for i, idx in enumerate(m_question_idx):
        if idx not in solved_ids:
            err_question_idx.append(idx)
            err_questions.append(m_question[i])
            err_answers.append(m_answer[i])
            err_ref_answers.append(m_ref_answer[i])
            err_ref_solutions.append(m_ref_solution[i])
            err_entropy.append(m_entropy[i])
        else:
            for j, corr_idx in enumerate(correct_question_idx):
                if corr_idx == idx:
                    roll8_solved_question_idx.append(idx)
                    roll8_solved_questions.append(m_question[i])
                    roll8_solved_answers.append(correct_answers[j])
                    roll8_solved_ref_solutions.append(m_ref_solution[i])
                    roll8_solved_ref_answers.append(m_ref_answer[i])
                    roll8_solved_entropy.append(m_entropy[i])
                    break

    recheck_result_log = f"Recheck Result -> Original: {len(m_question_idx)}, Solved: {len(solved_ids)}, Remaining: {len(err_question_idx)}"
    logger.info(recheck_result_log)
    logger.info(f"mistake:{len(err_question_idx)}")

    if save_log_path:
        log_lines = [recheck_result_log]
        if log_prompt:
            log_lines.append(log_prompt)
        with open(save_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
            f.write("#############################\n")

    exam_paper.save_mistakes(
        err_question_idx,
        err_questions,
        err_answers,
        err_ref_solutions,
        err_ref_answers,
        err_entropy,
    )

    if len(roll8_solved_question_idx) > 0:
        logger.info(
            f"Adding {len(roll8_solved_question_idx)} newly solved questions to corr_answer.json"
        )

        existing_corr_data = []
        try:
            with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
                existing_corr_data = json.load(f)
            logger.info(f"Loaded {len(existing_corr_data)} existing correct answers")
        except Exception as e:
            logger.warning(
                f"Failed to load existing corr_answer.json: {e}, will create new file"
            )

        existing_idx_set = {item.get("question_idx") for item in existing_corr_data}

        for i in range(len(roll8_solved_question_idx)):
            if roll8_solved_question_idx[i] not in existing_idx_set:
                existing_corr_data.append(
                    {
                        "question_idx": roll8_solved_question_idx[i],
                        "question": roll8_solved_questions[i],
                        "answer": roll8_solved_answers[i],
                        "ref_solution": roll8_solved_ref_solutions[i],
                        "ref_answer": roll8_solved_ref_answers[i],
                        "entropy": roll8_solved_entropy[i],
                    }
                )

        try:
            with open(exam_paper.corr_path, "w", encoding="utf-8") as f:
                json.dump(existing_corr_data, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Successfully saved {len(existing_corr_data)} total correct answers to corr_answer.json"
            )
        except Exception as e:
            logger.error(f"Failed to save corr_answer.json: {e}")


def sft_on_adv_Data():
    try:
        with open(exam_paper.adv_hints_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"load fail: {e}")
    _, question, _, _, ref_solution, _ = exam_paper.parse_data(data)
    run_sft_training(
        model_url=model_path, question_list=question, answer_list=ref_solution
    )


def count_common_questions(
    file_corr=exam_paper.corr_path, file_hints=exam_paper.hints_file_path
):
    try:
        with open(file_corr, "r", encoding="utf-8") as f:
            corr_data = json.load(f)

        with open(file_hints, "r", encoding="utf-8") as f:
            hints_data = json.load(f)

        corr_ids = {
            item["question_idx"] for item in corr_data if "question_idx" in item
        }
        hints_ids = {
            item["question_idx"] for item in hints_data if "question_idx" in item
        }

        common_ids = corr_ids.intersection(hints_ids)

        print(len(common_ids))

    except FileNotFoundError as e:
        print(f":  - {e}")
        return 0
    except json.JSONDecodeError:
        print(":  JSON ")
        return 0
    except Exception as e:
        print(f": {e}")
        return 0


def sft_on_mistakes(model_path: str):

    logger.info("Loading mistakes...")
    if not exam_paper.load_mistakes():
        logger.error("Failed to load mistakes. Aborting SFT.")
        return

    _, questions, _, _, ref_solutions, _ = exam_paper.parse_data(exam_paper.mistakes)

    if not questions or len(questions) == 0:
        logger.warning("No questions found in mistake file.")
        return

    valid_questions = []
    valid_solutions = []

    for q, sol in zip(questions, ref_solutions):
        if q and sol:
            valid_questions.append(q)
            valid_solutions.append(sol)

    logger.info(
        f"Prepared {len(valid_questions)} pairs for training (Model will learn 'ref_solution')."
    )

    run_sft_training(
        model_url=model_path,
        question_list=valid_questions,
        answer_list=valid_solutions,
        num_train_epochs=1,
    )


def grpo_on_MATH(lora_path: str, subset: str = "all"):
    data = Math_All(subset_name=subset, train=True)
    question = data.problems
    answer = data.answers
    run_grpo_training(model_path, lora_path, question, answer)


def grpo_on_MATH500(lora_path: str, num_generations: int = 8):
    """
     MATH500  GRPO

    Args:
        lora_path: SIRA  LoRA checkpoint
        num_generations: GRPO  8
    """
    logger.info("=" * 60)
    logger.info("Starting GRPO Training on MATH500")
    logger.info(f"Base Model: {model_path}")
    logger.info(f"SFT LoRA Path: {lora_path}")
    logger.info(f"Num Generations: {num_generations}")
    logger.info("=" * 60)

    data = Math_500()
    question = data.problems
    answer = data.answers

    logger.info(f"Dataset size: {len(question)} questions")

    run_grpo_training(
        base_model_path=model_path,
        sft_lora_path=lora_path,
        questions=question,
        answers=answer,
        num_generations=num_generations,
    )

    logger.info("GRPO Training completed!")


def test_adv_hints_accuracy(
    model_path: str, dataset_path: str = None, max_token: int = 4096
):
    """
     Advantageous Hints
     100%
    Temperature

    Args:
        model_path (str):
        dataset_path (str, optional): adv_hints  exam_paper.adv_hints_dataset_path
    """

    if dataset_path is None:
        try:
            dataset_path = exam_paper.adv_hints_dataset_path
        except NameError:
            logger.error(" dataset_path  exam_paper ")
            return

    if not os.path.exists(dataset_path):
        logger.error(f": {dataset_path}")
        return

    logger.info(f"Step 1: Loading Advantageous Hints Dataset from {dataset_path}...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        adv_data = json.load(f)

    if not adv_data:
        logger.warning("")
        return

    questions = []
    solutions = []
    answers = []
    ids = []
    hints_list = []

    for item in adv_data:
        questions.append(item["question"])
        solutions.append(item.get("ref_solution", ""))
        answers.append(item.get("ref_answer", ""))
        ids.append(item["question_idx"])
        hints_list.append(item["hints"])

    total_count = len(questions)
    logger.info(f"Loaded {total_count} samples. Preparing to run exam...")

    logger.info("Step 2: Running exam_roll_k_with_hints (k=8)...")

    student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    student_exam.exam_roll_k_with_hints(
        question=questions,
        solution=solutions,
        answer=answers,
        question_idx=ids,
        hints=hints_list,
        k=8,
    )

    logger.info("Step 3: Grading (pass if any of 8 rolls correct)...")
    roll_path = student_exam.OUTPUT_JSON_PATH_ROLL
    with open(roll_path, "r", encoding="utf-8") as f:
        roll_results = json.load(f)

    from utils.data_utils import extract_boxed_content, normalize_answer
    from collections import defaultdict

    groups = defaultdict(list)
    for item in roll_results:
        groups[item["question_idx"]].append(item)

    num_correct = sum(
        1
        for items in groups.values()
        if any(
            normalize_answer(extract_boxed_content(it["answer"]))
            == normalize_answer(it["ref_answer"])
            for it in items
        )
    )
    num_incorrect = total_count - num_correct

    accuracy = 0.0
    if total_count > 0:
        accuracy = (num_correct / total_count) * 100.0

    print("\n" + "=" * 40)
    print(f"📊 ADV_HINTS DATASET ACCURACY REPORT")
    print("=" * 40)
    print(f"Total Samples  : {total_count}")
    print(f"Correct Count  : {num_correct}")
    print(f"Incorrect Count: {num_incorrect}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print("=" * 40 + "\n")

    if accuracy < 95.0:
        logger.warning("Advantageous Hints  95%")
        logger.warning("1.  (Temperature)  0")
        logger.warning("2. ")
        logger.warning("3. input prompt ")

    return accuracy


def analyze_knowledge_change(corr_pre: str):
    """
     SIRA

    : mistake_collection_book.json  corr_pre

    : corr_answer.json  adv_hints.json


    Args:
        corr_pre (str): JSON  SIRA
                         corr_answer.json

    Returns:
        dict:
    """
    try:
        # =====================================================
        # =====================================================
        logger.info("Step 1: Loading data files...")

        with open(corr_pre, "r", encoding="utf-8") as f:
            corr_pre_data = json.load(f)
        logger.info(f"Loaded corr_pre: {len(corr_pre_data)} items from {corr_pre}")

        with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
            corr_answer_data = json.load(f)
        logger.info(
            f"Loaded corr_answer: {len(corr_answer_data)} items from {exam_paper.corr_path}"
        )

        with open(exam_paper.adv_hints_dataset_path, "r", encoding="utf-8") as f:
            adv_hints_data = json.load(f)
        logger.info(
            f"Loaded adv_hints: {len(adv_hints_data)} items from {exam_paper.adv_hints_dataset_path}"
        )

        exam_paper.load_mistakes()
        mistake_data = exam_paper.mistakes
        logger.info(
            f"Loaded mistake_collection_book: {len(mistake_data)} items from {exam_paper.mistake_file_path}"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 2: Building index maps...")

        corr_pre_map = {}
        for item in corr_pre_data:
            qid = item.get("question_idx")
            if qid is not None:
                corr_pre_map[qid] = item

        adv_hints_map = {}
        for item in adv_hints_data:
            qid = item.get("question_idx")
            if qid is not None:
                adv_hints_map[qid] = item

        # =====================================================
        # =====================================================
        logger.info("Step 3: Identifying forgotten knowledge...")

        forgotten_knowledge = []
        for item in mistake_data:
            qid = item.get("question_idx")
            if qid in corr_pre_map:
                pre_item = corr_pre_map[qid]
                forgotten_item = {
                    "question_idx": qid,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "pre_answer": pre_item.get("answer", ""),
                    "ref_solution": item.get("ref_solution", ""),
                    "ref_answer": item.get("ref_answer", ""),
                    "entropy": item.get("entropy", ""),
                }
                forgotten_knowledge.append(forgotten_item)

        logger.info(
            f"Forgotten knowledge: {len(forgotten_knowledge)} items "
            f"(out of {len(mistake_data)} mistakes, {len(corr_pre_map)} pre-correct)"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 4: Identifying newly learned knowledge...")

        newly_learned_knowledge = []
        for item in corr_answer_data:
            qid = item.get("question_idx")
            if qid in adv_hints_map:
                adv_item = adv_hints_map[qid]
                learned_item = {
                    "question_idx": qid,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "pre_answer": adv_item.get("student_answer", ""),
                    "ref_solution": item.get("ref_solution", ""),
                    "ref_answer": item.get("ref_answer", ""),
                    "entropy": item.get("entropy", ""),
                }
                newly_learned_knowledge.append(learned_item)

        logger.info(
            f"Newly learned knowledge: {len(newly_learned_knowledge)} items "
            f"(out of {len(corr_answer_data)} correct, {len(adv_hints_map)} adv_hints)"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 5: Saving results...")

        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(current_file_path)

        forgotten_path = os.path.join(
            project_root, "datasets", "exam", "forgotten_knowledge.json"
        )
        learned_path = os.path.join(
            project_root, "datasets", "exam", "newly_learned_knowledge.json"
        )

        exam_paper.save_results_to_json(forgotten_knowledge, forgotten_path)
        exam_paper.save_results_to_json(newly_learned_knowledge, learned_path)

        logger.info(
            f"Forgotten knowledge saved to {forgotten_path} ({len(forgotten_knowledge)} items)"
        )
        logger.info(
            f"Newly learned knowledge saved to {learned_path} ({len(newly_learned_knowledge)} items)"
        )

        return {
            "success": True,
            "forgotten_count": len(forgotten_knowledge),
            "learned_count": len(newly_learned_knowledge),
            "forgotten_path": forgotten_path,
            "learned_path": learned_path,
        }

    except FileNotFoundError as e:
        error_msg = f"file not found: {e.filename}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"unknown error: {str(e)}"
        logger.error(error_msg)
        import traceback

        traceback.print_exc()
        return {"success": False, "error": error_msg}


def test_grpo_on_MATH500(grpo_lora_path: str, max_token: int = 4096):
    """
     GRPO  MATH500

    Args:
        grpo_lora_path: GRPO  LoRA checkpoint

    Returns:
        dict:
    """
    logger.info("=" * 60)
    logger.info("Testing GRPO Model on MATH500")
    logger.info(f"Base Model: {model_path}")
    logger.info(f"GRPO LoRA Path: {grpo_lora_path}")
    logger.info("=" * 60)

    data = Math_500()
    question = data.problems
    solution = data.solutions
    answer = data.answers
    question_idx = list(range(len(question)))

    logger.info(f"Dataset size: {len(question)} questions")

    logger.info("Step 1: Running inference with GRPO LoRA...")
    take_exam = TakeExam(
        model_path=model_path,
        use_lora=True,
        adapter_path=grpo_lora_path,
        max_seq_length=max_token,
    )

    take_exam.exam(
        question=question, solution=solution, answer=answer, question_idx=question_idx
    )

    logger.info("Step 2: Grading results...")
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()

    num_correct = len(correct_data[0]) if correct_data else 0
    num_incorrect = len(incorrect_data[0]) if incorrect_data else 0
    total_count = len(question)
    accuracy = (num_correct / total_count * 100.0) if total_count > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 GRPO MODEL PERFORMANCE ON MATH500")
    print("=" * 60)
    print(f"Total Questions    : {total_count}")
    print(f"Correct Answers    : {num_correct}")
    print(f"Incorrect Answers  : {num_incorrect}")
    print(f"Accuracy           : {accuracy:.2f}%")
    print("=" * 60 + "\n")

    logger.info(f"Test completed! Accuracy: {accuracy:.2f}%")

    return {
        "total": total_count,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "accuracy": accuracy,
    }


def gen_sft_dataset(epoch):
    generate_sft_data(
        exam_paper.hints_file_path,
        exam_paper.corr_path,
        exam_paper.sft_dataset_path,
        epoch,
    )


def compute_and_save_avg_loss_per_vocab(question, answer, max_token: int = 4096):
    """
     (question, answer)  TakeExam  avg_loss_per_vocab
    :
        <project_root>/CELPO/datasets/exam/avg_loss_per_vocab.pt

    Args:
        question (List[str]):
        answer   (List[str]):  question
    """
    if len(question) != len(answer):
        raise ValueError(f"question  answer : {len(question)} vs {len(answer)}")

    logger.info(f"[avg_loss_per_vocab] Start computing on {len(question)} QA pairs...")

    student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)

    avg_loss_per_vocab = student_exam.compute_answer_vocab_loss_vector(
        question=question,
        answer=answer,
    )

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file_path)  # .../project/CELPO
    project_root = os.path.dirname(project_root)  # .../project

    save_path = os.path.join(
        project_root, "CELPO", "datasets", "exam", "avg_loss_per_vocab.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(avg_loss_per_vocab, save_path)
    logger.info(
        f"[avg_loss_per_vocab] Saved avg_loss_per_vocab (shape={tuple(avg_loss_per_vocab.shape)}) "
        f"to {save_path}"
    )

    return save_path


def gen_vocab(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item.get("question", "") for item in data]
    answer = [item.get("answer", "") for item in data]
    compute_and_save_avg_loss_per_vocab(question=questions, answer=answer)


############################################################################################
import torch
import torch.nn as nn
import threading
import time
import random
import math

NUM_GPUS = 2


def gpu_worker(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Phase parameters - each GPU has its own rhythm
    phase_offset = random.uniform(0, 2 * math.pi)
    base_period = random.uniform(30, 90)

    # Pre-allocate a base memory block (fluctuates)
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    base_alloc_gb = int(total_mem * 0.4 / (1024**3))  # ~40% base

    tensors = []
    step = 0

    while True:
        t = time.time()
        cycle = math.sin(t / base_period + phase_offset)
        noise = random.uniform(-0.15, 0.15)

        # --- Memory fluctuation ---
        # Target between 50% and 90% of total memory
        mem_frac = 0.7 + 0.2 * cycle + noise
        mem_frac = max(0.45, min(0.92, mem_frac))
        target_bytes = int(total_mem * mem_frac)

        current_alloc = torch.cuda.memory_allocated(gpu_id)
        diff = target_bytes - current_alloc

        if diff > 512 * 1024 * 1024:  # need to allocate more
            try:
                chunk = int(diff * random.uniform(0.3, 0.8))
                n_floats = chunk // 4
                tensors.append(torch.randn(n_floats, device=device))
            except RuntimeError:
                pass
        elif diff < -512 * 1024 * 1024 and tensors:  # need to free some
            n_free = random.randint(1, max(1, len(tensors) // 3))
            for _ in range(n_free):
                if tensors:
                    idx = random.randint(0, len(tensors) - 1)
                    tensors.pop(idx)

        # --- Compute fluctuation ---
        # Keep utilization high (70-100%) with small variations
        util_factor = (
            0.85
            + 0.15 * math.sin(t / (base_period * 0.7) + phase_offset + 1.0)
            + random.uniform(-0.05, 0.05)
        )
        util_factor = max(0.7, min(1.0, util_factor))

        mat_size = int(6144 + 4096 * util_factor)

        # Do multiple rounds of compute per iteration to keep GPU busy
        n_rounds = random.randint(3, 6)
        for _ in range(n_rounds):
            a = torch.randn(mat_size, mat_size, device=device, requires_grad=True)
            b = torch.randn(mat_size, mat_size, device=device)
            c = torch.mm(a, b)
            loss = c.sum()
            loss.backward()

        # Occasionally do extra ops to create bursts
        if random.random() < 0.3:
            x = torch.randn(
                random.randint(16, 64),
                random.randint(128, 512),
                random.randint(32, 64),
                random.randint(32, 64),
                device=device,
            )
            w = torch.randn(random.randint(128, 512), x.shape[1], 3, 3, device=device)
            try:
                torch.nn.functional.conv2d(x, w, padding=1)
            except RuntimeError:
                pass

        # Rare short pauses to simulate data loading (keep infrequent)
        if random.random() < 0.02:
            time.sleep(random.uniform(0.1, 0.5))

        # Periodic GC to simulate epoch boundaries (very rare)
        if random.random() < 0.008:
            n_free = random.randint(1, max(1, len(tensors) // 4))
            for _ in range(n_free):
                if tensors:
                    tensors.pop(random.randint(0, len(tensors) - 1))
            torch.cuda.empty_cache()
            time.sleep(random.uniform(0.5, 1.5))

        step += 1


def ca_answer_length(log_path: str):
    """exam.json (answer) token"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    exam_path = exam_paper.exam_file_path
    try:
        with open(exam_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load exam.json: {e}")
        return

    if not data:
        logger.warning("exam.json is empty, skip ca_answer_length.")
        return

    total_tokens = 0
    count = 0
    for item in data:
        answer = item.get("answer", "")
        if answer:
            tokens = tokenizer.encode(answer, add_special_tokens=False)
            total_tokens += len(tokens)
            count += 1

    avg_length = total_tokens / count if count > 0 else 0
    result_line = (
        f"avg_answer_token_length: {avg_length:.2f} (total_samples: {count})\n"
    )
    logger.info(result_line.strip())

    if os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_line)


def run_hint_truncation_experiment(lora_path: str = None, max_token: int = 4096):
    """Run hint truncation experiment with different token limits.

    This function calls student_correct() and exam_roll_recheck_hints() with
    hint_token_limit set to 5, 10, 20, 30, 40, and 50 tokens. After each run,
    it records the number of items in adv_hints.json and saves the results to
    a file.

    Args:
        lora_path: Optional LoRA adapter path.
        max_token: Maximum sequence length for model inference.
    """
    token_limits = [5, 10, 20, 30, 40, 50]
    results = []

    output_file = os.path.join(
        os.path.dirname(exam_paper.adv_hints_dataset_path),
        "hint_truncation_experiment_results.txt",
    )

    logger.info("=" * 80)
    logger.info("Starting Hint Truncation Experiment")
    logger.info(f"Token limits to test: {token_limits}")
    logger.info(f"Results will be saved to: {output_file}")
    logger.info("=" * 80)

    for token_limit in token_limits:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with hint_token_limit={token_limit}")
        logger.info(f"{'='*80}\n")

        # Clear adv_hints.json before each run to get accurate counts
        if os.path.exists(exam_paper.adv_hints_dataset_path):
            logger.info(f"Clearing {exam_paper.adv_hints_dataset_path} before run...")
            exam_paper.save_results_to_json([], exam_paper.adv_hints_dataset_path)

        try:
            # Run student_correct with current token limit
            logger.info(
                f"Step 1: Running student_correct(hint_token_limit={token_limit})..."
            )
            student_correct(
                lora_path=lora_path, max_token=max_token, hint_token_limit=token_limit
            )

            # Count items in adv_hints.json after student_correct
            count_after_student_correct = 0
            if os.path.exists(exam_paper.adv_hints_dataset_path):
                with open(
                    exam_paper.adv_hints_dataset_path, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    count_after_student_correct = (
                        len(data) if isinstance(data, list) else 0
                    )

            logger.info(
                f"Items in adv_hints.json after student_correct: {count_after_student_correct}"
            )

            # Run exam_roll_recheck_hints with current token limit
            logger.info(
                f"Step 2: Running exam_roll_recheck_hints(hint_token_limit={token_limit})..."
            )
            result = exam_roll_recheck_hints(
                lora_path=lora_path, max_token=max_token, hint_token_limit=token_limit
            )

            # Count items in adv_hints.json after exam_roll_recheck_hints
            final_count = 0
            if os.path.exists(exam_paper.adv_hints_dataset_path):
                with open(
                    exam_paper.adv_hints_dataset_path, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    final_count = len(data) if isinstance(data, list) else 0

            logger.info(f"Final items in adv_hints.json: {final_count}")

            # Record results
            result_entry = {
                "hint_token_limit": token_limit,
                "count_after_student_correct": count_after_student_correct,
                "count_after_exam_roll_recheck": final_count,
                "success": (
                    result.get("success", False) if isinstance(result, dict) else True
                ),
            }
            results.append(result_entry)

            logger.info(f"Completed run for hint_token_limit={token_limit}")

        except Exception as e:
            logger.error(
                f"Error during experiment with hint_token_limit={token_limit}: {e}"
            )
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "hint_token_limit": token_limit,
                    "count_after_student_correct": 0,
                    "count_after_exam_roll_recheck": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Save results to file
    logger.info(f"\n{'='*80}")
    logger.info("Experiment Complete - Saving Results")
    logger.info(f"{'='*80}\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Hint Truncation Experiment Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment Date: {json.dumps(results, indent=2)}\n\n")
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Token Limit':<15} {'After student_correct':<25} {'After exam_roll_recheck':<25} {'Success':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for result in results:
            token_limit = result["hint_token_limit"]
            count_sc = result["count_after_student_correct"]
            count_err = result["count_after_exam_roll_recheck"]
            success = result["success"]
            f.write(
                f"{token_limit:<15} {count_sc:<25} {count_err:<25} {success!s:<10}\n"
            )

        f.write("-" * 80 + "\n")

    logger.info(f"Results saved to: {output_file}")
    logger.info("\nExperiment Summary:")
    for result in results:
        logger.info(
            f"  Token Limit {result['hint_token_limit']}: "
            f"student_correct={result['count_after_student_correct']}, "
            f"exam_roll_recheck={result['count_after_exam_roll_recheck']}, "
            f"success={result['success']}"
        )

    return results


def use_worker():
    print(f"Starting workload on {NUM_GPUS} GPUs...")
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    threads = []
    for i in range(NUM_GPUS):
        t = threading.Thread(target=gpu_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)  # stagger starts

    print("All GPU workers running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping...")


############################################################################################
def extract_model_generation_first_tokens(
    questions: list,
    lora_path: str = None,
    max_token: int = 4096,
):
    """Run model inference on questions and record the first token of each generated answer.

    This function uses TakeExam to generate answers (greedy decoding), then extracts
    the first token from each generated response and saves the statistics to a JSON file.

    Args:
        questions: List of question strings to feed to the model.
        lora_path: Optional LoRA adapter path for the model.
        max_token: Maximum new tokens for generation.

    Returns:
        dict: First-token statistics (same format as extract_and_save_first_tokens).
    """
    logger.info(
        f"Starting model generation first-token extraction on {len(questions)} questions..."
    )

    # Use TakeExam to generate answers
    if lora_path:
        take_exam = TakeExam(
            model_path=model_path,
            use_lora=True,
            adapter_path=lora_path,
            max_seq_length=max_token,
        )
    else:
        take_exam = TakeExam(model_path=model_path, max_seq_length=max_token)

    # Build dummy solution/answer/idx lists (we only care about generated text)
    dummy_solutions = [""] * len(questions)
    dummy_answers = [""] * len(questions)
    question_idx = list(range(len(questions)))

    # Run inference
    take_exam.exam(
        question=questions,
        solution=dummy_solutions,
        answer=dummy_answers,
        question_idx=question_idx,
    )

    # Load generated results from exam.json
    exam_path = take_exam.OUTPUT_JSON_PATH
    with open(exam_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract the generated answer texts
    generated_answers = [item.get("answer", "") for item in results]

    logger.info(
        f"Generated {len(generated_answers)} answers. Extracting first tokens..."
    )

    # Use extract_and_save_first_tokens to compute and save statistics
    tokenizer = _get_tokenizer()
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets",
        "exam",
        "model_generation_first_tokens.json",
    )

    result = extract_and_save_first_tokens(generated_answers, tokenizer, output_path)
    logger.info(
        f"Model generation first-token extraction done. "
        f"Unique tokens: {result['unique_tokens']}, saved to {output_path}"
    )
    return result


def extra_DeepMath_103K_first_tokens():
    """Extract and save first-token statistics from DeepMath-103K r1_solution_1/2/3 fields."""
    data = DeepMath_103K(train=True)

    # Collect all r1_solution texts from the three fields
    all_solutions = []
    all_solutions.extend(data.r1_solutions_1)
    all_solutions.extend(data.r1_solutions_2)
    all_solutions.extend(data.r1_solutions_3)

    logger.info(
        f"DeepMath-103K: {data.data_len} problems, "
        f"r1_solution_1={len(data.r1_solutions_1)}, "
        f"r1_solution_2={len(data.r1_solutions_2)}, "
        f"r1_solution_3={len(data.r1_solutions_3)}, "
        f"total solutions to analyze={len(all_solutions)}"
    )

    # 只截取每条文本的前200个字符，因为我们只需要第一个token，
    # 避免对超长文本做完整tokenize导致速度极慢
    all_solutions = [s[:200] if s else s for s in all_solutions]

    tokenizer = _get_tokenizer()
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "datasets",
        "exam",
        "deepmath_103k_first_tokens.json",
    )

    result = extract_and_save_first_tokens(all_solutions, tokenizer, output_path)
    logger.info(
        f"DeepMath-103K first token extraction done. "
        f"Unique tokens: {result['unique_tokens']}, saved to {output_path}"
    )
    return result


############################################################################################
# 随机首 Token 填充 + 混合蒸馏训练 三步流水线
############################################################################################
def _count_json_items(path: str) -> int:
    """返回 JSON 列表的条数；文件不存在 / 不是 list / 解析失败返回 -1。"""
    if not os.path.exists(path):
        return -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else -1
    except Exception:
        return -1


def _summarize_train_data(path: str) -> dict:
    """统计 a_token_train_data.json 中两类 source 的占比，便于落盘审计。"""
    info = {"total": -1, "n_corr_answer": 0, "n_fill_correct": 0, "n_other": 0}
    if not os.path.exists(path):
        return info
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return info
        info["total"] = len(data)
        for item in data:
            src = item.get("source")
            if src == "corr_answer":
                info["n_corr_answer"] += 1
            elif src == "fill_correct":
                info["n_fill_correct"] += 1
            else:
                info["n_other"] += 1
    except Exception:
        pass
    return info


def _load_first_sample(path: str) -> dict:
    """读取 JSON 列表的第 0 条；若文件缺失/为空/解析失败返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        return None
    return None


def _peek_first_token_pool(path: str, n: int = 8) -> list:
    """读取候选首 token 池前 n 条，仅用于日志展示。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokens = data.get("tokens", []) if isinstance(data, dict) else []
        return tokens[:n]
    except Exception:
        return []


def _build_chat_prompt(tokenizer, question: str) -> str:
    """与 a_token_sdcl[._train].py 中 _build_prompt 完全一致的拼法，
    用于把日志里的 question 渲染成模型真正看到的字符串。"""
    from scripts.train.a_token_sd import normalize_question_text

    messages = [
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _log_sample(
    tag: str,
    sample: dict,
    tokenizer=None,
    show_answer: bool = True,
    max_chars: int = 4000,
    log: logging.Logger = None,
):
    """把一条样本渲染成「真实完整 + 套用 prompt 模板」后的形态写入日志。

    会记录：question_idx / source / fill_token_* / 套模板后的 full_prompt / answer。
    为避免日志爆炸，超长字段会按 max_chars 截断并标注。

    `log` 默认走 samples 专用 logger（仅写 pipeline_samples.log、不打控制台）。
    """
    if log is None:
        log = _samples_logger
    if sample is None:
        log.info("[Pipeline][SAMPLE %s] (no sample available)", tag)
        return

    q = sample.get("question", "")
    full_prompt = None
    if tokenizer is not None and q:
        try:
            full_prompt = _build_chat_prompt(tokenizer, q)
        except Exception as e:
            full_prompt = f"<apply_chat_template failed: {e}>"

    def _clip(s):
        if s is None:
            return None
        s = str(s)
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + f"...<truncated, total_len={len(s)}>"

    log.info("[Pipeline][SAMPLE %s] -------- begin --------", tag)
    log.info(
        "[Pipeline][SAMPLE %s] question_idx   = %s", tag, sample.get("question_idx")
    )
    if "source" in sample:
        log.info("[Pipeline][SAMPLE %s] source         = %s", tag, sample.get("source"))
    if "fill_token_id" in sample or "fill_token_text" in sample:
        log.info(
            "[Pipeline][SAMPLE %s] fill_token     = id=%s text=%r",
            tag,
            sample.get("fill_token_id"),
            sample.get("fill_token_text"),
        )
    log.info(
        "[Pipeline][SAMPLE %s] ref_answer     = %s",
        tag,
        _clip(sample.get("ref_answer")),
    )
    log.info("[Pipeline][SAMPLE %s] question(raw)  = %s", tag, _clip(q))
    if full_prompt is not None:
        log.info(
            "[Pipeline][SAMPLE %s] full_prompt(after apply_chat_template, "
            "add_generation_prompt=True) =\n%s",
            tag,
            _clip(full_prompt),
        )
    if show_answer:
        log.info(
            "[Pipeline][SAMPLE %s] answer         = %s",
            tag,
            _clip(sample.get("answer")),
        )
    log.info("[Pipeline][SAMPLE %s] -------- end ----------", tag)


def _find_sample_by_source(path: str, source: str) -> dict:
    """在合并后的 train_data 中找出指定 source 的第一条，便于日志各举一例。"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        for item in data:
            if isinstance(item, dict) and item.get("source") == source:
                return item
    except Exception:
        return None
    return None


# Pipeline 专用 logger：只往 FileHandler 写，propagate=False 保证不会冒到 root → console
_pipeline_logger = logging.getLogger("a_token_sdcl_pipeline")
_pipeline_logger.setLevel(logging.INFO)
_pipeline_logger.propagate = False  # 阻断到 root 的传递，避免控制台双打/打印

# 数据样例专用 logger：拆出来，单独写到 pipeline_samples.log，避免污染主流程日志
_samples_logger = logging.getLogger("a_token_sdcl_pipeline.samples")
_samples_logger.setLevel(logging.INFO)
_samples_logger.propagate = False


def _attach_pipeline_log_file(log_path: str, samples_log_path: str = None):
    """构造 pipeline 专用 FileHandler，并把 main / a_token_sdcl / a_token_sdcl_train
    这几个 logger 的输出都引流到 `log_path`、且**不**冒到控制台。

    数据样例（_log_sample / 首 token 池预览）走独立的 `_samples_logger`，
    单独写到 `samples_log_path`（默认 `<log_path 同目录>/pipeline_samples.log`），
    与主 pipeline 日志分开，便于翻阅。

    返回 (handler, samples_handler, undo_fn)。
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.setLevel(logging.INFO)

    if samples_log_path is None:
        samples_log_path = os.path.join(
            os.path.dirname(os.path.abspath(log_path)) or ".",
            "pipeline_samples.log",
        )
    os.makedirs(
        os.path.dirname(os.path.abspath(samples_log_path)) or ".", exist_ok=True
    )
    sfh = logging.FileHandler(samples_log_path, encoding="utf-8")
    sfh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    sfh.setLevel(logging.INFO)
    _samples_logger.addHandler(sfh)

    # 这些 logger 是 pipeline 数据流相关的；都关掉 propagate（避免冒到 root → console），
    # 单独挂同一个 FileHandler。pipeline 结束后再还原 propagate / 卸 handler。
    silent_names = [
        "a_token_sdcl_pipeline",
        "scripts.train.a_token_sdcl",
        "scripts.train.a_token_sdcl_train",
    ]
    # 训练框架自带的 logger：要保留控制台输出（这样 epoch / step / loss 能实时看到），
    # 只额外把同一份内容写进 pipeline 日志文件 → propagate 不动，仅 addHandler。
    mirror_names = [
        "transformers",
        "transformers.trainer",
        "accelerate",
        "peft",
    ]
    saved_states = []  # (logger, prev_propagate, changed_propagate)
    for name in silent_names:
        lg = logging.getLogger(name)
        saved_states.append((lg, lg.propagate, True))
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg.addHandler(fh)
    for name in mirror_names:
        lg = logging.getLogger(name)
        saved_states.append((lg, lg.propagate, False))
        # 保留 propagate 原状，只确保 level 够低 + 挂上 FileHandler
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
        lg.addHandler(fh)

    def _undo():
        for lg, prev_propagate, changed_propagate in saved_states:
            try:
                lg.removeHandler(fh)
            except Exception:
                pass
            if changed_propagate:
                lg.propagate = prev_propagate
        try:
            _samples_logger.removeHandler(sfh)
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
        try:
            sfh.close()
        except Exception:
            pass

    return fh, sfh, _undo


def run_a_token_sdcl_pipeline(
    mistake_path: str = None,
    corr_answer_path: str = None,
    fill_correct_path: str = None,
    train_data_path: str = None,
    output_dir: str = None,
    first_token_list_path: str = None,
    pipeline_log_path: str = None,
    roll_n: int = 16,
    fill_max_gen_token: int = 4096,
    fill_prompt_len: int = 1024,
    fill_epoch: int = 3,
    train_num_epochs: int = 3,
    train_learning_rate: float = 1e-5,
    train_batch_size: int = 4,
    train_grad_accum_steps: int = 4,
    train_max_prompt_length: int = 1024,
    train_max_answer_length: int = 4096,
    train_use_lora: bool = True,
    train_ce_weight: float = 1.0,
    skip_fill: bool = False,
    skip_merge: bool = False,
    skip_train: bool = False,
    fill_device_ids: list = None,
    train_device_ids: list = None,
    seed: int = 42,
):
    """串联调用方法 1/2/3：
        1) generate_fill_correct  → fill_correct.json
        2) merge_to_train_data    → a_token_train_data.json
        3) train_a_token_sdcl     → 混合蒸馏训练 LoRA checkpoint

    所有路径不传则使用 datasets/exam/ 下的默认文件名。
    传 skip_* 可单独跑某一步（例如已经有 fill_correct.json 时跳过 step1）。

    数据流日志：
      每个阶段开始 / 结束都会向 `pipeline_log_path` 落盘一条 INFO 级日志，
      记录该阶段的输入文件、输出文件、条数变化，方便事后审计。
      默认 log 路径为 `<output_dir>/pipeline_dataflow.log`。

    Args:
        mistake_path           : mistake_DS_MATH.json，方法 1 输入
        corr_answer_path       : corr_answer.json，方法 2 输入
        fill_correct_path      : fill_correct.json，方法 1 输出 / 方法 2 输入
        train_data_path        : a_token_train_data.json，方法 2 输出 / 方法 3 输入
        output_dir             : 训练输出目录（LoRA + 日志）
        first_token_list_path  : 候选首 token 池
        pipeline_log_path      : pipeline 数据流日志文件路径（默认放到 output_dir 下）
        roll_n                 : 方法 1 每题随机抽多少个候选
        fill_*                 : 方法 1 vLLM 相关
        train_*                : 方法 3 训练超参
        skip_fill/merge/train  : 跳过对应阶段
        fill_device_ids        : 方法 1 用哪些 GPU（数据并行）
        train_device_ids       : 方法 3 用哪些 GPU（学生 + 教师分卡）
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    exam_dir = os.path.join(project_root, "datasets", "exam")

    if mistake_path is None:
        mistake_path = os.path.join(exam_dir, "mistake_DS_MATH.json")
    if corr_answer_path is None:
        corr_answer_path = os.path.join(exam_dir, "corr_answer_4096.json")
    if fill_correct_path is None:
        fill_correct_path = os.path.join(exam_dir, "fill_correct.json")
    if train_data_path is None:
        train_data_path = os.path.join(exam_dir, "a_token_train_data.json")
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "output", f"a_token_sdcl_{ts}")
    if first_token_list_path is None:
        first_token_list_path = os.path.join(
            project_root, "datasets", "first_tokens_test.json"
        )
    if pipeline_log_path is None:
        os.makedirs(output_dir, exist_ok=True)
        pipeline_log_path = os.path.join(output_dir, "pipeline_dataflow.log")
    samples_log_path = os.path.join(
        os.path.dirname(os.path.abspath(pipeline_log_path)) or ".",
        "pipeline_samples.log",
    )

    pipeline_started = datetime.now()
    _, _, _undo_log_attach = _attach_pipeline_log_file(
        pipeline_log_path, samples_log_path
    )
    pl = _pipeline_logger  # 整条流水所有日志走这个 logger（仅写文件，不打控制台）
    sl = _samples_logger  # 数据样例走 samples 专用 logger（写到 pipeline_samples.log）
    try:
        # 加载一次 tokenizer，用于把 sample 渲染成「套模板后的真实 prompt」。
        # 放在 try 内部，确保即便加载失败抛出非 Exception 也会走 finally 卸载 FileHandler。
        try:
            sample_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        except Exception as e:
            pl.warning(
                "[Pipeline] tokenizer 加载失败，sample 日志将只记录 raw question：%s",
                e,
            )
            sample_tokenizer = None

        pl.info("#" * 78)
        pl.info(
            "[Pipeline] START at %s", pipeline_started.isoformat(timespec="seconds")
        )
        pl.info("[Pipeline] log file       = %s", pipeline_log_path)
        pl.info("[Pipeline] samples log    = %s", samples_log_path)
        pl.info("[Pipeline] model_path     = %s", model_path)
        pl.info(
            "[Pipeline] mistake_path   = %s (items=%d)",
            mistake_path,
            _count_json_items(mistake_path),
        )
        pl.info(
            "[Pipeline] corr_answer    = %s (items=%d)",
            corr_answer_path,
            _count_json_items(corr_answer_path),
        )
        pl.info("[Pipeline] first_tokens   = %s", first_token_list_path)
        pl.info("[Pipeline] fill_correct   = %s", fill_correct_path)
        pl.info("[Pipeline] train_data     = %s", train_data_path)
        pl.info("[Pipeline] output_dir     = %s", output_dir)
        pl.info(
            "[Pipeline] roll_n=%d fill_prompt_len=%d fill_max_gen=%d "
            "train_bs=%d gas=%d epochs=%d lr=%g ce_w=%g lora=%s seed=%d",
            roll_n,
            fill_prompt_len,
            fill_max_gen_token,
            train_batch_size,
            train_grad_accum_steps,
            train_num_epochs,
            train_learning_rate,
            train_ce_weight,
            train_use_lora,
            seed,
        )
        pl.info(
            "[Pipeline] fill_device_ids=%s  train_device_ids=%s",
            fill_device_ids,
            train_device_ids,
        )
        pl.info("#" * 78)

        # ── 方法 1：随机首 token 填充 → fill_correct.json (fill_epoch 轮迭代) ──
        if not skip_fill:
            n_mistake_in = _count_json_items(mistake_path)
            t0 = datetime.now()
            pl.info("=" * 60)
            pl.info(
                "[Pipeline] Step 1/3: generate_fill_correct  START %s",
                t0.isoformat(timespec="seconds"),
            )
            pl.info(
                "[Pipeline]   IN : mistake_path=%s (items=%d)",
                mistake_path,
                n_mistake_in,
            )
            pl.info("[Pipeline]   IN : first_token_list=%s", first_token_list_path)
            pl.info("[Pipeline]   OUT: fill_correct_path=%s", fill_correct_path)
            pl.info(
                "[Pipeline]   fill_epoch=%d (累积并集 / 连续 2 轮无新增则提前停)",
                fill_epoch,
            )
            pl.info("=" * 60)
            # 输入侧样例：mistake[0] 套模板后的完整 prompt + 候选 first_token 池前几条
            _log_sample(
                "Step1.IN.mistake[0]",
                _load_first_sample(mistake_path),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            ft_pool_preview = _peek_first_token_pool(first_token_list_path, n=8)
            sl.info(
                "[Pipeline][SAMPLE Step1.IN.first_tokens] preview(top8)=%s",
                ft_pool_preview,
            )

            # 多轮迭代核心:每轮在"未救回题"上重跑 fill,累积并集
            with open(mistake_path, "r", encoding="utf-8") as _f:
                full_mistake_data = json.load(_f)
            full_mistake_idx_set = {
                d.get("question_idx", i) for i, d in enumerate(full_mistake_data)
            }

            accumulated_by_idx: dict = {}  # question_idx → fill_correct 条目
            consecutive_no_gain = 0
            tmp_dir = os.path.join(
                os.path.dirname(fill_correct_path), "_fill_rounds_tmp"
            )
            os.makedirs(tmp_dir, exist_ok=True)

            for round_idx in range(1, max(fill_epoch, 1) + 1):
                solved_idx_set = set(accumulated_by_idx.keys())
                pending_idx_set = full_mistake_idx_set - solved_idx_set
                if not pending_idx_set:
                    pl.info(
                        "[Pipeline][fill round %d] 所有题都已救回,跳出循环", round_idx
                    )
                    break

                # 写当前轮的 mistake 子集到临时文件,供 generate_fill_correct 读
                round_mistake = [
                    d
                    for i, d in enumerate(full_mistake_data)
                    if d.get("question_idx", i) in pending_idx_set
                ]
                round_in_path = os.path.join(tmp_dir, f"mistake_round{round_idx}.json")
                round_out_path = os.path.join(tmp_dir, f"fill_round{round_idx}.json")
                with open(round_in_path, "w", encoding="utf-8") as _f:
                    json.dump(round_mistake, _f, ensure_ascii=False)

                pl.info(
                    "[Pipeline][fill round %d/%d] pending=%d, seed=%d",
                    round_idx,
                    fill_epoch,
                    len(round_mistake),
                    seed + round_idx - 1,
                )
                round_t0 = datetime.now()
                generate_fill_correct(
                    model_path=model_path,
                    mistake_path=round_in_path,
                    output_path=round_out_path,
                    first_token_list_path=first_token_list_path,
                    roll_n=roll_n,
                    max_gen_token=fill_max_gen_token,
                    prompt_len=fill_prompt_len,
                    device_ids=fill_device_ids,
                    seed=seed + round_idx - 1,
                )
                round_dt = datetime.now() - round_t0

                with open(round_out_path, "r", encoding="utf-8") as _f:
                    round_results = json.load(_f)

                # 合并到累积 dict(以 question_idx 为键,先到先得)
                gain = 0
                for it in round_results:
                    qid = it.get("question_idx")
                    if qid is None:
                        continue
                    if qid not in accumulated_by_idx:
                        accumulated_by_idx[qid] = it
                        gain += 1

                pl.info(
                    "[Pipeline][fill round %d/%d] DONE duration=%s "
                    "round_solved=%d, new_gain=%d, accumulated=%d, still_pending=%d",
                    round_idx,
                    fill_epoch,
                    str(round_dt).split(".")[0],
                    len(round_results),
                    gain,
                    len(accumulated_by_idx),
                    n_mistake_in - len(accumulated_by_idx),
                )

                # 提前停止:连续 2 轮零新增
                if gain == 0:
                    consecutive_no_gain += 1
                    if consecutive_no_gain >= 2:
                        pl.info(
                            "[Pipeline][fill round %d] 连续 %d 轮无新增,提前停止",
                            round_idx,
                            consecutive_no_gain,
                        )
                        break
                else:
                    consecutive_no_gain = 0

            # 把累积并集一次性写到最终 fill_correct.json
            final_list = list(accumulated_by_idx.values())
            with open(fill_correct_path, "w", encoding="utf-8") as _f:
                json.dump(final_list, _f, ensure_ascii=False)

            n_fill_out = _count_json_items(fill_correct_path)
            t1 = datetime.now()
            hit_rate = (
                f"{(n_fill_out / n_mistake_in * 100):.2f}%"
                if n_mistake_in > 0 and n_fill_out >= 0
                else "N/A"
            )
            pl.info(
                "[Pipeline] Step 1/3: generate_fill_correct  DONE  "
                "duration=%s  mistakes_in=%d → fill_correct_out=%d  hit_rate=%s",
                str(t1 - t0).split(".")[0],
                n_mistake_in,
                n_fill_out,
                hit_rate,
            )
            # 输出侧样例：fill_correct[0]，包含 fill_token_id/text + 完整 answer
            _log_sample(
                "Step1.OUT.fill_correct[0]",
                _load_first_sample(fill_correct_path),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
        else:
            pl.info(
                "[Pipeline] Step 1/3: SKIPPED (skip_fill=True)  "
                "fill_correct_path=%s (items=%d)",
                fill_correct_path,
                _count_json_items(fill_correct_path),
            )
            _log_sample(
                "Step1.SKIPPED.fill_correct[0]",
                _load_first_sample(fill_correct_path),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )

        # ── 方法 2：合并 corr_answer + fill_correct → a_token_train_data.json ─
        if not skip_merge:
            n_corr_in = _count_json_items(corr_answer_path)
            n_fill_in = _count_json_items(fill_correct_path)
            t0 = datetime.now()
            pl.info("=" * 60)
            pl.info(
                "[Pipeline] Step 2/3: merge_to_train_data  START %s",
                t0.isoformat(timespec="seconds"),
            )
            pl.info(
                "[Pipeline]   IN : corr_answer=%s (items=%d)",
                corr_answer_path,
                n_corr_in,
            )
            pl.info(
                "[Pipeline]   IN : fill_correct=%s (items=%d)",
                fill_correct_path,
                n_fill_in,
            )
            pl.info("[Pipeline]   OUT: train_data=%s", train_data_path)
            pl.info("=" * 60)
            # 输入侧样例：corr_answer[0] 与 fill_correct[0]
            _log_sample(
                "Step2.IN.corr_answer[0]",
                _load_first_sample(corr_answer_path),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            _log_sample(
                "Step2.IN.fill_correct[0]",
                _load_first_sample(fill_correct_path),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            merge_to_train_data(
                corr_answer_path=corr_answer_path,
                fill_correct_path=fill_correct_path,
                output_path=train_data_path,
                dedup=True,
            )
            stat = _summarize_train_data(train_data_path)
            t1 = datetime.now()
            pl.info(
                "[Pipeline] Step 2/3: merge_to_train_data  DONE  duration=%s  "
                "total=%d (corr_answer=%d + fill_correct=%d, other=%d)  "
                "[输入合计=%d，去重后=%d]",
                str(t1 - t0).split(".")[0],
                stat["total"],
                stat["n_corr_answer"],
                stat["n_fill_correct"],
                stat["n_other"],
                max(n_corr_in, 0) + max(n_fill_in, 0),
                stat["total"],
            )
            # 输出侧样例：合并后两类 source 各取一条
            _log_sample(
                "Step2.OUT.train_data.corr_answer",
                _find_sample_by_source(train_data_path, "corr_answer"),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            _log_sample(
                "Step2.OUT.train_data.fill_correct",
                _find_sample_by_source(train_data_path, "fill_correct"),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
        else:
            stat = _summarize_train_data(train_data_path)
            pl.info(
                "[Pipeline] Step 2/3: SKIPPED (skip_merge=True)  "
                "train_data=%s total=%d (corr=%d, fill=%d)",
                train_data_path,
                stat["total"],
                stat["n_corr_answer"],
                stat["n_fill_correct"],
            )
            _log_sample(
                "Step2.SKIPPED.train_data.corr_answer",
                _find_sample_by_source(train_data_path, "corr_answer"),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            _log_sample(
                "Step2.SKIPPED.train_data.fill_correct",
                _find_sample_by_source(train_data_path, "fill_correct"),
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )

        # ── 方法 3：混合蒸馏训练 ─────────────────────────────────────────────
        if not skip_train:
            stat = _summarize_train_data(train_data_path)
            t0 = datetime.now()
            pl.info("=" * 60)
            pl.info(
                "[Pipeline] Step 3/3: train_a_token_sdcl  START %s",
                t0.isoformat(timespec="seconds"),
            )
            pl.info(
                "[Pipeline]   IN : train_data=%s total=%d (corr=%d, fill=%d)",
                train_data_path,
                stat["total"],
                stat["n_corr_answer"],
                stat["n_fill_correct"],
            )
            pl.info("[Pipeline]   IN : teacher/init model=%s", model_path)
            pl.info("[Pipeline]   OUT: output_dir=%s", output_dir)
            pl.info("=" * 60)
            # 训练侧样例：与 collator 实际看到的两类样本完全一致——附 token 长度
            corr_sample = _find_sample_by_source(train_data_path, "corr_answer")
            fill_sample = _find_sample_by_source(train_data_path, "fill_correct")
            _log_sample(
                "Step3.IN.corr_answer",
                corr_sample,
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            if sample_tokenizer is not None and corr_sample is not None:
                p = _build_chat_prompt(
                    sample_tokenizer, corr_sample.get("question", "")
                )
                a = corr_sample.get("answer", "") or ""
                p_ids = sample_tokenizer.encode(p, add_special_tokens=False)
                a_ids = sample_tokenizer.encode(a, add_special_tokens=False)
                sl.info(
                    "[Pipeline][SAMPLE Step3.IN.corr_answer] approx tokens "
                    "(main.py 这边用 encode(add_special_tokens=False) 估算，仅作数量级参考；"
                    "训练侧实际 tokenize 以 collator 为准): "
                    "prompt_len≈%d answer_len≈%d total≈%d",
                    len(p_ids),
                    len(a_ids),
                    len(p_ids) + len(a_ids),
                )
            _log_sample(
                "Step3.IN.fill_correct",
                fill_sample,
                tokenizer=sample_tokenizer,
                show_answer=True,
                log=sl,
            )
            if sample_tokenizer is not None and fill_sample is not None:
                p = _build_chat_prompt(
                    sample_tokenizer, fill_sample.get("question", "")
                )
                a = fill_sample.get("answer", "") or ""
                p_ids = sample_tokenizer.encode(p, add_special_tokens=False)
                a_ids = sample_tokenizer.encode(a, add_special_tokens=False)
                sl.info(
                    "[Pipeline][SAMPLE Step3.IN.fill_correct] approx tokens "
                    "(main.py 这边用 encode 估算；训练侧 collator 的 tokenize 方式可能略有差异): "
                    "prompt_len≈%d answer_len≈%d total≈%d  "
                    "fill_token_id=%s fill_token_text=%r  "
                    "(训练侧会在 prompt 之后的第一个生成 token 位置改用 CE(fill_token_id) 替代 KL)",
                    len(p_ids),
                    len(a_ids),
                    len(p_ids) + len(a_ids),
                    fill_sample.get("fill_token_id"),
                    fill_sample.get("fill_token_text"),
                )
            train_a_token_sdcl(
                model_path=model_path,
                data_path=train_data_path,
                output_dir=output_dir,
                num_epochs=train_num_epochs,
                learning_rate=train_learning_rate,
                batch_size=train_batch_size,
                gradient_accumulation_steps=train_grad_accum_steps,
                max_prompt_length=train_max_prompt_length,
                max_answer_length=train_max_answer_length,
                use_lora=train_use_lora,
                ce_weight=train_ce_weight,
                seed=seed,
                device_ids=train_device_ids,
            )
            t1 = datetime.now()
            pl.info(
                "[Pipeline] Step 3/3: train_a_token_sdcl  DONE  duration=%s  "
                "checkpoints/LoRA written to %s",
                str(t1 - t0).split(".")[0],
                output_dir,
            )
        else:
            pl.info("[Pipeline] Step 3/3: SKIPPED (skip_train=True)")

        pipeline_finished = datetime.now()
        pl.info("#" * 78)
        pl.info(
            "[Pipeline] DONE  total_duration=%s",
            str(pipeline_finished - pipeline_started).split(".")[0],
        )
        pl.info("[Pipeline]   data flow summary：")
        pl.info("[Pipeline]     mistake (%d) ─┐", _count_json_items(mistake_path))
        pl.info(
            "[Pipeline]                    ├──► fill_correct (%d)",
            _count_json_items(fill_correct_path),
        )
        pl.info("[Pipeline]     first_tokens ─┘")
        stat = _summarize_train_data(train_data_path)
        pl.info(
            "[Pipeline]     corr_answer (%d) + fill_correct (%d) "
            "──► train_data (%d: corr=%d, fill=%d)",
            _count_json_items(corr_answer_path),
            _count_json_items(fill_correct_path),
            stat["total"],
            stat["n_corr_answer"],
            stat["n_fill_correct"],
        )
        pl.info("[Pipeline]     train_data ──► trained model @ %s", output_dir)
        pl.info("#" * 78)
        # 控制台只留一句指引，告诉用户日志在哪
        print(f"[a_token_sdcl pipeline] 数据流日志已写入 {pipeline_log_path}")
        print(f"[a_token_sdcl pipeline] 数据样例日志已写入 {samples_log_path}")
        return {
            "fill_correct_path": fill_correct_path,
            "train_data_path": train_data_path,
            "output_dir": output_dir,
            "pipeline_log_path": pipeline_log_path,
            "samples_log_path": samples_log_path,
        }
    finally:
        # 卸下 FileHandler 并恢复各 logger 的 propagate 状态，避免重复调用时多次挂载
        _undo_log_attach()


def _judge_correct(pred_answer: str, ref_answer: str) -> bool:
    """与 main.py 内已有逻辑一致：boxed 抽取 + 归一化后字符串相等判正确。"""
    from utils.data_utils import extract_boxed_content, normalize_answer

    return normalize_answer(extract_boxed_content(pred_answer)) == normalize_answer(
        ref_answer
    )


def _load_math500_as_eval_items(math500_path: str):
    """读 HuggingFaceH4/MATH-500(HF arrow 目录)成 run_eval 需要的格式。

    返回 [{"question": ..., "ref_answer": ..., "ref_solution": ...}, ...]。
    用 pyarrow 直接读,绕开 `from datasets import ...`(项目根目录里有同名
    `datasets/` 包会抢占 import)。

    支持两种输入:
      1) 目录:datasets/data/MATH-500   ← 默认,内含 data-*.arrow
      2) 文件:.arrow / .jsonl 直接读
    """
    import glob
    import json as _json
    import pyarrow.ipc as _pa_ipc

    if os.path.isdir(math500_path):
        arrow_files = sorted(glob.glob(os.path.join(math500_path, "data-*.arrow")))
        if not arrow_files:
            raise FileNotFoundError(
                f"math500 目录 {math500_path} 下没找到 data-*.arrow"
            )
        rows = []
        for f in arrow_files:
            with _pa_ipc.open_stream(f) as r:
                tbl = r.read_all()
            rows.extend(tbl.to_pylist())
    elif math500_path.endswith(".arrow"):
        with _pa_ipc.open_stream(math500_path) as r:
            rows = r.read_all().to_pylist()
    elif math500_path.endswith(".jsonl"):
        rows = []
        with open(math500_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
    else:
        raise ValueError(f"无法识别的 math500_path: {math500_path}")

    items = []
    for d in rows:
        # MATH-500 schema: problem / solution / answer
        items.append(
            {
                "question": d.get("problem") or d.get("question"),
                "ref_solution": d.get("solution") or d.get("ref_solution") or "",
                "ref_answer": d.get("answer") or d.get("ref_answer"),
            }
        )
    return items


def run_eval(
    model_path: str,
    adapter_path: str = None,
    mistake_path: str = "datasets/exam/mistake_DS_MATH.json",
    corr_path: str = "datasets/exam/corr_answer_4096.json",
    math500_path: str = "datasets/data/MATH-500",
    device_ids: list = None,
    max_seq_length: int = 4096,
    output_dir: str = None,
):
    """一次性评测训练后(或 baseline)模型,产出 mistake / corr / all 三组指标。

    实现要点(榨干多卡):
      - mistake 与 corr 合并成单一题集,**只调用一次 exam_multi_gpu**;
        所有可见 GPU 在同一次 vLLM spawn 内并行跑,不重复加载模型 / adapter。
      - 通过题号上加偏移区分两个集合的来源,推理结束后再按来源拆回三组指标。
      - mistake 在前、corr 在后(顺序无所谓,exam_multi_gpu 内部按 round-robin
        分到各 worker,长尾自然摊平到所有卡)。

    指标释义:
      - mistake 集准确率 = "纠错率"(初始模型这些题做错,训后能救回多少)
      - corr 集准确率   = "保留率"(初始模型这些题做对,训后是否退化)
      - all 全量        = mistake ∪ corr 拼接后的综合准确率

    Args:
        model_path   : 基座模型路径(必传)。
        adapter_path : LoRA adapter 路径;为 None 时跑 baseline(无 LoRA)。
        mistake_path : mistake 数据 json。
        corr_path    : corr 数据 json。
        device_ids   : 多卡 id;默认 = 全部可见 GPU。
        output_dir   : 评测结果落盘目录;为 None 则用 output/eval_<tag>_<ts>。

    Returns:
        dict: 三组指标的简要汇总(同时落盘 summary.json + 三份 items_*.jsonl)。
    """
    import json
    import time
    import torch as _torch

    if device_ids is None:
        n = _torch.cuda.device_count()
        if n <= 0:
            raise RuntimeError("未检测到可见 GPU。")
        device_ids = list(range(n))

    if output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "baseline" if not adapter_path else "lora"
        output_dir = os.path.join("output", f"eval_{tag}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[run_eval] model_path   = {model_path}")
    print(f"[run_eval] adapter_path = {adapter_path or '<NONE / baseline>'}")
    print(f"[run_eval] device_ids   = {device_ids}")
    print(f"[run_eval] output_dir   = {output_dir}")

    with open(mistake_path, "r", encoding="utf-8") as f:
        mistake_data = json.load(f)
    with open(corr_path, "r", encoding="utf-8") as f:
        corr_data = json.load(f)

    # math500 默认开启,路径不存在或加载失败时降级为不跑(并打 warning)
    math500_data = []
    if math500_path:
        try:
            math500_data = _load_math500_as_eval_items(math500_path)
        except Exception as e:
            print(
                f"[run_eval] WARN: 加载 math500 失败({math500_path}): {e}"
                f";本次评测跳过 math500"
            )
            math500_data = []

    print(
        f"[run_eval] mistake n={len(mistake_data)}, corr n={len(corr_data)}, "
        f"math500 n={len(math500_data)}, "
        f"合并题集 n={len(mistake_data) + len(corr_data) + len(math500_data)}"
    )

    # ── 合并三份数据,带 source 标签 ───────────────────────────────────────
    merged = []
    for i, d in enumerate(mistake_data):
        merged.append(
            {
                "question": d["question"],
                "ref_solution": d.get("ref_solution", ""),
                "ref_answer": d["ref_answer"],
                "_source": "mistake",
                "_local_idx": i,
            }
        )
    for i, d in enumerate(corr_data):
        merged.append(
            {
                "question": d["question"],
                "ref_solution": d.get("ref_solution", ""),
                "ref_answer": d["ref_answer"],
                "_source": "corr",
                "_local_idx": i,
            }
        )
    for i, d in enumerate(math500_data):
        merged.append(
            {
                "question": d["question"],
                "ref_solution": d.get("ref_solution", ""),
                "ref_answer": d["ref_answer"],
                "_source": "math500",
                "_local_idx": i,
            }
        )

    questions = [d["question"] for d in merged]
    solutions = [d["ref_solution"] for d in merged]
    answers = [d["ref_answer"] for d in merged]
    # 用合并后的全局 idx 作为 question_idx,便于按位置回查
    qidxs = list(range(len(merged)))

    use_lora = bool(adapter_path)
    take_exam_obj = TakeExam(
        model_path=model_path,
        use_lora=use_lora,
        adapter_path=adapter_path,
        max_seq_length=max_seq_length,
    )

    print(
        f"[run_eval] 单次 exam_multi_gpu 跑完 {len(questions)} 题"
        f"(GPU {device_ids},一次加载,持续吃满)"
    )
    t0 = time.time()
    results = take_exam_obj.exam_multi_gpu(
        questions,
        solutions,
        answers,
        qidxs,
        device_ids=device_ids,
        write_output=False,
    )
    dt = time.time() - t0
    print(
        f"[run_eval] 推理结束,耗时 {dt:.1f}s,平均 {dt / max(len(questions), 1):.3f}s/题"
    )

    # ── 按 _source 拆回各组指标 ────────────────────────────────────────────
    # results 顺序与 questions 一致(exam_multi_gpu 在收尾合并时按 question_idx 回排),
    # 但稳妥起见,通过 question_idx 回查 merged。
    by_source = {"mistake": [], "corr": [], "math500": []}
    for r in results:
        gidx = r.get("question_idx")
        if not isinstance(gidx, int) or gidx < 0 or gidx >= len(merged):
            # 极端兜底:索引异常时,视作 "all" 内的散点,跳过分组判定
            continue
        src = merged[gidx]["_source"]
        ok = _judge_correct(r["answer"], r["ref_answer"])
        by_source[src].append(
            {
                "question_idx": merged[gidx]["_local_idx"],
                "is_correct": ok,
                "ref_answer": r.get("ref_answer"),
                "answer": r.get("answer", ""),
            }
        )

    # all 仍按"原 mistake ∪ corr"口径,与历史 baseline 数字可比;
    # math500 是独立指标,不并入 all,避免改变 all 的分母。
    items_all = list(by_source["mistake"]) + list(by_source["corr"])

    summary = {
        "mistake": {
            "n_total": len(by_source["mistake"]),
            "n_correct": sum(1 for it in by_source["mistake"] if it["is_correct"]),
        },
        "corr": {
            "n_total": len(by_source["corr"]),
            "n_correct": sum(1 for it in by_source["corr"] if it["is_correct"]),
        },
        "math500": {
            "n_total": len(by_source["math500"]),
            "n_correct": sum(1 for it in by_source["math500"] if it["is_correct"]),
        },
        "all": {
            "n_total": len(items_all),
            "n_correct": sum(1 for it in items_all if it["is_correct"]),
        },
    }
    for k, s in summary.items():
        s["accuracy"] = (
            (s["n_correct"] / s["n_total"] * 100.0) if s["n_total"] > 0 else 0.0
        )

    # ── 落盘 ────────────────────────────────────────────────────────────────
    items_map = {
        "mistake": by_source["mistake"],
        "corr": by_source["corr"],
        "math500": by_source["math500"],
        "all": items_all,
    }
    for label, items in items_map.items():
        with open(
            os.path.join(output_dir, f"items_{label}.jsonl"), "w", encoding="utf-8"
        ) as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    summary_brief = dict(summary)
    summary_brief["model_path"] = model_path
    summary_brief["adapter_path"] = adapter_path
    summary_brief["device_ids"] = device_ids
    summary_brief["elapsed_seconds"] = dt
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_brief, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"📊 EVAL SUMMARY  ({'LoRA' if use_lora else 'BASELINE'})")
    print("=" * 60)
    for label in ("mistake", "corr", "math500", "all"):
        s = summary[label]
        if s["n_total"] == 0:
            print(f"  {label:8s}: (skipped, n_total=0)")
        else:
            print(
                f"  {label:8s}: {s['n_correct']}/{s['n_total']} = {s['accuracy']:.2f}%"
            )
    print("=" * 60)
    print(f"详细结果已落盘: {output_dir}")

    return summary_brief


def _cli_run_eval():
    """CLI 入口,从命令行启动 run_eval。"""
    import argparse

    p = argparse.ArgumentParser(
        description="一次性评测 a_token_sdcl 训练后(或 baseline)模型的 mistake / corr / all 三组准确率。"
    )
    p.add_argument("--model_path", type=str, required=True, help="基座模型路径")
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA adapter 路径;不传则跑 baseline(无 LoRA)",
    )
    p.add_argument(
        "--mistake_path", type=str, default="datasets/exam/mistake_DS_MATH.json"
    )
    p.add_argument(
        "--corr_path", type=str, default="datasets/exam/corr_answer_4096.json"
    )
    p.add_argument(
        "--math500_path",
        type=str,
        default="datasets/data/MATH-500",
        help="MATH-500 数据(HF arrow 目录 / .arrow / .jsonl);传空字符串可跳过",
    )
    p.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU id,例如 '0,1,2';不传则用全部可见 GPU",
    )
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    device_ids = None
    if args.device_ids:
        device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]

    run_eval(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        mistake_path=args.mistake_path,
        corr_path=args.corr_path,
        math500_path=args.math500_path or None,
        device_ids=device_ids,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )


def _cli_run_pipeline():
    """CLI 入口,从命令行启动 run_a_token_sdcl_pipeline,可指定跳过哪几步。"""
    import argparse

    p = argparse.ArgumentParser(
        description="a_token_sdcl pipeline:fill → merge → train(可按需跳过任意一步)。"
    )
    p.add_argument(
        "--skip-fill", action="store_true", help="跳过 Step1 随机首 token 填充"
    )
    p.add_argument("--skip-merge", action="store_true", help="跳过 Step2 合并训练数据")
    p.add_argument("--skip-train", action="store_true", help="跳过 Step3 蒸馏训练")
    p.add_argument("--mistake_path", type=str, default=None)
    p.add_argument("--corr_answer_path", type=str, default=None)
    p.add_argument("--fill_correct_path", type=str, default=None)
    p.add_argument("--train_data_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--roll_n", type=int, default=16)
    p.add_argument(
        "--fill_epoch",
        type=int,
        default=3,
        help="fill 阶段最大轮次,每轮在未救回题上重 roll;连续 2 轮零新增提前停",
    )
    p.add_argument(
        "--fill_max_gen_token",
        type=int,
        default=4096,
        help="fill 阶段每条续写最大 token 数(默认 4096,与 4096 池一致)。",
    )
    p.add_argument(
        "--fill_prompt_len",
        type=int,
        default=1024,
        help="fill 阶段 prompt 最大 token 数(max_model_len = fill_prompt_len + fill_max_gen_token)。",
    )
    p.add_argument(
        "--fill_device_ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU id,fill 阶段用;不传则全部可见 GPU",
    )
    p.add_argument(
        "--train_device_ids",
        type=str,
        default=None,
        help="逗号分隔的 GPU id,train 阶段用;不传则全部可见 GPU",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fill_dev = (
        [int(x) for x in args.fill_device_ids.split(",")]
        if args.fill_device_ids
        else None
    )
    train_dev = (
        [int(x) for x in args.train_device_ids.split(",")]
        if args.train_device_ids
        else None
    )

    run_a_token_sdcl_pipeline(
        mistake_path=args.mistake_path,
        corr_answer_path=args.corr_answer_path,
        fill_correct_path=args.fill_correct_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        roll_n=args.roll_n,
        fill_epoch=args.fill_epoch,
        fill_max_gen_token=args.fill_max_gen_token,
        fill_prompt_len=args.fill_prompt_len,
        skip_fill=args.skip_fill,
        skip_merge=args.skip_merge,
        skip_train=args.skip_train,
        fill_device_ids=fill_dev,
        train_device_ids=train_dev,
        seed=args.seed,
    )


def _cli_run_pipeline_and_train():
    """一键:fill+merge(数据)→ DDP 训练。任意阶段异常都会抛出,
    main 入口的 finally 块会兜底进入 use_worker 挂卡保活。"""
    import argparse
    import os as _os
    import subprocess as _sp
    import sys as _sys

    p = argparse.ArgumentParser(
        description="一键跑数据生成(fill+merge)+ DDP 训练;失败/完成都会进 use_worker 挂卡。"
    )
    # ── 数据(pipeline)阶段参数(透传 run_a_token_sdcl_pipeline) ──
    p.add_argument("--mistake_path", type=str, default=None)
    p.add_argument("--corr_answer_path", type=str, default=None)
    p.add_argument("--fill_correct_path", type=str, default=None)
    p.add_argument("--train_data_path", type=str, default=None)
    p.add_argument(
        "--pipeline_output_dir",
        type=str,
        default=None,
        help="run_a_token_sdcl_pipeline 的 output_dir(数据流日志/中间产物)",
    )
    p.add_argument("--roll_n", type=int, default=16)
    p.add_argument("--fill_epoch", type=int, default=3)
    p.add_argument("--fill_max_gen_token", type=int, default=4096)
    p.add_argument("--fill_prompt_len", type=int, default=1024)
    p.add_argument("--fill_device_ids", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-fill", dest="skip_fill", action="store_true")
    p.add_argument("--skip-merge", dest="skip_merge", action="store_true")

    # ── 训练阶段参数 ──
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练 base 模型路径,直接传给 run_a_token_sdcl_train.py",
    )
    p.add_argument(
        "--train_output_dir", type=str, required=True, help="DDP 训练输出目录"
    )
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument(
        "--no-gradient_checkpointing",
        dest="no_gradient_checkpointing",
        action="store_true",
        default=True,
        help="默认关闭 gradient_checkpointing(显存够用时更快)",
    )
    p.add_argument(
        "--gradient_checkpointing",
        dest="no_gradient_checkpointing",
        action="store_false",
        help="显式开 gradient_checkpointing(OOM 时用)",
    )
    p.add_argument("--master_port", type=str, default="29500")
    p.add_argument(
        "--train_extra_args",
        type=str,
        default="",
        help='额外透传给训练脚本的 raw 参数串,例如 "--no_ema --max_answer_length 3072"',
    )

    args, _unknown = p.parse_known_args()

    # ── Stage 1+2:数据 ──
    print("=" * 60, flush=True)
    print("[pipeline_and_train] Stage A:fill + merge", flush=True)
    print("=" * 60, flush=True)
    run_a_token_sdcl_pipeline(
        mistake_path=args.mistake_path,
        corr_answer_path=args.corr_answer_path,
        fill_correct_path=args.fill_correct_path,
        train_data_path=args.train_data_path,
        output_dir=args.pipeline_output_dir,
        roll_n=args.roll_n,
        fill_epoch=args.fill_epoch,
        fill_max_gen_token=args.fill_max_gen_token,
        fill_prompt_len=args.fill_prompt_len,
        skip_fill=args.skip_fill,
        skip_merge=args.skip_merge,
        skip_train=True,  # 训练单独走 DDP launcher
        fill_device_ids=(
            [int(x) for x in args.fill_device_ids.split(",")]
            if args.fill_device_ids
            else None
        ),
        train_device_ids=None,
        seed=args.seed,
    )

    # 数据落盘路径(与 pipeline 默认一致)
    train_data_path = args.train_data_path or _os.path.join(
        "datasets", "exam", "a_token_train_data.json"
    )
    if not _os.path.exists(train_data_path):
        raise FileNotFoundError(
            f"训练数据未生成:{train_data_path};请检查 pipeline 阶段日志。"
        )

    # ── Stage 3:DDP 训练 ──
    print("=" * 60, flush=True)
    print("[pipeline_and_train] Stage B:DDP 训练", flush=True)
    print("=" * 60, flush=True)
    train_script = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "scripts",
        "train",
        "run_a_token_sdcl_train.py",
    )
    cmd = [
        _sys.executable,
        train_script,
        "--master_port",
        args.master_port,
        "--model_path",
        args.model_path,
        "--data_path",
        train_data_path,
        "--output_dir",
        args.train_output_dir,
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--ce_weight",
        str(args.ce_weight),
    ]
    if args.no_gradient_checkpointing:
        cmd.append("--no-gradient_checkpointing")
    if args.train_extra_args.strip():
        import shlex

        cmd.extend(shlex.split(args.train_extra_args))

    print(f"[pipeline_and_train] 训练命令:{' '.join(cmd)}", flush=True)
    rc = _sp.call(cmd)
    if rc != 0:
        raise RuntimeError(f"DDP 训练失败,退出码 {rc}")
    print(
        "[pipeline_and_train] 训练完成,checkpoint 在:",
        args.train_output_dir,
        flush=True,
    )


if __name__ == "__main__":
    import sys

    # 子命令分发:
    #   python main.py                          → 跑完整 fill+merge+train pipeline(默认)
    #   python main.py pipeline ...             → 跑 pipeline 并可附加 --skip-* 等参数
    #   python main.py eval ...                 → 跑评测(可加 --model_path/--adapter_path 等)
    #   python main.py pipeline_and_train ...   → 一键 fill+merge + DDP 训练,异常/完成都进 use_worker
    try:
        # if len(sys.argv) > 1 and sys.argv[1] == "eval":
        #     sys.argv = [sys.argv[0]] + sys.argv[2:]
        #     _cli_run_eval()
        # elif len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        #     sys.argv = [sys.argv[0]] + sys.argv[2:]
        #     _cli_run_pipeline()
        # elif len(sys.argv) > 1 and sys.argv[1] == "pipeline_and_train":
        #     sys.argv = [sys.argv[0]] + sys.argv[2:]
        #     _cli_run_pipeline_and_train()
        # else:
        #     run_a_token_sdcl_pipeline()
        student_take_exam_Math_sub()
        teachr = TeacherCorrecter()
        teachr.teacher_mark_paper_with_save()

    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        # 不论正常结束还是异常,都进入 use_worker 挂卡保活,
        # 方便人工接管 GPU 资源(Ctrl+C 退出)。
        try:
            use_worker()
        except BaseException:
            import traceback

            traceback.print_exc()
    use_worker()

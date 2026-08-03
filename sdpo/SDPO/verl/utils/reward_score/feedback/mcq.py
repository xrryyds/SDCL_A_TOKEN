import re

FORMAT_PENALTY = 0.5
MIN_REASONING_CHARS = 100


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def is_correct_format(text: str) -> bool:
    """
    Check if the text is in the correct XML format.

    The text should contain at the end of the text:
    <answer>
    (A|B|C|D)
    </answer>
    """
    pattern = r"<answer>\s*(A|B|C|D)\s*</answer>$"
    return re.search(pattern, text) is not None


def get_reasoning_content(text: str) -> str:
    """Extract content inside <reasoning>...</reasoning>, or empty string if absent."""
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def has_valid_reasoning(text: str) -> bool:
    """Check if the text has a reasoning block with sufficient content."""
    content = get_reasoning_content(text)
    return len(content) >= MIN_REASONING_CHARS


def compute_score(solution: str, ground_truth: str) -> dict:
    multiple_choice_answer = extract_xml_answer(solution)

    correct = multiple_choice_answer == ground_truth
    reward = float(correct)

    reasoning_content = get_reasoning_content(solution)
    has_reasoning_block = bool(re.search(r"<reasoning>.*?</reasoning>", solution, re.DOTALL))
    reasoning_too_short = has_reasoning_block and len(reasoning_content) < MIN_REASONING_CHARS
    missing_reasoning = not has_reasoning_block

    # Correct answer without sufficient reasoning gets a penalty deduction.
    if correct and (missing_reasoning or reasoning_too_short):
        reward -= FORMAT_PENALTY

    return {
        "score": reward,
        "acc": float(correct),
        "pred": multiple_choice_answer,
        "incorrect_format": 1 if not is_correct_format(solution) else 0,
        "missing_reasoning": 1 if missing_reasoning else 0,
        "short_reasoning": 1 if reasoning_too_short else 0,
        "reasoning_len": len(reasoning_content),
        "feedback": "",
    }

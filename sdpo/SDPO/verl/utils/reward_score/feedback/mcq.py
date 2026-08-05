import re


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

def extract_reasoning(text: str) -> str:
    """Extract reasoning content between <reasoning> tags."""
    if "<reasoning>" not in text:
        return ""
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()

def compute_score(solution: str, ground_truth: str) -> dict:
    multiple_choice_answer = extract_xml_answer(solution)

    correct = float(multiple_choice_answer == ground_truth)
    incorrect_format = is_correct_format(solution)

    # Penalize shortcut answers without sufficient reasoning
    reasoning = extract_reasoning(solution)
    has_reasoning = len(reasoning) >= 50
    reward = correct * (0.5 if not has_reasoning else 1.0)

    return {
      "score": reward,
      "acc": correct,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if incorrect_format else 0,
      "feedback": "",
    }

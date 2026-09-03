"""Build train/test JSON for the local MATH dataset in the format data/preprocess.py expects.

Train comes from the official MATH train split (7500, all levels); the gold answer is the last
\\boxed{...} in the reference solution. Validation uses MATH-500, whose answers are already
extracted, so evaluation numbers stay comparable to the standard benchmark.

Usage:
    python data/build_math_local.py --repo_root /path/to/SDCL_A_TOKEN --out datasets/math
    python -m data.preprocess --data_source datasets/math
"""

import argparse
import json
import os

import datasets

from data.format.prompts import PROMPT
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _boxed_answer(solution: str):
    """Gold answer of a MATH reference solution, or None when there is no usable \\boxed{}."""
    try:
        boxed = last_boxed_only_string(solution)
        if not boxed:
            return None
        answer = remove_boxed(boxed)
        return answer.strip() or None
    except Exception:
        return None


def _row(idx: int, problem: str, answer: str, level, subject) -> dict:
    return {
        "idx": idx,
        "kind": "math",
        # data_source: selects math.compute_score in verl/utils/reward_score/feedback/__init__.py
        "dataset": "math",
        "answer": answer,
        "elo": 1500,
        "prompt": PROMPT.format(problem=problem),
        "description": problem,
        "tests": "-",
        "level": str(level),
        "subject": str(subject),
    }


def build_train(repo_root: str) -> list[dict]:
    rows, dropped = [], 0
    for subject in MATH_SUBJECTS:
        ds = datasets.load_from_disk(os.path.join(repo_root, "datasets/data/MATH/train", subject))
        for ex in ds:
            answer = _boxed_answer(ex["solution"])
            if answer is None:
                dropped += 1
                continue
            rows.append(_row(len(rows), ex["problem"], answer, ex["level"], ex["type"]))
    print(f"train: {len(rows)} rows kept, {dropped} dropped for missing \\boxed{{}}")
    return rows


def build_test(repo_root: str) -> list[dict]:
    ds = datasets.load_from_disk(os.path.join(repo_root, "datasets/data/MATH-500"))
    rows = [
        _row(i, ex["problem"], str(ex["answer"]).strip(), ex["level"], ex["subject"])
        for i, ex in enumerate(ds)
    ]
    print(f"test (MATH-500): {len(rows)} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default="/home/xiongrengrong.xrr/SDCL_A_TOKEN")
    parser.add_argument("--out", default="datasets/math")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split, rows in (("train", build_train(args.repo_root)), ("test", build_test(args.repo_root))):
        path = os.path.join(args.out, f"{split}.json")
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"wrote {path}")
    print("\nnow run:  python -m data.preprocess --data_source", args.out)


if __name__ == "__main__":
    main()

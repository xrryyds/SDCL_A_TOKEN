import logging
import os
from datasets import Dataset
from .load_dataset import LoadDataset
from prompt import QUESTION_PROMPT
from utils import extract_boxed_content

logger = logging.getLogger(__name__)


class DeepMath_103K:
    def __init__(self, train: bool = True):
        """
        DeepMath-103K dataset loader.
        HuggingFace repo: zwhe99/DeepMath-103K
        Fields: 'question', 'final_answer'
        """
        if train:
            target_split = "train"
            local_path = "./datasets/data/DeepMath-103K/train"
        else:
            target_split = "test"
            local_path = "./datasets/data/DeepMath-103K/test"

        logger.info(f"Loading DeepMath-103K (Split: {target_split})...")

        dataset_loader = LoadDataset(
            dataset_name="zwhe99/DeepMath-103K",
            split=target_split,
            local_path=local_path,
            config=None,
        )

        self.problems, self.solutions, self.answers, self.data_len = (
            self.extract_data(dataset_loader.get_dataset())
        )

    def extract_data(self, dataset: Dataset) -> tuple[list, list, list, int]:
        problems = []
        solutions = []
        answers = []

        if dataset is None:
            return [], [], [], 0

        for data in dataset:
            # DeepMath-103K 字段：question / final_answer
            problem = data.get("question", "").strip()
            answer = data.get("final_answer", "").strip()
            # DeepMath-103K 没有独立的 solution 字段，用 final_answer 代替
            solution = answer

            if not answer and solution:
                answer = extract_boxed_content(solution)

            if problem and answer:
                problems.append(problem)
                solutions.append(solution)
                answers.append(answer)

        return problems, solutions, answers, len(problems)

    def gen_prompt(self, data: list, max_token: int = 512):
        for i in range(len(data)):
            data[i] = QUESTION_PROMPT.format(
                max_token=max_token,
                problem_text=data[i],
            )


def main():
    print("Loading DeepMath-103K (train)...")
    dataset = DeepMath_103K(train=True)

    if dataset.data_len == 0:
        print("Error: No data loaded.")
        return

    sep = "=" * 30
    print(f"Total Data Loaded: {dataset.data_len}")
    print(sep)
    print("problem:  " + dataset.problems[0])
    print(sep)
    print("solution: " + dataset.solutions[0])
    print(sep)
    print("answer:   " + dataset.answers[0])
    print(sep)


if __name__ == "__main__":
    main()

"""
提取 solution 列表中所有不同的首个 token，统计出现次数，并保存为 JSON。

核心函数：
    extract_and_save_first_tokens(solutions, tokenizer, output_path)

输出 JSON 格式：
    {
        "total_solutions": 7500,
        "skipped": 3,
        "unique_tokens": 42,
        "tokens": [
            {"token_id": 791, "token_text": "We", "count": 1234},
            {"token_id": 578, "token_text": "The", "count": 567},
            ...
        ]
    }
    tokens 按 count 降序排列。

用法示例：
    from transformers import AutoTokenizer
    from scripts.train.extract_first_tokens import extract_and_save_first_tokens

    tokenizer = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
    solutions = ["We start by...", "The answer is...", "Let x = ...", ...]
    result = extract_and_save_first_tokens(solutions, tokenizer, "./first_tokens.json")
    print(result["unique_tokens"])  # 不同首 token 数
"""

import json
import os
import sys
from collections import Counter
from typing import List, Optional

from transformers import AutoTokenizer


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def extract_first_token(tokenizer: AutoTokenizer, text: str) -> Optional[tuple]:
    """
    从文本中提取首个 token。

    Args:
        tokenizer: HuggingFace tokenizer
        text: 输入文本

    Returns:
        (token_id, token_text) 或 None（文本为空时）
    """
    text = _stringify(text).strip()
    if not text:
        return None
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not token_ids:
        return None
    first_id = token_ids[0]
    first_text = tokenizer.decode([first_id], skip_special_tokens=False)
    return first_id, first_text


def extract_first_tokens_stats(
    solutions: List[str],
    tokenizer: AutoTokenizer,
) -> dict:
    """
    统计 solutions 列表中所有首 token 的出现次数。

    Args:
        solutions: solution 文本列表
        tokenizer: HuggingFace tokenizer

    Returns:
        {
            "total_solutions": int,
            "skipped": int,
            "unique_tokens": int,
            "tokens": [{"token_id": int, "token_text": str, "count": int}, ...]
        }
    """
    counter = Counter()
    id_to_text = {}
    skipped = 0

    for sol in solutions:
        result = extract_first_token(tokenizer, sol)
        if result is None:
            skipped += 1
            continue
        token_id, token_text = result
        counter[token_id] += 1
        id_to_text[token_id] = token_text

    tokens_list = [
        {"token_id": tid, "token_text": id_to_text[tid], "count": cnt}
        for tid, cnt in counter.most_common()
    ]

    return {
        "total_solutions": len(solutions),
        "skipped": skipped,
        "unique_tokens": len(tokens_list),
        "tokens": tokens_list,
    }


def extract_and_save_first_tokens(
    solutions: List[str],
    tokenizer: AutoTokenizer,
    output_path: str,
) -> dict:
    """
    提取所有不同首 token，统计个数，保存为 JSON 文件。

    Args:
        solutions: solution 文本列表（任意数据集均可）
        tokenizer: HuggingFace tokenizer
        output_path: 输出 JSON 文件路径

    Returns:
        统计结果 dict（同时已写入 output_path）
    """
    result = extract_first_tokens_stats(solutions, tokenizer)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{'='*60}")
    print(f"首 token 统计结果：")
    print(f"  总 solution 数：{result['total_solutions']}")
    print(f"  跳过（空 solution）：{result['skipped']}")
    print(f"  不同首 token 数：{result['unique_tokens']}")
    print(f"{'='*60}")
    print(f"Top 20：")
    for i, tok in enumerate(result["tokens"][:20], 1):
        print(
            f"  {i:3d}. id={tok['token_id']:6d}  "
            f"text={tok['token_text']!r:20s}  count={tok['count']}"
        )
    print(f"{'='*60}")
    print(f"已保存到 {output_path}")

    return result


def load_first_tokens(path: str) -> dict:
    """
    加载之前保存的首 token 统计文件。

    Args:
        path: JSON 文件路径

    Returns:
        统计结果 dict，包含 tokens 列表
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_first_token_ids(path: str) -> List[int]:
    """
    从保存的 JSON 中获取所有不同首 token 的 id 列表（按出现频率降序）。

    Args:
        path: JSON 文件路径

    Returns:
        token_id 列表
    """
    data = load_first_tokens(path)
    return [tok["token_id"] for tok in data["tokens"]]


def get_all_first_token_texts(path: str) -> List[str]:
    """
    从保存的 JSON 中获取所有不同首 token 的文本列表（按出现频率降序）。

    Args:
        path: JSON 文件路径

    Returns:
        token_text 列表
    """
    data = load_first_tokens(path)
    return [tok["token_text"] for tok in data["tokens"]]


# ---- 命令行入口 ----
# 用法：python scripts/train/extract_first_tokens.py
#       python scripts/train/extract_first_tokens.py --model_path /path/to/model
#       python scripts/train/extract_first_tokens.py --data_path ./my_data.json
if __name__ == "__main__":
    import argparse

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

    _DEFAULT_MODEL_PATH = os.path.join(
        _PROJECT_ROOT, "model", "DS", "DeepSeek-R1-Distill-Qwen-7B"
    )
    _DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "datasets", "first_tokens.json")

    parser = argparse.ArgumentParser(description="提取 solution 首 token 统计")
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help=f"tokenizer 模型路径（默认 {_DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="自定义 JSON 数据路径（需含 solution 字段）。不指定则使用 MATH_All 数据集",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=_DEFAULT_OUTPUT_PATH,
        help=f"输出 JSON 路径（默认 {_DEFAULT_OUTPUT_PATH}）",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.data_path:
        # 从自定义 JSON 加载
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        solutions = [
            _stringify(item.get("solution", item.get("ref_solution", "")))
            for item in raw_data
        ]
        print(f"Loaded {len(solutions)} solutions from {args.data_path}")
    else:
        # 默认使用 MATH_All 数据集
        from data_math import Math_All

        print("Loading MATH_All dataset ...")
        data = Math_All(train=True)
        solutions = data.solutions
        print(f"Loaded {len(solutions)} solutions from MATH_All")

    result = extract_and_save_first_tokens(solutions, tokenizer, args.output_path)

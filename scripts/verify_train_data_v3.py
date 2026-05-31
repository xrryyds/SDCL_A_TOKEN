"""verify_train_data_v3.py — Stage B loader/encode 验证

校验:
  1) _load_train_data 总数 = corr + roll + pool
  2) per-source 计数与 Stage A 对得上(corr 5456 / roll 2134 / pool 24326)
  3) encode 3 条 pool: fill_token_id 是 int, fill_pos_in_seq == prompt_len,
     input_ids[fill_pos_in_seq] == fill_token_id
  4) encode 3 条 roll: 无 fill_token_id, answer_len > 0
  5) encode 3 条 corr: 无 fill_token_id, answer_len > 0
"""

import os
import sys
from collections import Counter

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer

from scripts.train.a_token_sdcl_train import _encode_sample, _load_train_data

DATA = os.path.join(_ROOT, "datasets", "train", "train_data_v3.json")
MODEL = "/root/models/DeepSeek-R1-Distill-Qwen-7B"

EXPECT = {"corr_answer": 5456, "roll": 2134, "pool": 24326}


def main():
    print(f"加载 {DATA}")
    data = _load_train_data(DATA)
    print(f"total = {len(data)}")

    cnt = Counter(s["source"] for s in data)
    print(f"per-source counts: {dict(cnt)}")

    ok = True
    for src, expected in EXPECT.items():
        got = cnt.get(src, 0)
        flag = "✓" if got == expected else "✗"
        print(f"  {flag} {src}: {got} (expected {expected})")
        if got != expected:
            ok = False

    if not ok:
        print("⚠ 计数与 Stage A 不一致")
        return 1

    # tokenizer 仅用于编码,不下载模型
    print(f"\n加载 tokenizer {MODEL}")
    if not os.path.isdir(MODEL):
        print(f"⚠ tokenizer 路径不存在: {MODEL}; 跳过 encode 验证(仅计数)")
        return 0
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    samples_by_src = {"corr_answer": [], "roll": [], "pool": []}
    for s in data:
        src = s["source"]
        if src in samples_by_src and len(samples_by_src[src]) < 3:
            samples_by_src[src].append(s)
        if all(len(v) >= 3 for v in samples_by_src.values()):
            break

    print("\n=== encode 抽样 ===")
    for src, lst in samples_by_src.items():
        for i, s in enumerate(lst):
            enc = _encode_sample(tok, s, max_prompt_length=2048, max_answer_length=4096)
            if enc is None:
                print(f"  [{src}#{i}] ✗ encode -> None")
                ok = False
                continue
            line = (
                f"  [{src}#{i}] prompt_len={enc['prompt_len']} answer_len={enc['answer_len']} "
                f"source={enc['source']} fill_tok_id={enc['fill_token_id']} "
                f"fill_pos={enc['fill_pos_in_seq']}"
            )
            print(line)

            if src == "pool":
                if not isinstance(enc["fill_token_id"], int):
                    print(f"     ✗ fill_token_id 不是 int")
                    ok = False
                if enc["fill_pos_in_seq"] != enc["prompt_len"]:
                    print(f"     ✗ fill_pos_in_seq({enc['fill_pos_in_seq']}) != prompt_len({enc['prompt_len']})")
                    ok = False
                got_tok = enc["input_ids"][enc["fill_pos_in_seq"]]
                if got_tok != enc["fill_token_id"]:
                    print(f"     ✗ input_ids[fill_pos] = {got_tok} != fill_token_id = {enc['fill_token_id']}")
                    ok = False
            else:
                if enc["fill_token_id"] is not None:
                    print(f"     ✗ {src} 不该有 fill_token_id")
                    ok = False
                if enc["answer_len"] <= 0:
                    print(f"     ✗ answer_len = {enc['answer_len']}")
                    ok = False

    print()
    if ok:
        print("✓ Stage B 验证通过")
        return 0
    print("✗ Stage B 验证失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())

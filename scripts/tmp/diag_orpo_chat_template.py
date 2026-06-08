"""Debug: 看 ORPODataset 处理后 prompt_str / chosen_str 长什么样, 找首 token 错位 bug。"""
import json, os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True, use_fast=False,
)

SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."

d = json.load(open("/workspace/SDCL_A_TOKEN/datasets/train/train_data_orpo.json"))
it = d[0]
q = it["prompt"]
chosen = it["chosen"]

prompt_str = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": q}],
    tokenize=False, add_generation_prompt=True,
)
chosen_str = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": q},
     {"role": "assistant", "content": chosen}],
    tokenize=False,
)

print("=" * 70)
print("prompt_str 末尾 200 字符:")
print(repr(prompt_str[-200:]))
print()
print("=" * 70)
print("chosen_str 前 300 字符:")
print(repr(chosen_str[:300]))
print()
print("=" * 70)
print("chosen_str 在 prompt_str 后多出来的部分 前 300 字符:")
extra = chosen_str[len(prompt_str):]
print(repr(extra[:300]))
print()
print("=" * 70)
print("chosen 原文 (json 里) 前 80 字符:")
print(repr(chosen[:80]))

# tokenize 看具体 ids
prompt_ids = tok(prompt_str, add_special_tokens=False).input_ids
chosen_ids = tok(chosen_str, add_special_tokens=False).input_ids
print()
print("=" * 70)
print(f"len(prompt_ids) = {len(prompt_ids)}")
print(f"len(chosen_ids) = {len(chosen_ids)}")
print(f"chosen_ids[:len(prompt_ids)] == prompt_ids ? {chosen_ids[:len(prompt_ids)] == prompt_ids}")
print(f"prompt_ids 末尾 5 token:")
for tid in prompt_ids[-5:]:
    print(f"  {tid}: {tok.decode([tid], skip_special_tokens=False)!r}")
print(f"chosen_ids[len(prompt_ids):][:8] (response 前 8 token):")
for tid in chosen_ids[len(prompt_ids):len(prompt_ids)+8]:
    print(f"  {tid}: {tok.decode([tid], skip_special_tokens=False)!r}")

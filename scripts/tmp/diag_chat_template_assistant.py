"""查 DeepSeek-R1-Distill 的 chat_template 在 assistant 消息上的渲染逻辑。

观察现象: ORPODataset 里 apply_chat_template([sys, user, assistant=X]) 渲染出
chosen_str 末尾不是 X 本身, 而是别的内容 ("First, we start with...")。

可能 R1 chat_template 对 assistant 内容做了特殊处理 (吃掉 <think> 段, 截断开头等)。

本脚本不跑模型, 直接打 chat_template 字符串 + 不同 assistant 内容的渲染结果对比。
"""
import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True, use_fast=False,
)

print("=" * 70)
print("chat_template 原文 (前 1000 字符):")
print(repr(tok.chat_template[:1000] if tok.chat_template else None))
print()
print("=" * 70)
print("chat_template 全文长度:", len(tok.chat_template) if tok.chat_template else 0)
print()

# Test 1: 简单 assistant 内容 "Appreciate the problem"
print("=" * 70)
print("Test 1: assistant content = 'Appreciate the problem'")
out1 = tok.apply_chat_template(
    [{"role": "system", "content": "be helpful"},
     {"role": "user", "content": "what is 1+1?"},
     {"role": "assistant", "content": "Appreciate the problem"}],
    tokenize=False,
)
print(repr(out1))
print()

# Test 2: 含 <think> 标签的 assistant 内容
print("=" * 70)
print("Test 2: assistant content = '<think>let me think</think>Answer is 2'")
out2 = tok.apply_chat_template(
    [{"role": "system", "content": "be helpful"},
     {"role": "user", "content": "what is 1+1?"},
     {"role": "assistant", "content": "<think>let me think</think>Answer is 2"}],
    tokenize=False,
)
print(repr(out2))
print()

# Test 3: 不加 assistant, 只 add_generation_prompt
print("=" * 70)
print("Test 3: 只 sys+user, add_generation_prompt=True")
out3 = tok.apply_chat_template(
    [{"role": "system", "content": "be helpful"},
     {"role": "user", "content": "what is 1+1?"}],
    tokenize=False, add_generation_prompt=True,
)
print(repr(out3))

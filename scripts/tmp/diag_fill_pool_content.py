"""检查 fill_unsolve_pool.json 第 0 条的 candidate 实际内容,
确认 chosen 文本到底是什么 (用于定位 chat_template bug)。"""
import json, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_unsolve_pool.json")

d = json.load(open(PATH))
print(f"fill_unsolve_pool: {len(d)} 题")
print()
print("第 0 条:")
it = d[0]
print(f"  qi={it['question_idx']}")
print(f"  ref={it['ref_answer']!r}")
print(f"  question 前80: {it['question'][:80]!r}")
print(f"  candidates 数量: {len(it['candidates'])}")
print()
for i, c in enumerate(it["candidates"][:3]):
    print(f"  [{i}] token_id={c['token_id']} token_text={c['token_text']!r}")
    print(f"      answer 前150: {c['answer'][:150]!r}")
    print(f"      answer len: {len(c['answer'])} 字符")
    print()

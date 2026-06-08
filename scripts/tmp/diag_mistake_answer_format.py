"""看 mistake 池里 base greedy answer 的格式 (是否含 </think>+final, 决定 ORPO rejected 拼接方式)。"""
import json, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")

d = json.load(open(PATH))
print(f"mistake 池: {len(d)} 题")
print()

for q_idx in [0, 1, 2]:
    if q_idx >= len(d):
        break
    it = d[q_idx]
    ans = it.get("answer", "")
    print(f"==== 第 {q_idx} 条 (qi={it['question_idx']}, ref={it['ref_answer']!r}) ====")
    print(f"  answer 长度: {len(ans)} 字符")
    print(f"  answer 头 200:")
    print(f"    {ans[:200]!r}")
    print(f"  answer 尾 200:")
    print(f"    {ans[-200:]!r}")
    if "</think>" in ans:
        idx = ans.find("</think>")
        print(f"  含 </think> at idx {idx} (整长 {len(ans)})")
    else:
        print(f"  不含 </think>")
    if "\\boxed{" in ans:
        idx = ans.rfind("\\boxed{")
        print(f"  含 \\boxed{{ (rfind) at idx {idx}")
    else:
        print(f"  不含 \\boxed{{")
    print()

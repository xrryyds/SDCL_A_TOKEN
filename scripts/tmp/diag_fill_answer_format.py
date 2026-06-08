"""看 fill 池里 candidate.answer 完整结构 (头/尾), 定 ORPO chosen 格式。

R1-Distill 推理时 prompt 末尾有 <think>\n, 模型输出:
  <think>\n (prompt 给的) + thinking... + </think>\n + final answer with \\boxed{}

fill 时强塞 first token 进 prompt 末尾, 然后 greedy 续写。
candidate.answer = token_text + 续写文本。

问题: 续写文本是从 <think> 内部开始的吗? 还是含 </think> + final answer?

本脚本看完整 answer 头/尾, 确定 ORPO chosen_str 该怎么构造。
"""
import json, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(_PROJECT_ROOT, "datasets", "exam", "fill_unsolve_pool.json")

d = json.load(open(PATH))
print(f"fill_unsolve_pool: {len(d)} 题")
print()

for q_idx in [0, 1, 2]:
    if q_idx >= len(d):
        break
    it = d[q_idx]
    print(f"==== 第 {q_idx} 条 (qi={it['question_idx']}, ref={it['ref_answer']!r}) ====")
    c = it["candidates"][0]
    ans = c["answer"]
    print(f"  token_text={c['token_text']!r}")
    print(f"  answer 长度: {len(ans)} 字符")
    print(f"  answer 头 200:")
    print(f"    {ans[:200]!r}")
    print(f"  answer 尾 200:")
    print(f"    {ans[-200:]!r}")
    # 看是否含 </think> 标签
    if "</think>" in ans:
        idx = ans.find("</think>")
        print(f"  含 </think> at idx {idx}")
        print(f"  </think> 前 100:  {ans[max(0,idx-100):idx]!r}")
        print(f"  </think> 后 100:  {ans[idx:idx+100]!r}")
    else:
        print(f"  不含 </think> ← 说明续写没结束 thinking")
    if "\\boxed{" in ans:
        idx = ans.find("\\boxed{")
        print(f"  含 \\boxed{{ at idx {idx}")
    else:
        print(f"  不含 \\boxed{{ ← 异常 (按理 boxed 命中才进 candidates)")
    print()

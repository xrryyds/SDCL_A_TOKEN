"""对比 mistake 池里存的 answer vs 当前 take_exam 重新生成的 answer, 完整落盘。

矛盾: 同一份 take_exam (5/26后没改过), 同 Base, greedy 零抖动, 但
'生成==池里存的' 只有 0.7%。本脚本把前 5 题的 [prompt / 池里存的 / 现在生成]
完整 dump 到文件, 肉眼对比漂移点。

用法 (远程 4 卡):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_dump_compare.py --n 5 --max_new 8192
输出: scripts/tmp/dump_compare.txt
"""
import argparse, gc, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from utils.data_utils import extract_boxed_content, normalize_answer
from scripts.inference.take_exam import TakeExam, SYSTEM_PROMPT

MODEL = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
MIS = os.path.join(_ROOT, "datasets", "exam", "mistake_DS_MATH_pool.json")
OUT = os.path.join(_ROOT, "scripts", "tmp", "dump_compare.txt")


def boxed(t):
    b = extract_boxed_content(t or "")
    return normalize_answer(b) if b is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--max_prompt_length", type=int, default=10240)
    args = ap.parse_args()

    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_ids = list(range(len([x for x in cuda.split(",") if x.strip()]))) if cuda else [0]

    mis = json.load(open(MIS))[:args.n]
    q = [it["question"] for it in mis]
    ans_ref = [str(it["ref_answer"]) for it in mis]
    sol = [it.get("ref_solution", "") for it in mis]
    idx = list(range(len(q)))
    stored = [it.get("answer", "") for it in mis]

    print(f"[take_exam SYSTEM_PROMPT] {SYSTEM_PROMPT!r}", flush=True)

    te = TakeExam(model_path=MODEL, max_prompt_length=args.max_prompt_length, max_new_tokens=args.max_new)
    # 打印 take_exam 实际构造的 prompt (复用其 _build_prompts 看真实喂给模型的文本)
    try:
        built = None
        if hasattr(te, "_build_prompts"):
            try:
                built = te._build_prompts(q)
            except Exception as e:
                built = f"(_build_prompts 调用失败: {e})"
        res = te.exam_multi_gpu(q, sol, ans_ref, idx, device_ids=device_ids,
                                write_output=False, sample_n=1, temperature=0.0, top_p=1.0)
    finally:
        del te; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_idx = {r["question_idx"]: r["answer"] for r in res}

    lines = []
    lines.append(f"take_exam SYSTEM_PROMPT = {SYSTEM_PROMPT!r}")
    if isinstance(built, list):
        lines.append("\n===== take_exam 实际喂给模型的 prompt[0] (前500字符) =====")
        lines.append(repr(built[0][:500]))
    elif built:
        lines.append(f"\n_build_prompts: {built}")

    for k, i in enumerate(idx):
        new_ans = by_idx.get(i) or ""
        st = stored[i]
        lines.append("\n" + "=" * 80)
        lines.append(f"题 {k} q_idx={mis[i].get('question_idx')} ref={ans_ref[i]!r}")
        lines.append(f"  question(前200): {q[i][:200]!r}")
        lines.append(f"  [池里存的] boxed={boxed(st)!r} len={len(st)}")
        lines.append(f"    头300: {st[:300]!r}")
        lines.append(f"    尾150: {st[-150:]!r}")
        lines.append(f"  [现在生成] boxed={boxed(new_ans)!r} len={len(new_ans)}")
        lines.append(f"    头300: {new_ans[:300]!r}")
        lines.append(f"    尾150: {new_ans[-150:]!r}")
        lines.append(f"  完全相同: {st.strip() == new_ans.strip()}")

    txt = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt, flush=True)
    print(f"\n[written] {OUT}", flush=True)


if __name__ == "__main__":
    main()

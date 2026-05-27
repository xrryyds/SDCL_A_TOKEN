"""一次性盘点 a_token 训练数据（DeepMath 版）。

跑法（在项目根）：
    python scripts/inspect_train_data.py

可选：
    --base   datasets/exam            # 池子目录
    --train  a_token_train_data.json  # 训练集文件名
    --tokenizer  /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B
        # 给了就跑 token 长度分布；不给只统计字符数
    --sample 200                      # token 长度抽样条数（默认全量太慢时用）

输出：
  1) 各池条数（mistake / corr / fill_correct / a_token_train_data）
  2) 训练集 source 分布（corr_answer vs fill_correct）+ 救回率推算
  3) 训练集 fill_token 复用统计（top-20 fill_token_text 直方图）
  4) question / answer 字段长度分布（字符数 + 可选 token 数）
  5) 各举一条 corr / fill 样例（截断展示）
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

# 让 `from scripts.train.a_token_sd import ...` 在任何 cwd 下都能找到。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _safe_load_list(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, list) else None


def _percentiles(xs, ps=(50, 90, 95, 99, 100)):
    if not xs:
        return {p: None for p in ps}
    s = sorted(xs)
    n = len(s)
    out = {}
    for p in ps:
        if p >= 100:
            out[p] = s[-1]
        else:
            k = max(0, min(n - 1, int(round(p / 100 * (n - 1)))))
            out[p] = s[k]
    return out


def _fmt_pct(d):
    return " ".join(f"p{p}={v}" for p, v in d.items() if v is not None)


def _clip(s, n=160):
    if s is None:
        return None
    s = str(s)
    return s if len(s) <= n else s[:n] + f"...<+{len(s)-n}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="datasets/exam")
    ap.add_argument("--train", default="a_token_train_data.json")
    ap.add_argument("--mistake", default="mistake_DS_MATH_pool.json")
    ap.add_argument("--corr", default="corr_DS_MATH_pool.json")
    ap.add_argument("--fill", default="fill_correct.json")
    ap.add_argument("--tokenizer", default=None,
                    help="HF 模型路径；给了就跑 token 长度分布")
    ap.add_argument("--sample", type=int, default=0,
                    help="token 长度抽样条数（0=全量）")
    args = ap.parse_args()

    base = args.base
    files = {
        "mistake": os.path.join(base, args.mistake),
        "corr":    os.path.join(base, args.corr),
        "fill":    os.path.join(base, args.fill),
        "train":   os.path.join(base, args.train),
    }

    print("=" * 78)
    print("[1] 池子规模")
    print("=" * 78)
    pools = {}
    for k, p in files.items():
        d = _safe_load_list(p)
        pools[k] = d
        n = len(d) if d is not None else -1
        flag = "" if d is not None else "  <MISSING>"
        print(f"  {k:8s}  {p}  n={n}{flag}")

    n_mistake = len(pools["mistake"]) if pools["mistake"] is not None else None
    n_corr_pool = len(pools["corr"]) if pools["corr"] is not None else None
    n_fill = len(pools["fill"]) if pools["fill"] is not None else None
    n_train = len(pools["train"]) if pools["train"] is not None else None

    if n_mistake and n_fill is not None:
        print(f"\n  fill 救回率 = fill / mistake = {n_fill} / {n_mistake} "
              f"= {n_fill/n_mistake*100:.2f}%")
    if n_corr_pool is not None and n_fill is not None:
        print(f"  期望 train 总数 = corr({n_corr_pool}) + fill({n_fill}) "
              f"= {n_corr_pool + n_fill}")
        if n_train is not None:
            diff = n_corr_pool + n_fill - n_train
            note = "（去重 / 缺失）" if diff != 0 else "（一致）"
            print(f"  实际 train 总数 = {n_train}   差 = {diff} {note}")

    train = pools["train"]
    if not train:
        print("\n训练集为空 / 不存在，后续统计跳过。")
        return

    print()
    print("=" * 78)
    print("[2] 训练集 source 分布")
    print("=" * 78)
    src_cnt = Counter(x.get("source") for x in train)
    for k, v in src_cnt.most_common():
        print(f"  {str(k):15s} = {v:7d}  ({v/n_train*100:.2f}%)")

    print()
    print("=" * 78)
    print("[3] fill_token 复用 top-20")
    print("=" * 78)
    fill_items = [x for x in train if x.get("source") == "fill_correct"]
    tok_cnt = Counter(
        (x.get("fill_token_id"), x.get("fill_token_text"))
        for x in fill_items
    )
    print(f"  fill 条数 = {len(fill_items)}, 不同 fill_token = {len(tok_cnt)}")
    for (tid, ttext), c in tok_cnt.most_common(20):
        share = c / max(1, len(fill_items)) * 100
        print(f"    id={tid!s:>7s}  text={ttext!r:<20s} count={c:5d}  ({share:5.2f}%)")

    print()
    print("=" * 78)
    print("[4] 字段长度（字符数）")
    print("=" * 78)
    for field in ("question", "answer"):
        lens = [len(str(x.get(field, ""))) for x in train]
        if not lens:
            continue
        avg = statistics.mean(lens)
        print(f"  [{field}] mean={avg:.1f}  {_fmt_pct(_percentiles(lens))}")

    if args.tokenizer:
        print()
        print("=" * 78)
        print(f"[5] 字段长度（token 数, tokenizer={args.tokenizer}）")
        print("=" * 78)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True, use_fast=False
            )
            samples = train if args.sample <= 0 else train[: args.sample]
            for field in ("question", "answer"):
                lens = []
                for x in samples:
                    s = str(x.get(field, ""))
                    if not s:
                        lens.append(0)
                        continue
                    lens.append(len(tok(s, add_special_tokens=False).input_ids))
                avg = statistics.mean(lens) if lens else 0
                print(f"  [{field}] n={len(lens)}  mean={avg:.1f}  "
                      f"{_fmt_pct(_percentiles(lens))}")

            # 顺便算 prompt（apply_chat_template 后）的真实 token 长度，
            # 帮你判断 train_max_prompt_length=2048 是否够。
            try:
                from scripts.train.a_token_sd import (
                    SYSTEM_PROMPT, normalize_question_text,
                )
                prompt_lens = []
                for x in samples:
                    msgs = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": normalize_question_text(
                             str(x.get("question", "")))},
                    ]
                    p_text = tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    prompt_lens.append(
                        len(tok(p_text, add_special_tokens=False).input_ids)
                    )
                avg = statistics.mean(prompt_lens) if prompt_lens else 0
                print(f"  [chat_prompt] n={len(prompt_lens)}  mean={avg:.1f}  "
                      f"{_fmt_pct(_percentiles(prompt_lens))}")
                # 给训练侧的预算建议
                p95 = _percentiles(prompt_lens, ps=(95,))[95]
                if p95 is not None:
                    print(f"\n  → 训练 prompt 预算建议：--train_max_prompt_length "
                          f">= {p95} (p95)；当前你设的 2048 "
                          f"{'够' if p95 <= 2048 else '不够'}")
            except Exception as e:
                print(f"  chat_prompt 长度计算失败: {e}")
        except Exception as e:
            print(f"  tokenizer 加载失败: {e}")

    print()
    print("=" * 78)
    print("[6] 样例（各举一条）")
    print("=" * 78)
    for src in ("corr_answer", "fill_correct"):
        ex = next((x for x in train if x.get("source") == src), None)
        print(f"\n  --- source={src} ---")
        if ex is None:
            print("    (none)")
            continue
        for k in ("question_idx", "source", "fill_token_id", "fill_token_text",
                  "ref_answer", "question", "answer"):
            print(f"    {k:16s}= {_clip(ex.get(k))}")


if __name__ == "__main__":
    main()

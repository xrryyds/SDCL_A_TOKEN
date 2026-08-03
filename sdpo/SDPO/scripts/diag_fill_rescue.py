#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SRPO fill-rescue diagnostic.

Question: of the GRPO-no-signal data (prompts whose roll-8 is ALL wrong), how many
can first-token filling rescue? Compare two forced-token sources on the SAME
all-wrong prompts:

  (A) model's own first-token distribution (top-K of the logit right after the
      "<reasoning>\\n" prefix), excluding the defaults the roll-8 already used;
  (B) a broad opener candidate pool (datasets/first_tokens_test.json), excluding
      the same defaults.

For each all-wrong prompt we force prompt + "<reasoning>\\n" + candidate_token, greedily
generate, and score with the MCQ verifier. A prompt is "rescued" by a source if ANY of
its candidate tokens yields a correct answer.

Runs fully offline with vLLM. Example:
  python scripts/diag_fill_rescue.py \
    --model /home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B \
    --data datasets/sciknoweval/chemistry/train.json \
    --pool /home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_tokens_test.json \
    --limit 200 --roll_n 8 --max_tokens 1024 --topk_a 20 --pool_b 50 --tp 8
"""
import argparse
import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


SYSTEM_DEFAULT = None
PREFIX = "<reasoning>\n"


def extract_xml_answer(text: str) -> str:
    ans = text.split("<answer>")[-1]
    ans = ans.split("</answer>")[0]
    return ans.strip()


def score(text: str, gt: str) -> float:
    return float(extract_xml_answer(text) == gt)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt_ids(tok, question, system):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pool", required=True, help="first_tokens_test.json opener pool")
    ap.add_argument("--out", default="output/fill_rescue_diag.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="number of questions to evaluate")
    ap.add_argument("--roll_n", type=int, default=8)
    ap.add_argument("--roll_temp", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--topk_a", type=int, default=20, help="model-dist candidates after excluding defaults")
    ap.add_argument("--probe_logprobs", type=int, default=60, help="top-K logprobs to read from the first-token logit (also sets engine max_logprobs)")
    ap.add_argument("--pool_b", type=int, default=50, help="opener-pool candidates (top by count) after excluding defaults")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--success_threshold", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    prefix_ids = tok(PREFIX, add_special_tokens=False)["input_ids"]
    plen = len(prefix_ids)

    pool = json.load(open(args.pool))
    pool_tokens = [int(t["token_id"]) for t in pool["tokens"]]  # ranked by count desc

    rows = load_jsonl(args.data)[: args.limit]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        max_logprobs=args.probe_logprobs,
    )

    # ---------- Stage 1: roll-8, find all-wrong (no-signal) prompts ----------
    prompt_ids_list = [build_prompt_ids(tok, r["prompt"], r.get("system")) for r in rows]
    sp_roll = SamplingParams(n=args.roll_n, temperature=args.roll_temp, top_p=1.0, max_tokens=args.max_tokens)
    roll_out = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids_list], sp_roll)

    all_wrong = []  # list of dict: idx, gt, prompt_ids, defaults(set of first meaningful token ids)
    for i, out in enumerate(roll_out):
        gt = rows[i]["answer"]
        n_correct = 0
        defaults = set()
        for comp in out.outputs:
            if score(comp.text, gt) >= args.success_threshold:
                n_correct += 1
            if len(comp.token_ids) > plen:
                defaults.add(int(comp.token_ids[plen]))
            elif len(comp.token_ids) > 0:
                defaults.add(int(comp.token_ids[0]))
        if n_correct == 0:
            all_wrong.append({"idx": i, "gt": gt, "prompt_ids": prompt_ids_list[i], "defaults": defaults})

    print(f"[stage1] evaluated={len(rows)}  all_wrong(no-signal)={len(all_wrong)} "
          f"({100.0*len(all_wrong)/max(len(rows),1):.1f}%)", flush=True)
    if not all_wrong:
        print("No all-wrong prompts; nothing to rescue.")
        return

    # ---------- Stage 2a: probe model first-token logit (top-K) after prefix ----------
    sp_probe = SamplingParams(max_tokens=1, temperature=1.0, logprobs=args.probe_logprobs)
    probe_out = llm.generate(
        [TokensPrompt(prompt_token_ids=q["prompt_ids"] + prefix_ids) for q in all_wrong], sp_probe
    )
    for q, out in zip(all_wrong, probe_out):
        lp = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
        ranked = sorted(lp.keys(), key=lambda t: lp[t].logprob, reverse=True)
        cand_a = [int(t) for t in ranked if int(t) not in q["defaults"]][: args.topk_a]
        q["cand_a"] = cand_a
        q["cand_b"] = [t for t in pool_tokens if t not in q["defaults"]][: args.pool_b]

    # ---------- Stage 2b: build all forced prompts, batch generate greedily ----------
    forced_prompts = []
    index = []  # (q_pos, source, token_id)
    for qi, q in enumerate(all_wrong):
        for k in q["cand_a"]:
            forced_prompts.append(TokensPrompt(prompt_token_ids=q["prompt_ids"] + prefix_ids + [int(k)]))
            index.append((qi, "a", int(k)))
        for k in q["cand_b"]:
            forced_prompts.append(TokensPrompt(prompt_token_ids=q["prompt_ids"] + prefix_ids + [int(k)]))
            index.append((qi, "b", int(k)))

    sp_fill = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)
    print(f"[stage2] forced generations = {len(forced_prompts)} "
          f"(all_wrong={len(all_wrong)}, cand_a<= {args.topk_a}, cand_b<= {args.pool_b})", flush=True)
    fill_out = llm.generate(forced_prompts, sp_fill)

    # aggregate rescue per question per source
    for q in all_wrong:
        q["resc_a"] = False
        q["resc_b"] = False
        q["hit_a_tokens"] = []
        q["hit_b_tokens"] = []
    for (qi, src, k), out in zip(index, fill_out):
        prefix_text = tok.decode(prefix_ids)
        tok_text = tok.decode([k])
        full = prefix_text + tok_text + out.outputs[0].text
        if score(full, all_wrong[qi]["gt"]) >= args.success_threshold:
            if src == "a":
                all_wrong[qi]["resc_a"] = True
                all_wrong[qi]["hit_a_tokens"].append(k)
            else:
                all_wrong[qi]["resc_b"] = True
                all_wrong[qi]["hit_b_tokens"].append(k)

    # ---------- Report ----------
    N = len(all_wrong)
    resc_a = sum(1 for q in all_wrong if q["resc_a"])
    resc_b = sum(1 for q in all_wrong if q["resc_b"])
    both = sum(1 for q in all_wrong if q["resc_a"] and q["resc_b"])
    either = sum(1 for q in all_wrong if q["resc_a"] or q["resc_b"])
    a_only = sum(1 for q in all_wrong if q["resc_a"] and not q["resc_b"])
    b_only = sum(1 for q in all_wrong if q["resc_b"] and not q["resc_a"])
    avg_ca = sum(len(q["cand_a"]) for q in all_wrong) / N
    avg_cb = sum(len(q["cand_b"]) for q in all_wrong) / N

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q in all_wrong:
            f.write(json.dumps({
                "idx": q["idx"], "gt": q["gt"],
                "defaults": sorted(q["defaults"]),
                "n_cand_a": len(q["cand_a"]), "n_cand_b": len(q["cand_b"]),
                "rescued_model_dist": q["resc_a"], "rescued_pool": q["resc_b"],
                "hit_a_tokens": q["hit_a_tokens"], "hit_b_tokens": q["hit_b_tokens"],
            }, ensure_ascii=False) + "\n")

    print("\n================ FILL-RESCUE DIAGNOSTIC ================")
    print(f"all-wrong (GRPO no-signal) prompts:            {N}")
    print(f"avg candidates  A(model-dist, excl default):   {avg_ca:.2f}")
    print(f"avg candidates  B(opener pool,  excl default):  {avg_cb:.2f}")
    print(f"rescued by A (model first-token dist):         {resc_a}/{N} = {100.0*resc_a/N:.1f}%")
    print(f"rescued by B (broad opener pool):              {resc_b}/{N} = {100.0*resc_b/N:.1f}%")
    print(f"rescued by EITHER:                             {either}/{N} = {100.0*either/N:.1f}%")
    print(f"  both A&B: {both}   A-only: {a_only}   B-only: {b_only}")
    print(f"per-question detail -> {args.out}")
    print("=======================================================")


if __name__ == "__main__":
    main()

"""verify_pool_dist_first_token.py — 验证 pool_dist 训完后, 首 token 分布是否真集中到 target_token_ids.

直接拿 train_data_pool_dist.json 的 prompt, 过 LoRA forward, 看 prompt 末尾 (首 token 预测位)
的 logits → softmax → 看落在该题 target_token_ids 上的总概率.

不跑 eval, 不跟 Base 对比, 单纯回答: 训练 loss 把首 token 拉到 target 上了吗.

输出:
  逐题打 [q_idx, |target|, p@target, top5_in_target/top5, top1_token]
  汇总 p@target 的 mean / median / p25 / p75 / 命中 top1 比例 等

用法:
  python scripts/verify_pool_dist_first_token.py \
    --lora_path output/pool_dist_v1_20260603_110839/checkpoint_epoch_2 \
    --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
    --data_path datasets/train/train_data_pool_dist.json \
    --num_questions 50 \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.train.a_token_sd import SYSTEM_PROMPT, normalize_question_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": normalize_question_text(question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True,
                        help="训完的 LoRA 目录, 比如 output/pool_dist_v1_<ts>/checkpoint_epoch_2")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model 路径")
    parser.add_argument(
        "--stage1_lora_path", type=str, default=None,
        help="可选: stage1 LoRA 路径; 设了会先 merge 进 Base 再挂 --lora_path "
             "(评测 stage2 LoRA 时必须设, 因为 stage2 LoRA 是基于 Base+stage1 训的).",
    )
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(_PROJECT_ROOT, "datasets", "train", "train_data_pool_dist.json"))
    parser.add_argument("--fill_pool_path", type=str, default=None,
                        help="可选: 直接读 fill_multi_pool.json (按题 candidates), "
                             "每题 target_token_ids = 该题所有 candidate 的 token_id. "
                             "设了则忽略 --data_path.")
    parser.add_argument("--num_questions", type=int, default=50,
                        help="抽几题验证 (按题去重, 不是按样本)")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str, default=None,
                        help="结果落盘 JSONL (可选)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需 CUDA")

    # ---- 加载数据, 按 question_idx 去重 ----
    if args.fill_pool_path:
        # 模式 B: 从 fill_multi_pool.json 按题取 candidates 的 token_id 当 target
        with open(args.fill_pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        questions = []
        for item in pool:
            tids = []
            for c in item.get("candidates", []):
                tid = c.get("token_id")
                if tid is not None:
                    tids.append(int(tid))
            tids = sorted(set(tids))
            if not item.get("question") or not tids:
                continue
            questions.append({
                "question": item["question"],
                "question_idx": item.get("question_idx", -1),
                "target_token_ids": tids,
            })
        logger.info("fill_pool 模式: %d 题 (target=每题全部 candidate 首 token)", len(questions))
    else:
        with open(args.data_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        seen_q = {}
        for s in all_samples:
            qidx = s["question_idx"]
            if qidx not in seen_q:
                seen_q[qidx] = s
        questions = list(seen_q.values())
        logger.info("数据池总样本 %d, 去重后题数 %d", len(all_samples), len(questions))

    # 抽 num_questions 题
    rng = torch.Generator().manual_seed(args.seed)
    if args.num_questions < len(questions):
        idx = torch.randperm(len(questions), generator=rng).tolist()[:args.num_questions]
        sampled = [questions[i] for i in idx]
    else:
        sampled = questions
    logger.info("抽样验证 %d 题", len(sampled))

    # ---- 加载模型 ----
    logger.info("加载 tokenizer + Base + LoRA ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device)
    base.eval()

    # 如果给了 stage1 LoRA, 先 merge 进 Base 再挂 stage2 LoRA
    if args.stage1_lora_path:
        logger.info("先 merge stage1 LoRA: %s", args.stage1_lora_path)
        base = PeftModel.from_pretrained(base, args.stage1_lora_path).to(args.device)
        base = base.merge_and_unload()
        logger.info("stage1 LoRA 已 merge 进 Base")

    model = PeftModel.from_pretrained(base, args.lora_path).to(args.device)
    model.eval()
    logger.info("模型加载完毕")

    # ---- 跑 forward 取首 token logits ----
    results: List[Dict] = []
    p_at_target_list: List[float] = []
    top1_in_target_count = 0
    top1_eq_fill_count = 0  # top1 是不是该题任一 candidate 的 fill_token_id

    for i, sample in enumerate(sampled):
        q = sample["question"]
        target_ids = sample["target_token_ids"]
        N = len(target_ids)

        prompt_text = _build_prompt(tokenizer, q)
        ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if len(ids) > args.max_prompt_length:
            ids = ids[-args.max_prompt_length:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        attn = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            # 首 token 预测位 = prompt 最后一个 token (logits[-1] 预测下一个 token)
            first_logits = out.logits[0, -1, :].float()
            probs = F.softmax(first_logits, dim=-1)

        target_idx_t = torch.tensor(target_ids, dtype=torch.long, device=args.device)
        p_at_target = float(probs.index_select(0, target_idx_t).sum().item())
        p_at_target_list.append(p_at_target)

        topk = torch.topk(probs, k=10)
        top_ids = topk.indices.tolist()
        top_probs = topk.values.tolist()

        target_set = set(target_ids)
        top1_id = top_ids[0]
        if top1_id in target_set:
            top1_in_target_count += 1
            top1_eq_fill_count += 1

        # 解码 top10 token text 方便看
        top_decoded = []
        for tid, p in zip(top_ids, top_probs):
            txt = tokenizer.decode([tid])
            in_tgt = "✓" if tid in target_set else " "
            top_decoded.append(f"{in_tgt}{tid}={txt!r}({p:.3f})")

        record = {
            "question_idx": sample["question_idx"],
            "target_size": N,
            "p_at_target": p_at_target,
            "top1_id": top1_id,
            "top1_in_target": top1_id in target_set,
            "top10": top_decoded,
        }
        results.append(record)

        if i < 10 or i % 10 == 0:
            logger.info(
                "[%3d] q_idx=%s |tgt|=%d p@tgt=%.4f top1=%d(%r,in_tgt=%s) top3=%s",
                i, sample["question_idx"], N, p_at_target,
                top1_id, tokenizer.decode([top1_id]),
                top1_id in target_set,
                top_decoded[:3],
            )

    # ---- 汇总 ----
    p_sorted = sorted(p_at_target_list)
    n = len(p_sorted)
    if n > 0:
        mean_p = sum(p_sorted) / n
        median_p = p_sorted[n // 2]
        p25 = p_sorted[n // 4]
        p75 = p_sorted[3 * n // 4]
        p_min = p_sorted[0]
        p_max = p_sorted[-1]
        n_ge_50 = sum(1 for p in p_sorted if p >= 0.5)
        n_ge_80 = sum(1 for p in p_sorted if p >= 0.8)
        n_ge_99 = sum(1 for p in p_sorted if p >= 0.99)

        logger.info("=" * 60)
        logger.info("汇总 (n=%d 题):", n)
        logger.info("  p@target  min=%.4f p25=%.4f median=%.4f mean=%.4f p75=%.4f max=%.4f",
                    p_min, p25, median_p, mean_p, p75, p_max)
        logger.info("  p@target ≥ 0.50 : %d/%d (%.1f%%)", n_ge_50, n, 100*n_ge_50/n)
        logger.info("  p@target ≥ 0.80 : %d/%d (%.1f%%)", n_ge_80, n, 100*n_ge_80/n)
        logger.info("  p@target ≥ 0.99 : %d/%d (%.1f%%)", n_ge_99, n, 100*n_ge_99/n)
        logger.info("  top1 ∈ target   : %d/%d (%.1f%%)",
                    top1_in_target_count, n, 100*top1_in_target_count/n)
        logger.info("=" * 60)

    if args.out_path:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True) if os.path.dirname(args.out_path) else None
        with open(args.out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("逐题结果落盘: %s", args.out_path)


if __name__ == "__main__":
    main()

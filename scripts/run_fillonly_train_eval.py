"""V2-fillonly 实验：训练集仅用 fill_correct（踢出 corr），ep=5。

实验动机：
  V2 4k 下方案 C 仅 +5pp on mistake。诊断: anchor 信号密度只 18.8%
  (fill_correct 1264 / 总训练集 6735)。本实验把 corr_answer 5471 全部踢出，
  训练集 = 仅 1264 条 fill_correct，anchor 信号密度 100%；ep=5 拉高训练量。

  对照 V2 原 β=0.0 (anchor 18.8% / ep=2 / mistake +5.08pp), 看
  "anchor 信号密度 5×" + "训练量 2.5×" 能否把 mistake 增量从 +5pp 推到 +15pp+。

前置条件：复用 V2 原 fill_correct.json (1264 条)，不重跑 fill。

执行（2 卡机）：
    cd /workspace/SDCL_A_TOKEN
    git pull
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_fillonly_train_eval.py

阶段（全部串行）：
    Step 1) 从 fill_correct.json 直接复制成 fillonly 训练集（不 merge corr）
    Step 2) β=0.0 训练 ep=5
    Step 3) β=0.7 训练 ep=5
    Step 4) baseline 评测（V2 4k）— 复用 V2 baseline 的 summary 即可，但稳妥起见再跑一份
    Step 5) β=0.0 评测
    Step 6) β=0.7 评测

预计 ~4-5h。
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODEL_PATH = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"

# V2 原产物（复用，不重跑）
FILL_CORRECT_PATH = "datasets/exam/fill_correct.json"
# 本实验产出
TRAIN_DATA_FILLONLY_PATH = "datasets/exam/a_token_train_data_fillonly.json"

# V2 池（评测用，与 V2 4k 协议一致）
V2_MISTAKE_POOL = "datasets/exam/mistake_DS_MATH_pool.json"
V2_CORR_POOL = "datasets/exam/corr_DS_MATH_pool.json"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_fillonly_ep5_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_fillonly_ep5_{TS}"
CKPT_B00 = os.path.join(OUT_B00, "checkpoint_epoch_5")
CKPT_B07 = os.path.join(OUT_B07, "checkpoint_epoch_5")


def _run(cmd: list[str], stage: str):
    print("\n" + "=" * 70, flush=True)
    print(f"[stage={stage}] START {datetime.now().isoformat()}", flush=True)
    print(f"[stage={stage}] CMD: {' '.join(cmd)}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"[stage={stage}] FAILED rc={proc.returncode} after {dt/60:.1f} min"
        )
    print(f"\n[stage={stage}] DONE duration={dt/60:.1f} min", flush=True)


def _build_fillonly_train_data():
    """从 fill_correct.json 直出训练集，确保每条都有 source / fill_token_id 字段。"""
    src_path = os.path.join(_PROJECT_ROOT, FILL_CORRECT_PATH)
    dst_path = os.path.join(_PROJECT_ROOT, TRAIN_DATA_FILLONLY_PATH)

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"V2 原 fill_correct.json 缺失: {src_path}\n"
            f"前置条件: V2 pipeline 已跑过，确保该文件存在"
        )

    with open(src_path, "r", encoding="utf-8") as f:
        fill_data = json.load(f)

    # 字段兜底（fill_correct.json 本来就应该有 source / fill_token_id，
    # 但 a_token_sdcl.merge_to_train_data 还会补 fill_token_text，这里照搬）
    cleaned = []
    for item in fill_data:
        if item.get("fill_token_id") is None:
            continue
        cleaned.append({
            "question_idx": item.get("question_idx"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "ref_answer": item.get("ref_answer", ""),
            "ref_solution": item.get("ref_solution", ""),
            "source": "fill_correct",
            "fill_token_id": item.get("fill_token_id"),
            "fill_token_text": item.get("fill_token_text"),
        })

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[Step 1] fillonly train_data 已写: {dst_path}", flush=True)
    print(f"[Step 1]   样本数: {len(cleaned)} (V2 原 fill_correct 全量)", flush=True)
    print(f"[Step 1]   anchor 信号密度: 100% (vs V2 原训练集 18.8%)", flush=True)


def _train_cmd(beta: str, output_dir: str) -> list[str]:
    return [
        "python",
        "scripts/train/run_a_token_sdcl_train.py",
        "--model_path", MODEL_PATH,
        "--data_path", TRAIN_DATA_FILLONLY_PATH,
        "--output_dir", output_dir,
        "--num_epochs", "5",
        "--batch_size", "6",
        "--gradient_accumulation_steps", "3",
        "--learning_rate", "1e-5",
        "--max_prompt_length", "2048",
        "--max_answer_length", "4096",
        "--beta_fill", beta,
        "--use_lora",
        "--lora_r", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0.0",
        "--gradient_checkpointing",
        "--log_interval", "10",
        "--save_steps", "500",
        "--save_total_limit", "8",
        "--seed", "42",
    ]


def _eval_cmd(adapter_path: str | None) -> list[str]:
    cmd = [
        "python", "main.py", "eval",
        "--model_path", MODEL_PATH,
    ]
    if adapter_path:
        cmd += ["--adapter_path", adapter_path]
    cmd += [
        "--mistake_path", V2_MISTAKE_POOL,
        "--corr_path", V2_CORR_POOL,
        "--max_prompt_length", "6144",
        "--max_new_tokens", "4096",
        "--math500_roll_k", "8",
        "--math500_roll_temperature", "0.6",
        "--math500_roll_top_p", "0.95",
        "--device_ids", "0,1",
    ]
    return cmd


def main():
    print("=" * 70, flush=True)
    print(f"[run_fillonly_train_eval] START ts={TS}", flush=True)
    print(f"[run_fillonly_train_eval] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_fillonly_train_eval] FILL_CORRECT_PATH = {FILL_CORRECT_PATH} (复用 V2 原产物)", flush=True)
    print(f"[run_fillonly_train_eval] TRAIN_DATA = {TRAIN_DATA_FILLONLY_PATH}", flush=True)
    print(f"[run_fillonly_train_eval] OUT_B00 = {OUT_B00}", flush=True)
    print(f"[run_fillonly_train_eval] OUT_B07 = {OUT_B07}", flush=True)
    print(f"[run_fillonly_train_eval] 训练: ep=5 lr=1e-5 bs=6 gas=3 lora_r=32", flush=True)
    print(f"[run_fillonly_train_eval] 评测协议: V2 4k (max_prompt=6144 max_new=4096)", flush=True)
    print("=" * 70, flush=True)

    # ---------------- Step 1: 构造 fillonly 训练集 ----------------
    print("\n[Step 1] 从 fill_correct.json 直出 fillonly 训练集 ...", flush=True)
    _build_fillonly_train_data()

    # ---------------- Step 2: β=0.0 训练 ----------------
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="train_b00_fillonly_ep5")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[train_b00] 期望 ckpt 不存在: {CKPT_B00}")

    # ---------------- Step 3: β=0.7 训练 ----------------
    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="train_b07_fillonly_ep5")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[train_b07] 期望 ckpt 不存在: {CKPT_B07}")

    # ---------------- Step 4: β=0.0 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B00), stage="eval_b00_fillonly_ep5")

    # ---------------- Step 5: β=0.7 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B07), stage="eval_b07_fillonly_ep5")

    print("\n" + "=" * 70, flush=True)
    print(f"[run_fillonly_train_eval] ALL DONE", flush=True)
    print(f"  CKPT β=0.0: {CKPT_B00}", flush=True)
    print(f"  CKPT β=0.7: {CKPT_B07}", flush=True)
    print(f"  对照: V2 原 baseline (mistake 13.83 / roll-8 73.98)", flush=True)
    print(f"        V2 原 β=0.0 ep=2 corr+fill (mistake 18.91 / roll-8 73.55, +5.08pp)", flush=True)
    print(f"  本次: fillonly ep=5  β=0.0/0.7 → 看 mistake Δ 是否 ≥ +15pp", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

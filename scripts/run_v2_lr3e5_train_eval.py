"""V2 lr 消融：训练集/协议/ep 全部沿用 V2 baseline，唯一改动 lr 1e-5 → 3e-5。

实验动机：
  V2 原 β=0.0 ep=2 lr=1e-5 在 mistake 池只 +5.08pp，用户怀疑"救回率少 = 没学到位"，
  最小改动假设：训练量没变、训练集没变、协议没变，仅 lr 拉高 3×，看 mistake Δ 能否
  从 +5pp 推到 +15pp+。

对照（V2 原口径，同池子同协议）：
  baseline:         mistake 13.83 / math500 roll-8 73.98 / corr 90.46
  V2 β=0.0 lr=1e-5: mistake 18.91 (+5.08) / math500 73.55 (-0.43) / corr 89.73 (-0.73)

本次（lr=3e-5，其余全等）：
  训练集 = datasets/exam/a_token_train_data.json （V2 原 6735 条 corr+fill_correct merged）
  ep=2 / bs=6 / gas=3 / lora_r=32 / α=64 / max_prompt=2048 / max_ans=4096
  β=0.0 + β=0.7 双 ckpt
  评测 V2 4k 协议 (max_prompt=6144 max_new=4096)

执行（2 卡机）：
    cd /workspace/SDCL_A_TOKEN
    git pull
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_v2_lr3e5_train_eval.py

阶段（串行）：
    1) β=0.0 训练
    2) β=0.7 训练
    3) β=0.0 评测
    4) β=0.7 评测
"""

import os
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

# V2 原训练集（不重建，复用）
TRAIN_DATA_PATH = "datasets/exam/a_token_train_data.json"

# V2 原池（评测）
V2_MISTAKE_POOL = "datasets/exam/mistake_DS_MATH_pool.json"
V2_CORR_POOL = "datasets/exam/corr_DS_MATH_pool.json"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_lr3e5_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_lr3e5_{TS}"
CKPT_B00 = os.path.join(OUT_B00, "checkpoint_epoch_2")
CKPT_B07 = os.path.join(OUT_B07, "checkpoint_epoch_2")


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


def _train_cmd(beta: str, output_dir: str) -> list[str]:
    return [
        "python",
        "scripts/train/run_a_token_sdcl_train.py",
        "--model_path", MODEL_PATH,
        "--data_path", TRAIN_DATA_PATH,
        "--output_dir", output_dir,
        "--num_epochs", "2",
        "--batch_size", "6",
        "--gradient_accumulation_steps", "3",
        "--learning_rate", "3e-5",
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
        "--save_total_limit", "5",
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
    print(f"[run_v2_lr3e5_train_eval] START ts={TS}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] TRAIN_DATA = {TRAIN_DATA_PATH} (V2 原 6735 条 corr+fill_correct)", flush=True)
    print(f"[run_v2_lr3e5_train_eval] OUT_B00 = {OUT_B00}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] OUT_B07 = {OUT_B07}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] 唯一改动 vs V2: lr 1e-5 → 3e-5", flush=True)
    print(f"[run_v2_lr3e5_train_eval] 训练: ep=2 bs=6 gas=3 lora_r=32 max_prompt=2048 max_ans=4096", flush=True)
    print(f"[run_v2_lr3e5_train_eval] 评测协议: V2 4k (max_prompt=6144 max_new=4096)", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH)):
        raise FileNotFoundError(
            f"V2 原训练集缺失: {TRAIN_DATA_PATH}\n"
            f"前置条件: V2 pipeline 已跑过"
        )

    # ---------------- Step 1: β=0.0 训练 ----------------
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="train_b00_lr3e5")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[train_b00] 期望 ckpt 不存在: {CKPT_B00}")

    # ---------------- Step 2: β=0.7 训练 ----------------
    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="train_b07_lr3e5")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[train_b07] 期望 ckpt 不存在: {CKPT_B07}")

    # ---------------- Step 3: β=0.0 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B00), stage="eval_b00_lr3e5")

    # ---------------- Step 4: β=0.7 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B07), stage="eval_b07_lr3e5")

    print("\n" + "=" * 70, flush=True)
    print(f"[run_v2_lr3e5_train_eval] ALL DONE", flush=True)
    print(f"  CKPT β=0.0: {CKPT_B00}", flush=True)
    print(f"  CKPT β=0.7: {CKPT_B07}", flush=True)
    print(f"  对照 V2 baseline:           mistake 13.83 / roll-8 73.98 / corr 90.46", flush=True)
    print(f"  对照 V2 β=0.0 lr=1e-5 ep=2: mistake 18.91 (+5.08) / roll-8 73.55 / corr 89.73", flush=True)
    print(f"  本次 lr=3e-5 (ep/数据/协议全等): 看 mistake Δ 能否 ≥ +15pp", flush=True)
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

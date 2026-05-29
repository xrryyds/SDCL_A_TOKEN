"""目标：mistake → 90%。corr / math500 不管。

策略组合：
    1) 扩 fill_correct 池：fill_epoch 3 → 10 轮累积并集（预期 1264 → 1800+）
    2) 重新合并 train_data
    3) ep=5 + β=0.7 + bs=6 + gas=3 训练（强信号 + 强训练量）
    4) V2 4k 协议评测，盯 mistake 数字

执行：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_mistake_chase_90.py

阶段（全部串行，前一步失败后续不跑，但 use_worker 一定进入）：
    1) 扩 fill：generate_fill_correct fill_epoch=10（~1.5h）
    2) 合并 train_data（秒级）
    3) 训练 ep=5 β=0.7（~3.5h）
    4) 评测 V2 4k（~30min）
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

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_ep5_chase90_{TS}"
CKPT_EXPECTED = os.path.join(OUTPUT_DIR, "checkpoint_epoch_5")


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


def main():
    print("=" * 70, flush=True)
    print(f"[chase90] START ts={TS}", flush=True)
    print(f"[chase90] OUTPUT_DIR = {OUTPUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    # ---------------- Step 1: 扩 fill_correct 池（fill_epoch=10）+ 合并 ----------------
    # main.py run_a_token_sdcl_pipeline 三步流水线:
    #   step1: generate_fill_correct → fill_correct.json
    #   step2: 合并 corr + fill_correct → a_token_train_data.json
    #   step3: train  (此处用 skip_step3 跳过, 自己手动训练)
    _run(
        [
            "python", "main.py", "pipeline",
            "--fill_epoch", "10",
            "--max_prompt_length", "6144",
            "--max_new_tokens", "4096",
            "--mistake_path", "datasets/exam/mistake_DS_MATH_pool.json",
            "--corr_answer_path", "datasets/exam/corr_DS_MATH_pool.json",
            "--skip_step3",  # 不跑训练，自己跑
        ],
        stage="expand_fill_pool_to_ep10",
    )

    # ---------------- Step 2: 训练 ep=5 + β=0.7 ----------------
    _run(
        [
            "python",
            "scripts/train/run_a_token_sdcl_train.py",
            "--model_path", MODEL_PATH,
            "--data_path", "datasets/exam/a_token_train_data.json",
            "--output_dir", OUTPUT_DIR,
            "--num_epochs", "5",
            "--batch_size", "6",
            "--gradient_accumulation_steps", "3",
            "--learning_rate", "1e-5",
            "--max_prompt_length", "2048",
            "--max_answer_length", "4096",
            "--beta_fill", "0.7",
            "--use_lora",
            "--lora_r", "32",
            "--lora_alpha", "64",
            "--lora_dropout", "0.0",
            "--gradient_checkpointing",
            "--log_interval", "10",
            "--save_steps", "500",
            "--save_total_limit", "8",
            "--seed", "42",
        ],
        stage="train_ep5_b07_chase90",
    )

    if not os.path.exists(CKPT_EXPECTED):
        raise FileNotFoundError(
            f"[train_ep5_b07_chase90] 期望 ckpt 不存在: {CKPT_EXPECTED}"
        )

    # ---------------- Step 3: 评测 V2 4k 协议 ----------------
    _run(
        [
            "python", "main.py", "eval",
            "--model_path", MODEL_PATH,
            "--adapter_path", CKPT_EXPECTED,
            "--mistake_path", "datasets/exam/mistake_DS_MATH_pool.json",
            "--corr_path", "datasets/exam/corr_DS_MATH_pool.json",
            "--max_prompt_length", "6144",
            "--max_new_tokens", "4096",
            "--math500_roll_k", "8",
            "--math500_roll_temperature", "0.6",
            "--math500_roll_top_p", "0.95",
            "--device_ids", "0,1",
        ],
        stage="eval_b07_ep5_V2_4k",
    )

    print("\n" + "=" * 70, flush=True)
    print(f"[chase90] ALL DONE", flush=True)
    print(f"  ckpt: {CKPT_EXPECTED}", flush=True)
    print(f"  目标: mistake → 90%（实际看 output/eval_lora_*/summary.json）", flush=True)
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

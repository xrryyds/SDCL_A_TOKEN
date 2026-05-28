"""一次性跑完 β=0.7-V2 训练 + V2 口径评测 + V1 8k 口径评测（β=0.0-V2 已有 ckpt）。

跑完或任何阶段异常都调 use_worker() 挂卡保活。

执行：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_p0_p2_train_and_eval.py

阶段：
    1) P0 训练：β=0.7-V2，2 卡 DDP，2ep（~1.5h）
    2) P0 评测：β=0.7-V2 ckpt @ V2 口径 6144+4096（~30min）
    3) P2 评测：β=0.0-V2 ckpt @ V1 口径 10240+8192（~50min）

所有阶段串行。前一步失败后续步骤不会跑，但 use_worker 一定会进入。
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
B00_CKPT = "/workspace/SDCL_A_TOKEN/output/a_token_betaC_b00_mathV2_20260528_115209/checkpoint_epoch_2"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
B07_OUTPUT_DIR = f"/workspace/SDCL_A_TOKEN/output/a_token_betaC_b07_mathV2_{TS}"
B07_CKPT_EXPECTED = os.path.join(B07_OUTPUT_DIR, "checkpoint_epoch_2")


def _run(cmd: list[str], stage: str):
    """跑一条 shell 命令，非零退出码抛 RuntimeError。"""
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
    print(f"[run_p0_p2] START ts={TS}", flush=True)
    print(f"[run_p0_p2] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_p0_p2] B07_OUTPUT_DIR = {B07_OUTPUT_DIR}", flush=True)
    print(f"[run_p0_p2] B00 ckpt (P2)  = {B00_CKPT}", flush=True)
    print("=" * 70, flush=True)

    # ---------------- P0 训练 ----------------
    _run(
        [
            "python",
            "scripts/train/run_a_token_sdcl_train.py",
            "--model_path", MODEL_PATH,
            "--data_path", "datasets/exam/a_token_train_data.json",
            "--output_dir", B07_OUTPUT_DIR,
            "--num_epochs", "2",
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
            "--save_total_limit", "5",
            "--seed", "42",
        ],
        stage="P0_train_b07_V2",
    )

    if not os.path.exists(B07_CKPT_EXPECTED):
        raise FileNotFoundError(
            f"[P0_train_b07_V2] 期望的 ckpt 不存在: {B07_CKPT_EXPECTED}"
        )

    # ---------------- P0 评测：β=0.7-V2 @ V2 口径 6144+4096 ----------------
    _run(
        [
            "python", "main.py", "eval",
            "--model_path", MODEL_PATH,
            "--adapter_path", B07_CKPT_EXPECTED,
            "--mistake_path", "datasets/exam/mistake_DS_MATH_pool.json",
            "--corr_path", "datasets/exam/corr_DS_MATH_pool.json",
            "--max_prompt_length", "6144",
            "--max_new_tokens", "4096",
            "--math500_roll_k", "8",
            "--math500_roll_temperature", "0.6",
            "--math500_roll_top_p", "0.95",
            "--device_ids", "0,1",
        ],
        stage="P0_eval_b07_V2_at_6144_4096",
    )

    # ---------------- P2 评测：β=0.0-V2 @ V1 口径 10240+8192 ----------------
    _run(
        [
            "python", "main.py", "eval",
            "--model_path", MODEL_PATH,
            "--adapter_path", B00_CKPT,
            "--mistake_path", "datasets/exam/mistake_DS_MATH_pool.json",
            "--corr_path", "datasets/exam/corr_DS_MATH_pool.json",
            "--max_prompt_length", "10240",
            "--max_new_tokens", "8192",
            "--math500_roll_k", "8",
            "--math500_roll_temperature", "0.6",
            "--math500_roll_top_p", "0.95",
            "--device_ids", "0,1",
        ],
        stage="P2_eval_b00_V2_at_10240_8192",
    )

    print("\n" + "=" * 70, flush=True)
    print(f"[run_p0_p2] ALL DONE", flush=True)
    print(f"  P0 ckpt: {B07_CKPT_EXPECTED}", flush=True)
    print(f"  output/eval_*  下应有 3 份 summary（baseline + b07_V2 + b00_V2@8k 评测目录）", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        # 不论正常结束还是异常，都进入 use_worker 挂卡保活
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

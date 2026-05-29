"""V2-short 实验 Phase B + C + D：fill pipeline → β=0.0/β=0.7 训练 → 3 份评测。

前置条件：scripts/rebuild_math_pool_short_roll8.py 已跑完，已产出：
  - datasets/exam/mistake_DS_MATH_pool_short_roll8.json
  - datasets/exam/corr_DS_MATH_pool_short.json

执行：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_v2_short_full.py

阶段（全部串行，前一步失败后续不跑，但 use_worker 一定进入）：
    B) fill pipeline + merge train_data（新池 + 4096+2048 协议）
    C1) β=0.0 训练 ep=2 max_prompt_length=2048 max_answer_length=2048
    C2) β=0.7 训练 ep=2（同协议）
    D1) baseline 评测 @ 4096+2048
    D2) β=0.0 ckpt 评测 @ 4096+2048
    D3) β=0.7 ckpt 评测 @ 4096+2048
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

NEW_POOL_MISTAKE = "datasets/exam/mistake_DS_MATH_pool_short_roll8.json"
NEW_POOL_CORR = "datasets/exam/corr_DS_MATH_pool_short.json"
FILL_CORRECT_PATH = "datasets/exam/fill_correct_short.json"
TRAIN_DATA_PATH = "datasets/exam/a_token_train_data_short.json"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_short_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_short_{TS}"
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
        "--learning_rate", "1e-5",
        "--max_prompt_length", "2048",
        "--max_answer_length", "2048",
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
        "--mistake_path", NEW_POOL_MISTAKE,
        "--corr_path", NEW_POOL_CORR,
        "--max_prompt_length", "4096",
        "--max_new_tokens", "2048",
        "--math500_roll_k", "8",
        "--math500_roll_temperature", "0.6",
        "--math500_roll_top_p", "0.95",
        "--device_ids", "0,1",
    ]
    return cmd


def main():
    print("=" * 70, flush=True)
    print(f"[run_v2_short_full] START ts={TS}", flush=True)
    print(f"[run_v2_short_full] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_v2_short_full] mistake pool = {NEW_POOL_MISTAKE}", flush=True)
    print(f"[run_v2_short_full] corr    pool = {NEW_POOL_CORR}", flush=True)
    print(f"[run_v2_short_full] OUT_B00 = {OUT_B00}", flush=True)
    print(f"[run_v2_short_full] OUT_B07 = {OUT_B07}", flush=True)
    print("=" * 70, flush=True)

    if not (os.path.exists(os.path.join(_PROJECT_ROOT, NEW_POOL_MISTAKE))
            and os.path.exists(os.path.join(_PROJECT_ROOT, NEW_POOL_CORR))):
        raise FileNotFoundError(
            f"前置池缺失: {NEW_POOL_MISTAKE} 或 {NEW_POOL_CORR}\n"
            f"请先跑 scripts/rebuild_math_pool_short_roll8.py"
        )

    # ---------------- Phase B：fill pipeline + merge ----------------
    _run(
        [
            "python", "main.py", "pipeline",
            "--mistake_path", NEW_POOL_MISTAKE,
            "--corr_answer_path", NEW_POOL_CORR,
            "--fill_correct_path", FILL_CORRECT_PATH,
            "--train_data_path", TRAIN_DATA_PATH,
            "--fill_prompt_len", "4096",
            "--fill_max_gen_token", "2048",
            "--fill_epoch", "3",
            "--skip_train",
        ],
        stage="B_fill_pipeline_short",
    )

    # ---------------- Phase C1：β=0.0 训练 ----------------
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="C1_train_b00_short")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[C1] 期望 ckpt 不存在: {CKPT_B00}")

    # ---------------- Phase C2：β=0.7 训练 ----------------
    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="C2_train_b07_short")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[C2] 期望 ckpt 不存在: {CKPT_B07}")

    # ---------------- Phase D1：baseline 评测 ----------------
    _run(_eval_cmd(adapter_path=None), stage="D1_eval_baseline_short")

    # ---------------- Phase D2：β=0.0 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B00), stage="D2_eval_b00_short")

    # ---------------- Phase D3：β=0.7 评测 ----------------
    _run(_eval_cmd(adapter_path=CKPT_B07), stage="D3_eval_b07_short")

    print("\n" + "=" * 70, flush=True)
    print(f"[run_v2_short_full] ALL DONE", flush=True)
    print(f"  CKPT β=0.0: {CKPT_B00}", flush=True)
    print(f"  CKPT β=0.7: {CKPT_B07}", flush=True)
    print(f"  output/eval_lora_*  下应有 3 份新 summary（baseline + b00 + b07）", flush=True)
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

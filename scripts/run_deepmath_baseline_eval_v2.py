"""4 卡机：DeepMath 池 baseline @ V2 4k 协议（6144+4096）评测。

为了让 DeepMath β=0.5 ckpt 的 mistake/corr 数字有同协议对照组，必须先跑这一次。

执行（4 卡机）：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/run_deepmath_baseline_eval_v2.py

口径：
    - 无 LoRA（baseline）
    - mistake/corr 路径用 4 卡机本地的 DeepMath 池（59171 / 43849）
    - max_prompt_length=6144 (= 2048 prompt + 4096 gen vLLM 总窗口)
    - max_new_tokens=4096
    - math500 roll-8, T=0.6, top_p=0.95
    - 4 卡并行

跑完或异常都调 use_worker() 挂卡保活。
预计 ~50min（103k 题 / 4 卡）。
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
    print(f"[deepmath_baseline_v2] START ts={TS}", flush=True)
    print(f"[deepmath_baseline_v2] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print("=" * 70, flush=True)

    _run(
        [
            "python", "main.py", "eval",
            "--model_path", MODEL_PATH,
            # 不传 --adapter_path，纯 base model
            "--mistake_path", "datasets/exam/mistake_DS_MATH_pool.json",
            "--corr_path", "datasets/exam/corr_DS_MATH_pool.json",
            "--max_prompt_length", "6144",
            "--max_new_tokens", "4096",
            "--math500_roll_k", "8",
            "--math500_roll_temperature", "0.6",
            "--math500_roll_top_p", "0.95",
            "--device_ids", "0,1,2,3",
        ],
        stage="deepmath_baseline_V2_at_6144_4096",
    )

    print("\n" + "=" * 70, flush=True)
    print(f"[deepmath_baseline_v2] ALL DONE", flush=True)
    print(f"  output/eval_*  下应有一份新 summary（DeepMath baseline @ V2 4k）", flush=True)
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

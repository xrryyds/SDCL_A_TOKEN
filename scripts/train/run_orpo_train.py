"""DDP launcher for orpo_train.py (单命令启动, 不需 torchrun)。

用法 (4 卡 H800):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train/run_orpo_train.py \\
        --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \\
        --data_path datasets/train/train_data_orpo.json \\
        --output_dir output/orpo_4card_$(date +%Y%m%d_%H%M%S) \\
        --num_epochs 1 \\
        --batch_size 4 \\
        --gradient_accumulation_steps 8 \\
        --learning_rate 2e-5 \\
        --alpha 0.2

进程数自动按可见 GPU 数选; 可用 --nproc 显式覆盖。
所有未识别参数原样转发给 orpo_train.py。
训练完成或异常都调 use_worker() 挂卡保活。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import List

import torch
from torch.distributed.run import main as torchrun_main


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_SCRIPT = os.path.join(_THIS_DIR, "orpo_train.py")


def _decide_nproc(explicit) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError(
            "未检测到可见 GPU; 请设置 CUDA_VISIBLE_DEVICES 或确认驱动环境。"
        )
    return n


def _launch_training():
    parser = argparse.ArgumentParser(
        description="单命令启动 orpo_train.py 的 DDP 训练 (无需 torchrun)。",
        add_help=True,
    )
    parser.add_argument("--nproc", type=int, default=None,
                        help="DDP 进程数; 默认 = torch.cuda.device_count()。")
    parser.add_argument("--master_port", type=str,
                        default=os.environ.get("MASTER_PORT", "29503"),
                        help="rendezvous 端口, 默认 29503 (避开 sdcl=29500/sdft=29501/grpo=29502)。")
    args, forwarded = parser.parse_known_args()

    nproc = _decide_nproc(args.nproc)
    print(
        f"[run_orpo_train] world_size={nproc}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"master_port={args.master_port}",
        flush=True,
    )

    # GPU 健康检查 (照搬 sdcl launcher)
    print("[run_orpo_train] 健康检查 GPU init...", flush=True)
    for i in range(nproc):
        try:
            with torch.cuda.device(i):
                _t = torch.zeros(1024, device=f"cuda:{i}")
                _ = (_t + 1).sum().item()
                torch.cuda.synchronize(i)
                del _t
            print(f"  cuda:{i} OK", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"GPU {i} 初始化失败: {type(e).__name__}: {e}\n"
                f"很可能上次训练崩溃后 driver 残留脏 context。处理:\n"
                f"  1) ps -ef | grep python | grep -v grep\n"
                f"  2) fuser -v /dev/nvidia*\n"
                f"  3) kill -9 占用 PID\n"
                f"  4) nvidia-smi --gpu-reset -i {i}\n"
            ) from e
    print("[run_orpo_train] GPU 健康检查通过。", flush=True)

    torchrun_argv: List[str] = [
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_port={args.master_port}",
        _TRAIN_SCRIPT,
        *forwarded,
    ]

    sys.argv = ["torchrun", *torchrun_argv]
    torchrun_main(torchrun_argv)


def main():
    overall = "ok"
    top_err = None
    try:
        _launch_training()
    except BaseException as e:
        overall = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("\n" + "=" * 60, flush=True)
        print("训练异常:", flush=True)
        print(top_err, flush=True)
        print("=" * 60, flush=True)
    finally:
        print("\n" + "=" * 60, flush=True)
        print(f"训练状态: {overall}", flush=True)
        if top_err:
            print("异常详情见上方输出", flush=True)
        print("=" * 60, flush=True)
        print("\n进入 use_worker 保活 (Ctrl+C 退出)...", flush=True)
        try:
            from main import use_worker
            use_worker()
        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()

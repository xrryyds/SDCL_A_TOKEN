"""DDP launcher for a_token_sdcl_train.py.

用法（与原 torchrun 命令等价，但只用 python 启动）：

    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/train/run_a_token_sdcl_train.py \
        --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
        --data_path datasets/exam/a_token_train_data.json \
        --output_dir output/a_token_sdcl_ddp_$(date +%Y%m%d_%H%M%S) \
        --num_epochs 3 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --ce_weight 1.0

进程数自动按可见 GPU 数（torch.cuda.device_count()）选；可用 --nproc 显式覆盖。
所有未识别参数会原样转发给 a_token_sdcl_train.py。

训练完成或异常都会调 use_worker() 挂卡保活。
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
_TRAIN_SCRIPT = os.path.join(_THIS_DIR, "a_token_sdcl_train.py")


def _decide_nproc(explicit: int | None) -> int:
    if explicit is not None and explicit > 0:
        return explicit
    n = torch.cuda.device_count()
    if n <= 0:
        raise RuntimeError(
            "未检测到可见 GPU；请设置 CUDA_VISIBLE_DEVICES 或确认驱动环境。"
        )
    return n


def _launch_training():
    """训练主逻辑（原 main 函数内容）。"""
    parser = argparse.ArgumentParser(
        description="单命令启动 a_token_sdcl_train.py 的 DDP 训练（无需 torchrun）。",
        add_help=True,
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="DDP 进程数；默认 = torch.cuda.device_count()。",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default=os.environ.get("MASTER_PORT", "29500"),
        help="rendezvous 端口，默认 29500。",
    )
    args, forwarded = parser.parse_known_args()

    nproc = _decide_nproc(args.nproc)
    print(
        f"[run_a_token_sdcl_train] world_size={nproc}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"master_port={args.master_port}",
        flush=True,
    )

    # 健康检查：每张可见 GPU 都能成功创建一个小张量并做一次 cuda.synchronize。
    # 上次训练崩溃 / kill -9 后,driver 层可能残留脏 context 占着显存,
    # 此时启动 N 个 DDP worker,会有部分 worker init CUDA 失败 abort,
    # 剩下 worker 等不到对端 → 训练在第一个 all_reduce 死锁(2h+ 0 step)。
    # 这里提前发现,直接 fail-fast 而不是带病往下跑。
    print("[run_a_token_sdcl_train] 健康检查 GPU init...", flush=True)
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
                f"GPU {i} 初始化失败：{type(e).__name__}: {e}\n"
                f"很可能是上次训练崩溃后 driver 残留了脏 context。处理：\n"
                f"  1) ps -ef | grep python | grep -v grep   # 看有没有僵尸进程\n"
                f"  2) fuser -v /dev/nvidia*                  # 看占用 PID\n"
                f"  3) 把所有占用 PID kill -9\n"
                f"  4) nvidia-smi --gpu-reset -i {i}          # 仍占着就重置该卡\n"
                f"  5) 实在不行重启 devbox。"
            ) from e
    print("[run_a_token_sdcl_train] GPU 健康检查通过。", flush=True)

    # 构造等价的 torchrun argv
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
    """入口：训练 + finally use_worker 保活。"""
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
        print("\n进入 use_worker 保活（Ctrl+C 退出）...", flush=True)
        try:
            from main import use_worker
            use_worker()
        except BaseException:
            traceback.print_exc()


if __name__ == "__main__":
    main()

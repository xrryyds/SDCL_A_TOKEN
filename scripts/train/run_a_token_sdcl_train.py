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
"""

from __future__ import annotations

import argparse
import os
import sys
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


def main():
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


if __name__ == "__main__":
    main()

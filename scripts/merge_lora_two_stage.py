"""merge_lora_two_stage.py — 把 Base + stage1 LoRA + stage2 LoRA 合并为一个完整模型目录.

用途:
  stage2 LoRA 是基于 (Base + stage1 LoRA merged) 训的, 直接喂 vLLM 时双 LoRA
  组合不好搞. 这个脚本把两层 LoRA 都 merge 进 Base 权重, 输出一个标准 HF 模型目录,
  vLLM / transformers / eval_v3.py 都能直接当 Base 用.

流程:
  1. 加载 Base
  2. PeftModel.from_pretrained(Base, stage1_lora) → merge_and_unload() → 含 stage1 的 Base
  3. PeftModel.from_pretrained(.., stage2_lora) → merge_and_unload() → 含 stage1+stage2 的 Base
  4. save_pretrained 到 out_dir
  5. 复制 tokenizer (从 base_model_path)

用法:
  python scripts/merge_lora_two_stage.py \
    --base_model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
    --stage1_lora_path output/pool_dist_v1_20260603_110839/checkpoint_epoch_2 \
    --stage2_lora_path output/sdft_v3_stage2_20260603_153927/checkpoint_epoch_2 \
    --out_dir output/sdft_v3_stage2_merged
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def merge_one_lora(model, lora_path: str, label: str):
    logger.info("加载 %s LoRA: %s", label, lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    logger.info("merge_and_unload %s LoRA ...", label)
    model = model.merge_and_unload()
    logger.info("%s LoRA 已 merge", label)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--stage1_lora_path", type=str, default=None,
                        help="第一层 LoRA (pool_dist). 留空跳过 stage1 直接 merge stage2.")
    parser.add_argument("--stage2_lora_path", type=str, default=None,
                        help="第二层 LoRA (sdft_v3_stage2). 留空只 merge stage1.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="merge 用的 device. CPU 也行但慢.")
    args = parser.parse_args()

    if not args.stage1_lora_path and not args.stage2_lora_path:
        raise ValueError("至少要指定 stage1_lora_path 或 stage2_lora_path 其中之一")

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise ValueError(
            f"out_dir 已存在且非空: {args.out_dir} (避免覆盖,请删掉或换路径)"
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    logger.info("加载 Base: %s", args.base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    if args.stage1_lora_path:
        model = merge_one_lora(model, args.stage1_lora_path, "stage1")

    if args.stage2_lora_path:
        model = merge_one_lora(model, args.stage2_lora_path, "stage2")

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("保存合并后模型 → %s", args.out_dir)
    model.save_pretrained(args.out_dir)

    # 复制 tokenizer (从 Base 拿就行, LoRA 不改 tokenizer)
    logger.info("保存 tokenizer (来自 Base)")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.out_dir)

    # 复制 Base 里所有非权重 / 非 config 的辅助文件 (自定义 .py / chat_template / generation_config 等)
    # 防止 R1-Distill 这种 trust_remote_code 模型缺 modeling_*.py / configuration_*.py
    import shutil
    SKIP_SUFFIX = (".safetensors", ".bin", ".pt", ".pth")
    SKIP_EXACT = {"pytorch_model.bin.index.json", "model.safetensors.index.json"}
    n_copied = 0
    for fname in os.listdir(args.base_model_path):
        if fname.startswith("."):
            continue
        if any(fname.endswith(suf) for suf in SKIP_SUFFIX):
            continue
        if fname in SKIP_EXACT:
            continue
        src = os.path.join(args.base_model_path, fname)
        dst = os.path.join(args.out_dir, fname)
        if os.path.isdir(src):
            continue  # 不递归子目录, 避免复制过多内容
        if os.path.exists(dst):
            continue  # save_pretrained 已写过的, 不覆盖 (config.json 等)
        shutil.copy2(src, dst)
        logger.info("  复制辅助文件: %s", fname)
        n_copied += 1
    logger.info("从 Base 复制了 %d 个辅助文件", n_copied)

    logger.info("=" * 60)
    logger.info("Done. 输出目录: %s", args.out_dir)
    logger.info("现在可以直接当 Base 用, 比如:")
    logger.info("  python scripts/verify_sdft_hint_effect.py --model_path %s",
                args.out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

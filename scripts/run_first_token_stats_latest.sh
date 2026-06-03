#!/bin/bash
# 一键: 找最新 eval_v3_* 目录, 对里面所有 5 个数据集的 LoRA greedy jsonl 做首 token 分布统计
#
# 用法: bash scripts/run_first_token_stats_latest.sh
# 可选: bash scripts/run_first_token_stats_latest.sh <eval_dir>  (手动指定目录)

set -e

cd /workspace/SDCL_A_TOKEN

if [ -n "$1" ]; then
    EVAL_DIR="$1"
else
    EVAL_DIR=$(ls -td output/eval_v3_*/ 2>/dev/null | head -1)
fi

if [ -z "$EVAL_DIR" ] || [ ! -d "$EVAL_DIR" ]; then
    echo "找不到 eval 目录, 用法:"
    echo "  bash scripts/run_first_token_stats_latest.sh [eval_dir]"
    exit 1
fi

# 去掉末尾斜杠, 让显示更整洁
EVAL_DIR="${EVAL_DIR%/}"

echo "========================================"
echo " eval_dir: $EVAL_DIR"
echo "========================================"
echo

# 列一下里面有什么
ls -la "$EVAL_DIR"/*.jsonl 2>/dev/null | head -10

echo
echo "========================================"
echo " 跑首 token 分布统计 (单卡 ~30s)"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python scripts/first_token_stats_all.py \
  --eval_dir "$EVAL_DIR" \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B

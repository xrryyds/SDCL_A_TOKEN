#!/bin/bash
# 测试 GRPO 3 种 IS 配置，对比 ratio/grad_norm，定位 ratio=0.0005 根因
# 用法: bash scripts/test_grpo_is_configs.sh
# 注: 不用 set -e, 让某个配置崩了也能继续跑剩下的

MODEL_PATH="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"
GPUS="0,1,2,3"  # 4 卡机器全用
MAX_TRAIN=20
LOG_INTERVAL=1

# 关键: 设 NO_WORKER=1 让每个训练跑完直接退出, 不进 use_worker 保活,
# 否则脚本会卡在第一个配置永远不跑 B/C
export NO_WORKER=1

echo "=========================================="
echo " GRPO IS 配置对比测试"
echo "=========================================="
echo "GPU: $GPUS"
echo "MAX_TRAIN: $MAX_TRAIN"
echo ""

cd /workspace/SDCL_A_TOKEN
git pull

# 清理旧 log
rm -rf logs/grpo_ranks/*

# ============ 配置 A: TRL 默认 (sequence_mask, cap=3) ============
echo ""
echo "[1/3] 配置 A: TRL 默认 (sequence_mask, cap=3.0)"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python scripts/train/run_grpo_a_token_train.py \
  --model_path $MODEL_PATH \
  --output_dir output/grpo_debug_A_default \
  --max_train $MAX_TRAIN \
  --num_epochs 1 \
  --log_interval $LOG_INTERVAL \
  --vllm_is_correction default \
  2>&1 | tee /tmp/grpo_A.log

# 提取指标
LOG_A=$(ls -td logs/grpo_ranks/*/ | head -1)
echo "Log dir: $LOG_A"
grep "sampling/importance_sampling_ratio/mean\|grad_norm\|reward_correctness/mean" \
  $LOG_A/attempt_0/0/stdout.log | tail -20 > /tmp/grpo_A_metrics.txt
echo "指标已保存到 /tmp/grpo_A_metrics.txt"

# 清理 log + vLLM 残留避免混淆
rm -rf logs/grpo_ranks/*
pkill -9 -f vllm 2>/dev/null; pkill -9 -f run_grpo 2>/dev/null; sleep 5

# ============ 配置 B: 关闭 IS ============
echo ""
echo "[2/3] 配置 B: 关闭 IS correction"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python scripts/train/run_grpo_a_token_train.py \
  --model_path $MODEL_PATH \
  --output_dir output/grpo_debug_B_off \
  --max_train $MAX_TRAIN \
  --num_epochs 1 \
  --log_interval $LOG_INTERVAL \
  --vllm_is_correction off \
  2>&1 | tee /tmp/grpo_B.log

LOG_B=$(ls -td logs/grpo_ranks/*/ | head -1)
echo "Log dir: $LOG_B"
grep "grad_norm\|reward_correctness/mean" \
  $LOG_B/attempt_0/0/stdout.log | tail -20 > /tmp/grpo_B_metrics.txt
echo "指标已保存到 /tmp/grpo_B_metrics.txt (无 IS ratio)"

rm -rf logs/grpo_ranks/*
pkill -9 -f vllm 2>/dev/null; pkill -9 -f run_grpo 2>/dev/null; sleep 5

# ============ 配置 C: token_truncate, cap=10 ============
echo ""
echo "[3/3] 配置 C: token_truncate, cap=10.0"
echo "----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPUS python scripts/train/run_grpo_a_token_train.py \
  --model_path $MODEL_PATH \
  --output_dir output/grpo_debug_C_token_truncate \
  --max_train $MAX_TRAIN \
  --num_epochs 1 \
  --log_interval $LOG_INTERVAL \
  --vllm_is_correction token_truncate \
  --vllm_is_cap 10.0 \
  2>&1 | tee /tmp/grpo_C.log

LOG_C=$(ls -td logs/grpo_ranks/*/ | head -1)
echo "Log dir: $LOG_C"
grep "sampling/importance_sampling_ratio/mean\|grad_norm\|reward_correctness/mean" \
  $LOG_C/attempt_0/0/stdout.log | tail -20 > /tmp/grpo_C_metrics.txt
echo "指标已保存到 /tmp/grpo_C_metrics.txt"

# ============ 汇总对比 ============
echo ""
echo "=========================================="
echo " 汇总对比"
echo "=========================================="
echo ""
echo "配置 A (TRL 默认 sequence_mask):"
echo "----------------------------------------"
cat /tmp/grpo_A_metrics.txt
echo ""
echo "配置 B (关闭 IS):"
echo "----------------------------------------"
cat /tmp/grpo_B_metrics.txt
echo ""
echo "配置 C (token_truncate cap=10):"
echo "----------------------------------------"
cat /tmp/grpo_C_metrics.txt
echo ""
echo "=========================================="
echo " 判断标准"
echo "=========================================="
echo "- 如果 A 的 ratio mean ≈ 0.0005, grad_norm ≈ 1e-6 → 复现问题 ✓"
echo "- 如果 B 的 grad_norm 涨到 0.01-1 → IS 是罪魁, 关掉就能学"
echo "- 如果 C 的 ratio mean ≈ 1, grad_norm 正常 → token_truncate 是修法"
echo "- 如果 B/C 的 grad_norm 还是 1e-6 → IS 不是根因, 继续查别的"
echo ""
echo "完整 log 在 /tmp/grpo_{A,B,C}.log"
echo "=========================================="

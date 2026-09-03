#!/bin/bash
# Measure the model's first-token distribution on the real gsm8k_math training mix.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/xiongrengrong.xrr/miniconda3/envs/xrr_new/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p runs/fill_multiround
LOG=runs/fill_multiround/measure_dist.log
exec > >(tee "$LOG") 2>&1
exec "$PY" scripts/measure_first_token_distribution.py \
  --data_path sdpo/SDPO_official/datasets/gsm8k_math/train.json \
  --scaffold "" \
  --topk 60 --report_tokens 200 --batch_size 16 \
  --pool_path datasets/first_token_candidates_math_8.json \
  --out_path datasets/first_token_distribution_gsm8k_math.json

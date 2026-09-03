#!/bin/bash
# Full 467-step GRPO+FILL run at the baseline grid, multi-round FILL (23-token pool, 3 rounds).
# Comparison target: runs/grpo_gsm8k_math/baseline_grpo_467steps.log
#   GSM8K 95.43 final / 95.91 peak, MATH-500 84.95 final / 86.05 peak
cd "$(dirname "$0")" || exit 1
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround/full_467steps.log
exec > >(tee "$LOG") 2>&1

export RUN_TAG=fillfull-r3-coef4e3
exec bash run_local_grpo_fill.sh full

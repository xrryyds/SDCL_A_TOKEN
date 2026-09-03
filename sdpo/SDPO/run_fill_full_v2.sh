#!/bin/bash
# Rerun of the full 467-step GRPO+FILL arm after the step-384 CUDA OOM.
# The OOM came from the SDPO teacher forward, which SRPO_DISABLE_SDPO=1 made dead weight
# (dp_actor.py sdpo_branch_active); peak memory was pinned at 88.47 of 95.07 GB for the
# whole crashed run. No checkpoint existed (save_freq=0), so this is a rerun from step 0.
#
# Comparison target: runs/grpo_gsm8k_math/baseline_grpo_467steps.log
#   GSM8K 95.43 final / 95.91 peak@450, MATH-500 84.95 final / 86.05 peak@450
# The open question is the ceiling: steps 0-350 already landed at parity (offset-adjusted).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround/full_467steps_v2.log
exec > >(tee "$LOG") 2>&1
export RUN_TAG=fillfull-r3-nosdpofwd
exec bash run_local_grpo_fill.sh full

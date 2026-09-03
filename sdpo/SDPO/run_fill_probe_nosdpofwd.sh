#!/bin/bash
# Neutrality + memory regression for skipping the SDPO teacher forward (dp_actor.py
# sdpo_branch_active). Same 10 steps as run_fill_probe.sh, so the metrics are directly
# comparable against runs/fill_multiround/probe_10steps.log:
#   critic/score/mean, actor/entropy, srpo/lambda_grpo=1.0, srpo/fill_loss,
#   srpo/fill_token_cnt, actor/grad_norm  -- must match.
#   perf/max_memory_allocated_gb -- must drop well below the 73.3 the old probe hit.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround/probe_10steps_nosdpofwd.log
exec > >(tee "$LOG") 2>&1
export TOTAL_STEPS=10
export TEST_FREQ=0
export VAL_BEFORE_TRAIN=False
export RUN_TAG=fillprobe-r3-nosdpofwd
exec bash run_local_grpo_fill.sh probe

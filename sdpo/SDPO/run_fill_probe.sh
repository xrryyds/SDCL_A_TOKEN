#!/bin/bash
# 10-step validation-free probe: measure the per-attempt FILL rescue rate.
#
# ~10 steps x ~2.5 dead groups x 24 attempts (8 slots x 3 rounds) ~= 600 attempts. If none
# succeeds, p < 0.005 at 95% and first-token forcing is dead on GSM8K+MATH; that is the
# decision this probe exists to make, before spending 20-24 h on a full run.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround/probe_10steps.log
exec > >(tee "$LOG") 2>&1
export TOTAL_STEPS=10
export TEST_FREQ=0
export VAL_BEFORE_TRAIN=False
export RUN_TAG=fillprobe-r3
exec bash run_local_grpo_fill.sh probe

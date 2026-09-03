#!/bin/bash
# One-token-per-round FILL, 467 steps.
#
# Each round spends all 8 free slots on a SINGLE candidate: round 0 forces t_1 eight
# times, and only groups still all-fail move to t_2, up to t_8. So a dead group gets up
# to 64 attempts, and rescue/round{r}/n_winners is the per-token revival rate for
# candidate r -- the 8x8 layout gave each token exactly one attempt per group, which
# cannot separate token quality from attempt count.
#
# t_1..t_8 of the 23-token pool (descending prob_mean): We Let To Sure When You The In.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_multiround/full_467steps_t1x8.log
exec > >(tee "$LOG") 2>&1
export N_TOKENS_PER_GROUP=1
export FILL_ROUNDS=8
export RUN_TAG=fillfull-t1x8
exec bash run_local_grpo_fill.sh full

#!/bin/bash
# 8-distinct-tokens FILL (the original layout), 3 epochs = 1401 steps.
#
# Layout: each dead group's 8 free slots get 8 DIFFERENT candidates, 3 rounds over the
# 23-token pool = up to 24 attempts per group. Measured revival 21.8%. The t1x8 arm
# (one token per round, 64 attempts) reached only 24.0%, so the extra 40 attempts were
# not worth 2x the wall clock -- hence back to this layout.
#
# 3 epochs is the point of this run: at total_epochs=1 every prompt is seen exactly once
# (467 x 32 = 14944 ~ 14971 prompts), so a group FILL revived was never revisited and
# "can the policy solve it on its own later" was unanswerable from the log.
#
# Checkpointing is deliberately OFF (SAVE_FREQ inherits the script default of 0). A crash
# therefore cannot be resumed, and the trained policy will not exist afterwards, so no
# offline re-roll of the rescued prompts is possible on it. The log plus
# fill_rescued.jsonl are the only artifacts.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_3ep
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_3ep/fill_8tok_3ep.log
exec > >(tee "$LOG") 2>&1
export N_TOKENS_PER_GROUP=8
export FILL_ROUNDS=3
export TOTAL_EPOCHS=3
export RUN_TAG=fill8tok-3ep
exec bash run_local_grpo_fill.sh full

#!/bin/bash
# GRPO + FILL, 8-distinct-tokens layout, trained on MATH only, 3 epochs = 702 steps.
#
# Why MATH only: the measured deficit was always on MATH-500 and never on GSM8K, and MATH is
# where a flattened opening actually costs something (long structured derivations). 7498
# prompts / 32 = 234 steps per epoch, so 3 epochs is 702 steps instead of 1401 on the mix.
#
# Test set is still both (GSM8K test 1319 + MATH-500 500), so val-core/gsm8k/* stays
# comparable to the archived logs -- but it is now a pure TRANSFER measurement, not
# in-domain. And the archived baseline (95.91 / 86.05 peak) was trained on the mix, so it is
# NOT the comparison target for this run; a MATH-only GRPO baseline still has to be run.
#
# GPU_MEM_UTIL 0.4 -> 0.7: generation dominates the step (151 s vs ~20 s in update_actor) and
# free_cache_engine=True releases the KV pool before training, so this is pure throughput.
# PPO_MAX_TOKEN_LEN deliberately stays at the inherited 16384: FILL is a sibling term built
# from per-micro-batch masked_mean while GRPO's loss_scale_factor telescopes, so repartitioning
# silently rescales FILL's effective weight -- and fill_coef=4e-3 was calibrated at 16384.
#
# Checkpointing is deliberately OFF (SAVE_FREQ inherits the script default of 0). A crash
# cannot be resumed and the trained policy will not exist afterwards. The log plus
# fill_rescued.jsonl are the only artifacts.
set -uo pipefail
cd "$(dirname "$0")"

POOL=/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_math_only.json
PY=/home/xiongrengrong.xrr/miniconda3/envs/xrr_new/bin/python

mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_math3ep
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_math3ep/fill_math_8tok_3ep.log
exec > >(tee "$LOG") 2>&1

export DATA_PATH=datasets/math_only
export CANDIDATE_POOL_PATH="$POOL"
export N_TOKENS_PER_GROUP=8
# Read n_rounds off the pool itself so the round count can never disagree with the pool size.
export FILL_ROUNDS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['n_rounds'])" "$POOL")
export TOTAL_EPOCHS=3
export GPU_MEM_UTIL=0.7
export RUN_TAG=fillmath-3ep

echo "MATH-only train (7498) / both test (1819), 3 epochs = 702 steps"
echo "pool=$POOL  rounds=$FILL_ROUNDS  gpu_mem_util=$GPU_MEM_UTIL"
exec bash run_local_grpo_fill.sh full

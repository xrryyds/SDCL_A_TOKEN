#!/bin/bash
# Memory/throughput probe for the MATH-only 3-epoch arm: 10 steps, no validation, ~20 min.
#
# The gate is NOT the absolute memory peak -- 10 steps only reach response_length/mean ~343 and
# cannot reproduce the ~1200 that OOM'd the pre-Step-F run. The gate is that raising the sglang
# reservation from 0.4 to 0.7 does not raise the TRAINING-phase footprint, which is what
# free_cache_engine=True (rollout.yaml:47) with tensor_model_parallel_size=2 (:50) guarantees:
#
#   perf/max_memory_allocated_gb  ~= 64-66  (unchanged from the 0.4 runs)
#   perf/max_memory_reserved_gb   <  ~75    (was 67.4 at 0.4; 85+ means fragmentation -> use 0.55)
#   timing_s/gen                  must drop measurably, else the KV cache was not the bottleneck
#                                 and 0.7 buys nothing -- revert to 0.4 rather than carry the risk
set -uo pipefail
cd "$(dirname "$0")"

POOL=/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_math_only.json
PY=/home/xiongrengrong.xrr/miniconda3/envs/xrr_new/bin/python

mkdir -p /home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_math3ep
LOG=/home/xiongrengrong.xrr/SDCL_A_TOKEN/runs/fill_math3ep/probe_10steps_mem07.log
exec > >(tee "$LOG") 2>&1

export DATA_PATH=datasets/math_only
export CANDIDATE_POOL_PATH="$POOL"
export N_TOKENS_PER_GROUP=8
export FILL_ROUNDS=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['n_rounds'])" "$POOL")
export GPU_MEM_UTIL=0.7
export TOTAL_STEPS=10
export TEST_FREQ=0
export VAL_BEFORE_TRAIN=False
export RUN_TAG=fillmath-probe-mem07

echo "probe: MATH-only, 10 steps, no val, pool=$POOL rounds=$FILL_ROUNDS gpu_mem_util=$GPU_MEM_UTIL"
exec bash run_local_grpo_fill.sh smoke

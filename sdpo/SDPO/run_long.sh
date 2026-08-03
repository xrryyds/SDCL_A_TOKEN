#!/bin/bash
# SRPO Chemistry LONG run on 8 GPUs (paper hyperparams).
#
# Goal: give the soft-teacher fill enough steps to raise P(forced opener) from ~1e-8
# until the model rolls those openers out by itself in the main rollout, at which point
# the rescued samples are plain on-policy GRPO samples and earn reward normally.
#
# Note on max_response_length: observed response length on chemistry is mean ~230,
# max ~540 tokens, so 2048 leaves a large margin while keeping activation memory safe
# (the 1024 dry-run already peaked at ~71GB/81GB reserved per GPU). Raise it if you
# want the paper's 8192 and have memory headroom.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p output

MODEL_PATH=/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B \
N_GPUS_PER_NODE=8 \
bash run_local_srpo.sh chem_long \
  data.train_batch_size=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  data.max_response_length=2048 \
  trainer.total_epochs=30 \
  trainer.test_freq=5 \
  trainer.save_freq=-1 \
  2>&1 | tee output/srpo_chem_long.log

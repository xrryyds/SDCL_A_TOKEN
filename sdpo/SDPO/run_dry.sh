#!/bin/bash
# SRPO Chemistry dry-run (small batch / short response / 1 epoch) on 8 GPUs.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mkdir -p output

MODEL_PATH=/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B \
N_GPUS_PER_NODE=8 \
bash run_local_srpo.sh chem_dry \
  data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  data.max_response_length=1024 \
  actor_rollout_ref.actor.srpo.probe_num_samples=8 \
  trainer.total_epochs=1 \
  trainer.test_freq=2 \
  2>&1 | tee output/srpo_chem_dry.log

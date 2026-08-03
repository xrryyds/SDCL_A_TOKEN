#!/bin/bash

# Usage: ./run_local_srpo_rescue.sh [experiment_name_suffix]
#
# v6: three-way routing on SciKnowEval Chemistry.
#   correct                        -> GRPO branch
#   wrong + has correct sibling    -> SDPO branch (logit-level self-distillation)
#   whole group wrong (no teacher) -> all-fail rescue (forced first token)
#
# The third case is the blind spot of the paper's SRPO: without a correct sibling
# there is no teacher, and GRPO's group advantage is zero, so those rollouts are
# wasted. v5 measured that blind spot at 25-28% of prompts early in training.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="srpo_rescue"

DATA_PATH="datasets/sciknoweval/chemistry"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
# Paper SRPO: lr 5e-6, mini-batch 32
LR=5e-6
MODEL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"

# All-fail group rescue
CANDIDATE_POOL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_chemistry.json"
SUCCESS_REWARD_THRESHOLD=1.0
N_BASELINE_KEEP=2
N_TOKENS_PER_GROUP=3

# Countermeasures for the v5 degradation (entropy collapse + length blowup)
ENTROPY_COEFF=0.001
MAX_RESPONSE_LENGTH=4096

export N_GPUS_PER_NODE=8

SUFFIX=${1:-"local_srpo_rescue"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export HF_HUB_OFFLINE=1
export RAY_memory_monitor_refresh_ms=0

# The conda env is on a network FS, so Ray's runtime env agent can exceed the
# default 30s startup window, after which the raylet kills itself. Give it room.
export RAY_agent_register_timeout_ms=300000

export VARS_DIR="$PROJECT_ROOT"

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-SRPO_RESCUE-train${TRAIN_BATCH_SIZE}-keep${N_BASELINE_KEEP}-tok${N_TOKENS_PER_GROUP}-ent${ENTROPY_COEFF}-resp${MAX_RESPONSE_LENGTH}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SRPO_RESCUE-local \
vars.dir=$VARS_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.token_roll.enable=True \
actor_rollout_ref.actor.token_roll.candidate_pool_path=$CANDIDATE_POOL_PATH \
actor_rollout_ref.actor.token_roll.success_reward_threshold=$SUCCESS_REWARD_THRESHOLD \
actor_rollout_ref.actor.token_roll.n_baseline_keep=$N_BASELINE_KEEP \
actor_rollout_ref.actor.token_roll.n_tokens_per_group=$N_TOKENS_PER_GROUP \
data.max_response_length=$MAX_RESPONSE_LENGTH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.logger=[console] \
trainer.default_local_dir=$PROJECT_ROOT/outputs/srpo_rescue_chem \
trainer.total_epochs=30 \
trainer.total_training_steps=450 \
trainer.val_before_train=True \
trainer.test_freq=5 \
trainer.save_freq=0 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
+ray_kwargs.ray_init.object_store_memory=10000000000"

echo "----------------------------------------------------------------"
echo "Starting Local SRPO + All-Fail Rescue (v6, three-way routing)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Candidate Pool: $CANDIDATE_POOL_PATH"
echo "entropy_coeff: $ENTROPY_COEFF | max_response_length: $MAX_RESPONSE_LENGTH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

#!/bin/bash

# Usage: ./run_local_token_roll.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="token_roll"

DATA_PATH="datasets/sciknoweval_all"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=5e-6
MODEL_PATH="/workspace/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"

# Token-Roll specific hyperparameters
TOKEN_POOL_PATH="/workspace/SDCL_A_TOKEN/datasets/first_tokens_test.json"
SUCCESS_REWARD_THRESHOLD=1.0
CE_LOSS_WEIGHT=1.0
REVERSE_KL_WEIGHT=1.0
NUM_FORCED_ATTEMPTS=1

export N_GPUS_PER_NODE=2

SUFFIX=${1:-"local_token_roll"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export HF_HUB_OFFLINE=1

# Override vars.dir from user.yaml (which defaults to /users/$USER/SDPO)
export VARS_DIR="$PROJECT_ROOT"

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-TOKEN_ROLL-train${TRAIN_BATCH_SIZE}-ce${CE_LOSS_WEIGHT}-rkl${REVERSE_KL_WEIGHT}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=TOKEN_ROLL-local \
vars.dir=$VARS_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.token_roll.token_pool_path=$TOKEN_POOL_PATH \
actor_rollout_ref.actor.token_roll.success_reward_threshold=$SUCCESS_REWARD_THRESHOLD \
actor_rollout_ref.actor.token_roll.ce_loss_weight=$CE_LOSS_WEIGHT \
actor_rollout_ref.actor.token_roll.reverse_kl_weight=$REVERSE_KL_WEIGHT \
actor_rollout_ref.actor.token_roll.num_forced_attempts=$NUM_FORCED_ATTEMPTS \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "----------------------------------------------------------------"
echo "Starting Local Token-Roll Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Token Pool: $TOKEN_POOL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

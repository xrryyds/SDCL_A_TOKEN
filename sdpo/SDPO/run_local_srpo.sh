#!/bin/bash

# Usage: ./run_local_srpo.sh [experiment_name_suffix]
#
# SRPO (Sample-Routed Policy Optimization) + all-wrong-group re-rollout.
# Paper hyperparameters (Qwen3-8B): train_batch=32, mini_batch=32, rollout.n=8,
# lr=5e-6, warmup=10, weight_decay=0.01, grad clip 1.0, KL=0, temperature=1.0,
# clip_ratio_high=0.28, rollout IS clip 2.0, distillation top-k=100, JSD(alpha=0.5),
# EMA teacher rate=0.05, dynamic-weighting beta=1.0, val n=16 T=0.6 top_p=0.95.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="srpo"

# Run one benchmark at a time (paper trains each benchmark separately).
# Options: datasets/sciknoweval/{chemistry,physics,biology,material}, datasets/tooluse
DATA_PATH="datasets/sciknoweval/chemistry"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=5e-6
ALPHA=0.5
# Point this at your local Qwen3-8B (set HF_HUB_OFFLINE=1 for offline runs).
# Override with:  MODEL_PATH=/path/to/Qwen3-8B bash run_local_srpo.sh
MODEL_PATH="${MODEL_PATH:-/workspace/SDCL_A_TOKEN/model/Qwen/Qwen3-8B}"

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

SUFFIX=${1:-"local_srpo"}
# Extra hydra overrides after the suffix are appended (handy for dry-runs), e.g.:
#   bash run_local_srpo.sh chem_dry data.train_batch_size=16 data.max_response_length=1024
EXTRA_ARGS="${@:2}"

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
EXP_NAME="LOCAL-SRPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SRPO-local \
vars.dir=$VARS_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.srpo.enable_reroll=True \
actor_rollout_ref.actor.srpo.dw_beta=1.0 \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.logger=[console] \
trainer.val_before_train=True \
trainer.test_freq=5 \
trainer.save_freq=-1 \
trainer.default_local_dir=$PROJECT_ROOT/output/$EXP_NAME \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py"

echo "----------------------------------------------------------------"
echo "Starting Local SRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS $EXTRA_ARGS

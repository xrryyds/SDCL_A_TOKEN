#!/bin/bash

# Usage:
#   RESCUE=False ./run_local_srpo_v7.sh   # run (1): faithful SRPO reproduction
#   RESCUE=True  ./run_local_srpo_v7.sh   # run (2): SRPO + all-fail rescue
#   ./run_local_srpo_v7.sh [experiment_name_suffix]
#
# v7 on SciKnowEval Chemistry. Two runs differ by RESCUE alone, so (2)-(1) is the
# net contribution of the all-fail rescue.
#
# Routing:
#   correct                        -> GRPO branch
#   wrong + has correct sibling    -> SDPO branch (logit-level self-distillation)
#   whole group wrong (no teacher) -> all-fail rescue (forced first token), if RESCUE=True
#
# v7 fixes two deviations that made v6 an unfaithful SRPO and blocked attribution:
#   - branch losses now share the paper's single union denominator (weighted by token share)
#   - entropy-aware dynamic weighting enabled (dw_beta=1)
# and drops the v6 KL anchor / entropy bonus to match paper Table 3.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="srpo_v7"

DATA_PATH="datasets/sciknoweval/chemistry"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
# Paper SRPO: lr 5e-6, mini-batch 32
LR=5e-6
MODEL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"

# The single switch that separates run (1) from run (2).
RESCUE=${RESCUE:-True}

# All-fail group rescue
CANDIDATE_POOL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_chemistry.json"
SUCCESS_REWARD_THRESHOLD=1.0
N_BASELINE_KEEP=2
N_TOKENS_PER_GROUP=3

# Paper Table 3 / §3.2
ENTROPY_COEFF=0
MAX_RESPONSE_LENGTH=8192
DW_BETA=1.0

# Checkpoints are needed to re-measure the first-token distribution offline against
# the final weights. user.yaml keeps only the newest one.
SAVE_FREQ=100

export N_GPUS_PER_NODE=8

SUFFIX=${1:-"local_srpo_v7"}

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

EXP_NAME="LOCAL-SRPO_V7-rescue${RESCUE}-dw${DW_BETA}-train${TRAIN_BATCH_SIZE}-resp${MAX_RESPONSE_LENGTH}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SRPO_V7-local \
vars.dir=$VARS_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.self_distillation.dw_beta=$DW_BETA \
actor_rollout_ref.actor.token_roll.enable=$RESCUE \
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
trainer.default_local_dir=$PROJECT_ROOT/outputs/srpo_v7_chem_rescue${RESCUE} \
trainer.total_epochs=30 \
trainer.total_training_steps=450 \
trainer.val_before_train=True \
trainer.test_freq=5 \
trainer.save_freq=$SAVE_FREQ \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
+ray_kwargs.ray_init.object_store_memory=10000000000"

echo "----------------------------------------------------------------"
echo "Starting Local SRPO v7 (paper-faithful union norm + dynamic weighting)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "All-fail rescue (token_roll.enable): $RESCUE"
echo "Candidate Pool: $CANDIDATE_POOL_PATH"
echo "dw_beta: $DW_BETA | entropy_coeff: $ENTROPY_COEFF | max_response_length: $MAX_RESPONSE_LENGTH"
echo "Watch actor/entropy: the KL anchor and entropy bonus are gone, so abort if it"
echo "falls below 0.01 while val declines for 3 consecutive points."
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

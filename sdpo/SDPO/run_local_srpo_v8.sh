#!/bin/bash
# Usage:
#   RESCUE=True ./run_local_srpo_v8.sh   # v8: SRPO + all-fail rescue + EMA KL anchor
#   ./run_local_srpo_v8.sh [experiment_name_suffix]
#
# v8 on SciKnowEval Chemistry. Builds on v7 RESCUE=True, adds EMA first-token
# KL anchor to prevent the entropy explosion (1.0->10.36) that caused v7 to
# diverge after step 365.
#
# Routing:
#   correct                        -> GRPO branch
#   wrong + has correct sibling    -> SDPO branch (logit-level self-distillation)
#   whole group wrong (no teacher) -> all-fail rescue (forced first token)
#
# v8 changes from v7:
#   - Two-sided entropy band on the aggregated entropy: squared hinge outside
#     [entropy_band_lo, entropy_band_hi], zero gradient inside.
#   - EMA first-token anchor disabled (ft_ema_kl_coef=0). Both the KL and the CE
#     variants failed: the EMA target is a batch marginal while the policy is a
#     per-prompt conditional, so pulling toward it inflates entropy (Jensen) and
#     the inflated batch feeds back into the EMA. Stats still logged for monitoring.

# =============================================================================
# CONFIGURATION
# =============================================================================
# =============================================================================

CONFIG_NAME="srpo_v8"

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

# V8: EMA first-token stats kept for monitoring only (coef=0 => no gradient)
FT_EMA_ALPHA=0.99
FT_EMA_KL_COEF=0.0

# V8: two-sided entropy band (the actual entropy control)
ENTROPY_BAND_COEF=0.02
ENTROPY_BAND_LO=0.4
ENTROPY_BAND_HI=1.5

# V8: mirrored KL-Cov brake on the most-negative-covariance tokens.
# Off by default. Our per-micro-batch GRPO token count is only ~3900, so the
# paper default 0.0002 would brake a single token; use ~0.005 (~19 tokens).
NEG_COV_RATIO=${NEG_COV_RATIO:-0.0}
NEG_COV_KL_COEF=${NEG_COV_KL_COEF:-0.1}

# Checkpoints are needed to re-measure the first-token distribution offline against
# the final weights. user.yaml keeps only the newest one.
SAVE_FREQ=0

export N_GPUS_PER_NODE=8

SUFFIX=${1:-"local_srpo_v8"}

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

EXP_NAME="LOCAL-SRPO_V8C-rescue${RESCUE}-band${ENTROPY_BAND_LO}_${ENTROPY_BAND_HI}x${ENTROPY_BAND_COEF}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SRPO_V8-local \
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
actor_rollout_ref.actor.token_roll.rescue_loss_weight=0.0 \
actor_rollout_ref.actor.token_roll.fill_ce_beta=0.01 \
actor_rollout_ref.actor.token_roll.fill_ce_clip=0.28 \
actor_rollout_ref.actor.token_roll.ft_ema_alpha=$FT_EMA_ALPHA \
actor_rollout_ref.actor.token_roll.ft_ema_kl_coef=$FT_EMA_KL_COEF \
actor_rollout_ref.actor.entropy_band_coef=$ENTROPY_BAND_COEF \
actor_rollout_ref.actor.entropy_band_lo=$ENTROPY_BAND_LO \
actor_rollout_ref.actor.entropy_band_hi=$ENTROPY_BAND_HI \
actor_rollout_ref.actor.policy_loss.neg_cov_ratio=$NEG_COV_RATIO \
actor_rollout_ref.actor.policy_loss.neg_cov_kl_coef=$NEG_COV_KL_COEF \
data.max_response_length=$MAX_RESPONSE_LENGTH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.logger=[console] \
trainer.default_local_dir=$PROJECT_ROOT/outputs/srpo_v8c_chem_rescue${RESCUE} \
trainer.total_epochs=30 \
trainer.total_training_steps=450 \
trainer.val_before_train=True \
trainer.test_freq=5 \
trainer.save_freq=$SAVE_FREQ \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
+actor_rollout_ref.actor.use_rollout_log_probs=True \
+ray_kwargs.ray_init.object_store_memory=10000000000"

echo "----------------------------------------------------------------"
echo "Starting Local SRPO v8c (two-sided entropy band)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "All-fail rescue (token_roll.enable): $RESCUE"
echo "Candidate Pool: $CANDIDATE_POOL_PATH"
echo "dw_beta: $DW_BETA | entropy_coeff: $ENTROPY_COEFF | max_response_length: $MAX_RESPONSE_LENGTH"
echo "Entropy band: [$ENTROPY_BAND_LO, $ENTROPY_BAND_HI] coef=$ENTROPY_BAND_COEF (squared hinge, 0 grad inside)"
echo "EMA first-token anchor DISABLED (ft_ema_kl_coef=$FT_EMA_KL_COEF); stats still logged for monitoring"
echo "Watch actor/entropy (should stay <= ~1.5) and actor/entropy_band_loss (should stay near 0)."
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

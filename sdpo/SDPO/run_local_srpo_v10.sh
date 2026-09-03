#!/bin/bash
# Usage:
#   ./run_local_srpo_v10.sh [experiment_name_suffix]
#   FILL_FT_WEIGHT=5 ./run_local_srpo_v10.sh      # tune the forced-token gate push
#
# v10 on SciKnowEval Chemistry: paper SRPO + a third FILL branch.
#
# Routing:
#   correct                        -> GRPO branch
#   wrong + has correct sibling    -> SDPO branch (logit-level self-distillation)
#   whole group wrong              -> FILL branch (this is what the paper skips)
#
# The paper's SRPO leaves all-fail groups at zero advantage, so those prompts never
# contribute anything. FILL replaces all 8 slots with one forced first token each,
# keeps the rollouts that then solved the prompt, and imitates their full trajectory.
# Failed forced rollouts are dropped from every branch and every denominator.
#
# Baseline to beat: paper SRPO reproduction = 83.0 avg@16 on Chemistry.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="srpo_v10"

DATA_PATH="${DATA_PATH:-datasets/sciknoweval/chemistry}"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=${LR:-5e-6}
MODEL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"

# 8-token pool: the 6 low-probability candidates plus To/The, which the 6-token
# pool had excluded as baseline-dominant. One token per rollout slot.
CANDIDATE_POOL_PATH="${CANDIDATE_POOL_PATH:-/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_chemistry_8.json}"
SUCCESS_REWARD_THRESHOLD=1.0
N_BASELINE_KEEP=0
N_TOKENS_PER_GROUP=8

# FILL_ENABLE=False gives a clean paper-SRPO baseline from this same code: rescue
# returns early, the fill masks are never added, so lambda_fill is 0 and the objective
# collapses back to the paper's two-branch union. Use it to check that the v10 changes
# did not regress the 83.0 reproduction.
FILL_ENABLE=${FILL_ENABLE:-True}

# Paper Table 3
ENTROPY_COEFF=0
MAX_RESPONSE_LENGTH=8192
DW_BETA=1.0

# FILL branch
FILL_CE_CLIP=0.28
# Forced-token weight inside the FILL branch. 1.0 = the forced token is weighted like
# any other token in the trajectory. At 9.0 the branch carried under 5% of the loss yet
# dominated the globally shared first-token distribution, which then churned every ~100
# steps (Determin 22.3% -> 0.7%, Analy 3.2% -> 27.4%), repeatedly invalidating the
# reasoning paths built on top of each opening and capping val at 74-78%.
FILL_FT_WEIGHT=${FILL_FT_WEIGHT:-1.0}
FILL_COEF=${FILL_COEF:-1.0}

SAVE_FREQ=0

export N_GPUS_PER_NODE=8

SUFFIX=${1:-"local_srpo_v10"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export HF_HUB_OFFLINE=1
export RAY_memory_monitor_refresh_ms=0
[ -n "$PYTORCH_CUDA_ALLOC_CONF" ] && export PYTORCH_CUDA_ALLOC_CONF

# The conda env is on a network FS, so Ray's runtime env agent can exceed the
# default 30s startup window, after which the raylet kills itself. Give it room.
export RAY_agent_register_timeout_ms=300000

export VARS_DIR="$PROJECT_ROOT"

if [ -n "$RUN_TAG" ]; then
  RUN_TAG="$RUN_TAG"
elif [ "$FILL_ENABLE" = "True" ]; then
  RUN_TAG="fill8-wft${FILL_FT_WEIGHT}"
else
  RUN_TAG="baseline-nofill"
fi

# Two concurrent runs must not share these. /tmp/ray/session_latest points at
# whichever started last, and concurrent compilation into a single inductor cache
# corrupts it (source of the earlier "pickle data was truncated" errors).
export RAY_TMPDIR="/tmp/ray_${RUN_TAG}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${RUN_TAG}"
mkdir -p "$RAY_TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

OUT_DIR="$PROJECT_ROOT/outputs/srpo_v10_${RUN_TAG}"
mkdir -p "$OUT_DIR"
# Stable pointer so log analysis never has to guess which Ray session is which run.
echo "$RAY_TMPDIR" > "$OUT_DIR/RAY_TMPDIR"

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-SRPO_V10-${RUN_TAG}-lr${LR}-${SUFFIX}"

# The "<reasoning>\n" scaffold is SciKnowEval-specific. Datasets that answer directly
# (e.g. MATH) need the forced first token at response position 0.
PREFIX_ARG=""
if [ "${FILL_PREFIX_NONE:-0}" != "0" ]; then
  PREFIX_ARG="actor_rollout_ref.actor.token_roll.response_prefix=null"
fi

# DAPO component 4 (Overlong Reward Shaping). Only the `dapo` reward manager takes
# these kwargs; `naive` would raise TypeError on them.
OVERLONG_ARGS=""
if [ "${REWARD_MANAGER:-naive}" = "dapo" ]; then
  OVERLONG_ARGS="+reward_model.reward_kwargs.max_resp_len=$MAX_RESPONSE_LENGTH \
+reward_model.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER:-False} \
+reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN:-4096} \
+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY:-1.0} \
+reward_model.reward_kwargs.overlong_buffer_cfg.log=True"
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.seed=${SEED:-null} \
trainer.group_name=SRPO_V10-local \
vars.dir=$VARS_DIR \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH:-32} \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean} \
actor_rollout_ref.actor.entropy_band_coef=0.0 \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.policy_loss.neg_cov_ratio=0.0 \
actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH:-0.2} \
actor_rollout_ref.actor.self_distillation.dw_beta=$DW_BETA \
actor_rollout_ref.actor.self_distillation.mask_answer_from_demonstration=${MASK_ANSWER:-False} \
actor_rollout_ref.actor.self_distillation.sdpo_skip_when_n_correct_ge=${SDPO_SKIP_K:-0} \
actor_rollout_ref.actor.self_distillation.cir_select_enable=${CIR_SELECT:-False} \
actor_rollout_ref.actor.token_roll.enable=$FILL_ENABLE \
actor_rollout_ref.actor.token_roll.candidate_pool_path=$CANDIDATE_POOL_PATH \
actor_rollout_ref.actor.token_roll.success_reward_threshold=$SUCCESS_REWARD_THRESHOLD \
actor_rollout_ref.actor.token_roll.n_baseline_keep=$N_BASELINE_KEEP \
actor_rollout_ref.actor.token_roll.n_tokens_per_group=$N_TOKENS_PER_GROUP \
actor_rollout_ref.actor.token_roll.rescue_loss_weight=0.0 \
actor_rollout_ref.actor.token_roll.fill_ce_beta=0.0 \
actor_rollout_ref.actor.token_roll.fill_ce_clip=$FILL_CE_CLIP \
actor_rollout_ref.actor.token_roll.fill_first_token_weight=$FILL_FT_WEIGHT \
actor_rollout_ref.actor.token_roll.fill_coef=$FILL_COEF \
actor_rollout_ref.actor.token_roll.ft_ema_kl_coef=0.0 \
data.max_response_length=$MAX_RESPONSE_LENGTH \
+data.gen_batch_size=${GEN_BATCH_SIZE:-$TRAIN_BATCH_SIZE} \
algorithm.filter_groups.enable=${DAPO_FILTER:-False} \
algorithm.filter_groups.metric=${DAPO_METRIC:-acc} \
algorithm.filter_groups.max_num_gen_batches=${DAPO_MAX_GEN:-10} \
reward_model.reward_manager=${REWARD_MANAGER:-naive} \
$OVERLONG_ARGS \
actor_rollout_ref.actor.clip_ratio_c=${CLIP_C:-3.0} \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.logger=[console] \
trainer.default_local_dir=$PROJECT_ROOT/outputs/srpo_v10_${RUN_TAG} \
trainer.total_epochs=${TOTAL_EPOCHS:-30} \
trainer.total_training_steps=${TOTAL_STEPS:-null} \
trainer.val_before_train=True \
trainer.test_freq=${TEST_FREQ:-5} \
trainer.first_token_probe_dump_freq=5 \
trainer.validation_data_dir=${VAL_DUMP_DIR:-null} \
trainer.save_freq=$SAVE_FREQ \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.4} \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-16384} \
+ray_kwargs.ray_init.object_store_memory=10000000000 \
actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE:-vllm} $PREFIX_ARG"

echo "----------------------------------------------------------------"
echo "Starting Local SRPO v10 (paper SRPO + FILL branch)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Candidate pool (8 tokens): $CANDIDATE_POOL_PATH"
echo "FILL: n_baseline_keep=$N_BASELINE_KEEP n_tokens=$N_TOKENS_PER_GROUP clip=$FILL_CE_CLIP w_ft=$FILL_FT_WEIGHT"
echo "FILL selection: slots filled in fixed pool order (To, The, then the 6 novel);"
echo "                only the lowest-index correct rollout per group is learned from."
echo "SRPO branches unchanged from the paper; entropy band / EMA anchor / neg-cov all OFF."
echo "Baseline to beat: 83.0 avg@16 (paper SRPO reproduction)."
echo "Watch: first_token/novel_frac (should rise steadily, not churn), rescue/winner_slot_rank_mean,"
echo "       srpo/lambda_fill, rescue/n_fill_winners, actor/entropy. NOTE pool_frac now saturates at 1.0."
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

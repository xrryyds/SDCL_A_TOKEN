#!/bin/bash
# GRPO + FILL on GSM8K + MATH, at the exact grid point of the finished GRPO baseline.
#
# Ablation against SDPO_official/run_local_grpo_paper.sh (467 steps, one epoch):
#   GSM8K 86.94 -> 95.43 final / 95.91 peak
#   MATH-500 73.10 -> 84.95 final / 86.05 peak
# The only intended difference here is the FILL branch, so every shared knob below is
# pinned to the value the baseline actually resolved to, not to this fork's defaults.
#
# Routing with SRPO_DISABLE_SDPO=1 (ray_trainer.py:977):
#   any group with >=1 correct   -> GRPO branch (wrong siblings keep their negative advantage)
#   whole group wrong            -> FILL branch (8 forced first tokens, imitate what solves it)
# The SDPO branch is off, so FILL is the single variable versus the baseline.
#
# Usage:
#   ./run_local_grpo_fill.sh                       # full run, 467 steps
#   TOTAL_STEPS=3 TEST_FREQ=1 ./run_local_grpo_fill.sh smoke

CONFIG_NAME="srpo_v10"
DATA_PATH="${DATA_PATH:-datasets/gsm8k_math}"
MODEL_PATH="/home/xiongrengrong.xrr/SDCL_A_TOKEN/model/Qwen/Qwen3-8B"

# --- baseline parity (all values read back out of the baseline's resolved config) ---
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
MINI_BATCH=${MINI_BATCH:-8}          # 32/8 = 4 off-policy updates per rollout batch
LR=${LR:-1e-6}
CLIP_HIGH=${CLIP_HIGH:-0.28}
CLIP_C=3.0
LOSS_AGG_MODE=token-mean
ENTROPY_COEFF=0
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=10240
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-16384}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
VAL_N=${VAL_N:-4}
TEST_FREQ=${TEST_FREQ:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Disk has ~87 GB free of 200 GB; a full model+optimizer+extra FSDP checkpoint for
# Qwen3-8B is ~90-130 GB and does not fit. Model-only (~16 GB bf16) with one retained
# checkpoint does, which is a salvage path (Adam state is lost on resume), not a clean
# resume -- hence still 0 by default.
SAVE_FREQ=${SAVE_FREQ:-0}

# --- FILL branch ---
# Pool measured on the actual gsm8k_math train mix (14971 prompts) with scaffold='' and
# enable_thinking=False, sorted strictly by descending baseline probability so the slot
# index means "opening naturalness".
# 23 tokens, not 32: the report's prob_mean is conditional on the token appearing in the
# top-k, so a proper noun from one word problem (Tom/Maria/Jennifer) scores like a generic
# opening. Only 31 of the 200 measured tokens appear in the top-k for >=90% of prompts,
# and after dropping markdown/CJK fragments and case variants 23 generic openings remain.
# Multi-round: round r forces candidates[r*N_TOKENS_PER_GROUP : ...], and only groups
# still all-fail enter the next round, so a dead group gets up to
# N_TOKENS_PER_GROUP*FILL_ROUNDS attempts.
# Single-round scored 0/40 forced rollouts correct, which only bounds the per-attempt
# success rate at p<0.072 -- the extra rounds exist to measure p, not to assume it.
# N_TOKENS_PER_GROUP=1 switches to one-token-per-round: the window holds a single
# candidate, so ray_trainer.py's window[j % k] hands every free slot the same token and
# rescue/round{r}/n_winners becomes the per-token revival rate for candidate r. The 8x8
# layout instead spends one attempt per token, which cannot separate the two.
CANDIDATE_POOL_PATH="${CANDIDATE_POOL_PATH:-/home/xiongrengrong.xrr/SDCL_A_TOKEN/datasets/first_token_candidates_gsm8k_math_23.json}"
SUCCESS_REWARD_THRESHOLD=1.0
N_BASELINE_KEEP=0                    # replace all 8 slots
N_TOKENS_PER_GROUP=${N_TOKENS_PER_GROUP:-8}
FILL_ROUNDS=${FILL_ROUNDS:-3}        # ceil(23/8)
FILL_FT_WEIGHT=${FILL_FT_WEIGHT:-1.0}
# Measured on the 10-step probe: grad_norm was 36.6-68.9 on the 5 steps where FILL fired
# vs 0.10-0.17 on the 5 where it did not. grad_clip=1.0 (config/actor/dp_actor.yaml:29)
# therefore rescaled those updates by 1/37-1/69, shrinking the step's GRPO gradient by the
# same factor -- every rescue cancelled that step's baseline learning. 4e-3 puts FILL's
# contribution (~50) on GRPO's scale (~0.2).
FILL_COEF=${FILL_COEF:-4e-3}
FILL_CE_CLIP=0.28

# GSM8K and MATH answer directly; the "<reasoning>\n" scaffold is SciKnowEval-only, so the
# forced token must land at response position 0.
RESPONSE_PREFIX_ARG="actor_rollout_ref.actor.token_roll.response_prefix=null"

# SDPO off. dw_beta=0 with it, so the teacher top-k distribution is never needed and
# core_algos.compute_self_distillation_loss skips the dynamic-weighting block entirely.
export SRPO_DISABLE_SDPO=1
DW_BETA=0

SUFFIX=${1:-"grpo_fill"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export HF_HUB_OFFLINE=1
export RAY_memory_monitor_refresh_ms=0
# The conda env sits on a network FS and Ray's runtime env agent can blow past the
# default 30s window, after which the raylet kills itself.
export RAY_agent_register_timeout_ms=300000

CONDA_ENV_BIN=/home/xiongrengrong.xrr/miniconda3/envs/xrr_new/bin
export PATH="$CONDA_ENV_BIN:$PATH"

export VARS_DIR="$PROJECT_ROOT"
RUN_TAG="${RUN_TAG:-grpofill-${SUFFIX}}"

# Never shared between concurrent runs: /tmp/ray/session_latest points at whichever
# started last, and two processes compiling into one inductor cache corrupts it.
export RAY_TMPDIR="/tmp/ray_${RUN_TAG}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${RUN_TAG}"
mkdir -p "$RAY_TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

OUT_DIR="$PROJECT_ROOT/outputs/grpo_fill_${RUN_TAG}"
mkdir -p "$OUT_DIR"
echo "$RAY_TMPDIR" > "$OUT_DIR/RAY_TMPDIR"

EXP_NAME="LOCAL-GRPO_FILL-${RUN_TAG}-lr${LR}"

# =============================================================================
# EXECUTION
# =============================================================================

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.seed=${SEED:-null} \
+data.gen_batch_size=$TRAIN_BATCH_SIZE \
max_model_len=$MAX_MODEL_LEN \
vars.dir=$VARS_DIR \
trainer.group_name=GRPO_FILL-local \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE:-sglang} \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.actor.entropy_band_coef=0.0 \
actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
actor_rollout_ref.actor.clip_ratio_c=$CLIP_C \
actor_rollout_ref.actor.use_kl_loss=False \
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.policy_loss.neg_cov_ratio=0.0 \
actor_rollout_ref.actor.self_distillation.dw_beta=$DW_BETA \
actor_rollout_ref.actor.self_distillation.cir_select_enable=False \
actor_rollout_ref.actor.token_roll.enable=True \
actor_rollout_ref.actor.token_roll.candidate_pool_path=$CANDIDATE_POOL_PATH \
actor_rollout_ref.actor.token_roll.success_reward_threshold=$SUCCESS_REWARD_THRESHOLD \
actor_rollout_ref.actor.token_roll.n_baseline_keep=$N_BASELINE_KEEP \
actor_rollout_ref.actor.token_roll.n_tokens_per_group=$N_TOKENS_PER_GROUP \
actor_rollout_ref.actor.token_roll.n_rounds=$FILL_ROUNDS \
actor_rollout_ref.actor.token_roll.rescue_loss_weight=0.0 \
actor_rollout_ref.actor.token_roll.fill_ce_beta=0.0 \
actor_rollout_ref.actor.token_roll.fill_ce_clip=$FILL_CE_CLIP \
actor_rollout_ref.actor.token_roll.fill_first_token_weight=$FILL_FT_WEIGHT \
actor_rollout_ref.actor.token_roll.fill_coef=$FILL_COEF \
actor_rollout_ref.actor.token_roll.ft_ema_kl_coef=0.0 \
algorithm.adv_estimator=grpo \
algorithm.norm_adv_by_std_in_grpo=False \
algorithm.filter_groups.enable=False \
algorithm.rollout_correction.rollout_is=token \
algorithm.rollout_correction.rollout_is_threshold=2.0 \
reward_model.reward_manager=naive \
custom_reward_function.path=$PROJECT_ROOT/verl/utils/reward_score/feedback/__init__.py \
trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
trainer.logger=[console] \
trainer.default_local_dir=$OUT_DIR \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.total_training_steps=${TOTAL_STEPS:-null} \
trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
trainer.test_freq=$TEST_FREQ \
trainer.save_freq=$SAVE_FREQ \
actor_rollout_ref.actor.checkpoint.save_contents=[model] \
trainer.max_actor_ckpt_to_keep=1 \
trainer.first_token_probe_dump_freq=0 \
+ray_kwargs.ray_init.object_store_memory=10000000000 \
$RESPONSE_PREFIX_ARG"

echo "----------------------------------------------------------------"
echo "GRPO + FILL on $DATA_PATH  (SDPO branch OFF via SRPO_DISABLE_SDPO=1)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH   ($("$CONDA_ENV_BIN/python" -c "
import pandas as pd,sys
for split in ('train','test'):
    d=pd.read_parquet(f'$PROJECT_ROOT/$DATA_PATH/{split}.parquet')
    print(f'{split} {len(d)} =', dict(d.data_source.value_counts()), end='  ')
" 2>/dev/null || echo "unreadable"))"
echo "Grid: bs=$TRAIN_BATCH_SIZE n=$ROLLOUT_N mb=$MINI_BATCH lr=$LR clip_high=$CLIP_HIGH val_n=$VAL_N"
echo "Pool: $CANDIDATE_POOL_PATH  ($N_TOKENS_PER_GROUP slots x $FILL_ROUNDS rounds = up to $((N_TOKENS_PER_GROUP * FILL_ROUNDS)) attempts per dead group, no response prefix)"
if [ "$DATA_PATH" = "datasets/gsm8k_math" ]; then
  echo "Baseline to beat: GSM8K 95.43 final / 95.91 peak, MATH-500 84.95 final / 86.05 peak"
else
  echo "Baseline: NONE for $DATA_PATH. The archived 95.91 / 86.05 peaks were trained on the"
  echo "          gsm8k_math mix and are NOT the comparison target -- a GRPO run on this same"
  echo "          training set has to be done before these accuracies mean anything."
fi
echo "Watch: rescue/n_forced_rollouts, rescue/n_rescued_rollouts, rescue/n_rounds_run,"
echo "       rescue/winner_round_mean, rescue/winner_slot_rank_mean, srpo/fill_loss,"
echo "       val-core/gsm8k/*, val-core/math/*, actor/entropy, response_length/mean"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS

# SRPO v7 Experiment Log — Faithful Paper Reproduction

## Overview

| Item | Value |
|------|-------|
| Experiment | SRPO v7 RESCUE=False (faithful paper reproduction) |
| Date | 2026-08-05 |
| Model | Qwen3-8B |
| GPUs | 8×H20 (97GB each) |
| Dataset | SciKnowEval Chemistry |
| Paper target | ~83.0% val acc@16 |
| **Peak result** | **83.2% (step 420)** |
| Final result | 80.8% (step 450) |
| Status | RESCUE=False complete; RESCUE=True pending |

## Config (paper-faithful)

| Parameter | Value | Source |
|-----------|-------|--------|
| loss_mode | srpo | v7 three-way routing |
| lr | 5e-6 | Paper Table 3 |
| train_batch_size | 32 | Paper |
| ppo_mini_batch_size | 32 | on-policy (no PPO clip) |
| rollout.n | 8 | Paper |
| max_response_length | 8192 | Paper |
| max_prompt_length | 2048 | Paper |
| entropy_coeff | 0 | Paper (no entropy bonus) |
| use_kl_loss | False | Paper (no KL anchor) |
| dw_beta | 1.0 | Paper §3.2 |
| is_clip | 2.0 | Paper |
| teacher_update_rate | 0.05 | actor.yaml default |
| total_training_steps | 450 | Paper |
| test_freq | 5 | — |
| token_roll.enable | False | RESCUE=False switch |

## Bugs Fixed Before This Run

### Bug 1: EMA teacher never created for SRPO mode
Four code locations gated on `loss_mode == "sdpo"` only, excluding `"srpo"`:
- `main_ppo.py:131` — `self_distillation_needs_ref` → role=ActorRolloutRef (reference model not loaded)
- `fsdp_workers.py:900` — teacher module not assigned from ref_module_fsdp
- `dp_actor.py:135` — `_update_teacher()` EMA update skipped
- **Fix**: Changed all gates to `loss_mode in ("sdpo", "srpo")`

### Bug 2: IS correction disabled (on_policy shortcut)
When `ppo_mini_batch_size == train_batch_size` and `ppo_epochs == 1`, `on_policy=True`
short-circuits `old_log_prob = log_prob.detach()`, making IS ratio always 1.0.
- **Fix**: Added `use_rollout_log_probs=True` to force `old_log_prob` from pre-computed
  rollout log probs, enabling proper IS correction.

### Bug 3: DW normalization per-micro-batch + NCCL deadlock
- DW normalization was per-micro-batch (incorrect, not global)
- `all_reduce` was inside `if loss_mask.sum() > 0:` — ranks with no SDPO tokens
  skipped it, causing NCCL deadlock at step 1
- **Fix** (`core_algos.py:1186`): Moved `all_reduce` outside `loss_mask.sum() > 0` guard.
  All DP ranks now participate in the collective. `per_token_loss * dw_weight` is still
  guarded by `loss_mask.sum() > 0` to prevent NaN on empty ranks.

## Validation Accuracy Trajectory (every 5 steps)

```
step   0: 41.9%   (initial)
step  30: 66.5%
step  50: 71.9%
step 100: 77.5%
step 150: 76.2%
step 200: 78.3%
step 250: 79.9%
step 255: 82.1%   (first break 82%)
step 300: 78.1%   (dip)
step 350: 74.5%   (dip)
step 385: 80.2%
step 415: 82.3%
step 420: 83.2%   *** PEAK — matches paper ***
step 435: 82.1%
step 450: 80.8%   (final)
```

## Key Health Metrics

### Entropy (actor/entropy)
- Stable throughout: 0.25–0.55
- No collapse (previous runs collapsed to 0.006–0.067)
- Final step 450: 0.324

### Teacher Entropy (srpo/teacher_entropy_mean)
- Non-zero throughout: 0.04–0.27
- Confirms EMA teacher is active and providing distillation signal
- Final step 450: 0.255

### DW Weight Std (srpo/dw_weight_std)
- Non-zero throughout: 0.10–0.32
- Confirms dynamic weighting differentiates confident vs uncertain teacher positions

### SRPO Routing (srpo/lambda_grpo / lambda_sdpo)
- Early: ~75% GRPO + ~25% SDPO (many wrong-with-correct-sibling samples)
- Late: ~95% GRPO + ~5% SDPO (model improved, fewer SDPO samples)
- Self-decay working as designed

### Response Length
- Stable: 270–350 tokens throughout
- No length collapse or explosion
- Early decrease (454→245) is normal learning behavior

### IS Correction (rollout_corr)
- rollout_is_mean ≈ 1.0 throughout
- rollout_corr/kl ≈ 0.0003–0.0005
- IS ratio stable, no divergence

## Previous Failed Runs (for reference)

| Run | Peak val acc | Step of collapse | Root cause |
|-----|-------------|------------------|------------|
| v7 run 1 (bugs) | 61.7% (step 20) | step 30 (22.3%) | EMA teacher not created + IS disabled + DW per-batch |
| v7 run 2 (prior session) | 76.8% (step 170) | entropy→0.006 | Same bugs (partial fix attempt) |
| v7 run 3 (prior session) | 78.4% (step 50) | entropy→0.067 | Same bugs |
| v7 run 4 (after fix, before NCCL fix) | 41.6% (step 0) | step 1 (NCCL timeout) | all_reduce deadlock in DW norm |
| **v7 run 5 (all fixes)** | **83.2% (step 420)** | **no collapse** | **all bugs fixed** |

## Files Modified

| File | Change |
|------|--------|
| `verl/trainer/main_ppo.py:131` | `loss_mode in ("sdpo", "srpo")` for self_distillation_needs_ref |
| `verl/workers/fsdp_workers.py:900` | `loss_mode in ("sdpo", "srpo")` for teacher module creation |
| `verl/workers/actor/dp_actor.py:135` | `loss_mode not in ("sdpo", "srpo")` for EMA update guard |
| `verl/trainer/ppo/core_algos.py:1186` | DW all_reduce outside loss_mask guard; NaN-safe per_token_loss |
| `run_local_srpo_v7.sh` | `use_rollout_log_probs=True`, `ENTROPY_COEFF=0`, `SAVE_FREQ=0` |

## RESCUE=True Runs & First-Token Analysis

### Problem: Model never learns pool first tokens

RESCUE=True baseline (no amplification): `pool_frac=0.0` throughout — the 6-token
candidate pool (We, Calcul, Determin, Analy, This, 1) never appears in the model's
natural first-token distribution. Only "To" and "The" are used.

**Root cause analysis (two-layer signal dilution):**

1. **Too few samples**: Only ~5% of rollouts are forced-token (dead groups only,
   1-3 per step out of 32 groups × 8 rollouts). Of those, ~10% are correct →
   effective positive signal ~0.5%.

2. **First-token gradient diluted**: GRPO loss is a token-level masked mean over
   ~300 response tokens. The forced first token is 1/~300 → gradient signal is
   ~0.3% of total response gradient. Even with 50x advantage amplification, the
   effective signal is ~0.0008 — noise level.

### Fix 1: Rescue advantage amplification (`rescue_loss_weight`)

Config: `token_roll.rescue_loss_weight=1.0`

Scales advantages of rescued samples so their total contribution matches non-rescue
GRPO samples. Formula:
```
n_rescue = rescued tokens count
n_non_rescue = non-rescue GRPO tokens count
equal_scale = n_non_rescue / n_rescue  (e.g., ~50x when rescue is 2% of tokens)
scale_factor = 1 + rescue_loss_weight * (equal_scale - 1)
advantages[rescued] *= scale_factor
```

When `rescue_loss_weight=0.0`: no change (original behavior).
When `rescue_loss_weight=1.0`: equal total contribution.

### Fix 2: Fill CE loss (`fill_ce_beta`, `fill_ce_clip`)

Config: `token_roll.fill_ce_beta=1.0`, `token_roll.fill_ce_clip=0.28`

A direct cross-entropy loss at the forced-token position, with PPO-style ratio
clip to prevent over-optimization. Unlike GRPO (which dilutes the first-token
signal across all response tokens), this loss ONLY targets the forced token.

**Mechanism:**
- `fill_first_token_mask`: Built during rescue — marks the forced-token position
  in each rescued sample's response (at `len(tokenizer.encode(response_prefix))`).
- CE loss: `ratio = exp(log_prob_new - old_log_prob)` at forced positions,
  `clipped_ratio = clamp(ratio, 1-fill_ce_clip, 1+fill_ce_clip)`,
  `fill_ce_loss = mean(-clipped_ratio * fill_first_token_mask)`
- Total loss: `pg_loss + fill_ce_beta * fill_ce_loss`
- Equivalent to PPO with advantage=1 at the forced position — always pushes the
  token probability up, regardless of whether the forced rollout was correct.
- Ratio clip stops optimization once the model has sufficiently increased the
  token's probability (ratio > 1+clip → gradient zeroed).

**Why CE (not advantage-boost):**
- Advantage-boost (atoken-style) still depends on reward signal — if the forced
  rollout is wrong (negative advantage), the model is pushed AWAY from that token.
- CE loss directly increases the token probability regardless of reward, which is
  the goal: enrich the first-token distribution with pool tokens.

### Three approaches considered

| Approach | Signal source | Depends on reward? | Complexity |
|----------|--------------|-------------------|------------|
| A: Advantage boost (atoken-style) | Scaled advantage at FT pos | Yes (wrong→negative) | Low |
| B: Independent PG loss at FT pos | PPO clip loss, FT-only norm | Yes | Medium |
| **C: CE with clip (chosen)** | `-clamp(ratio, 1±eps)` at FT pos | **No** | Low |

## Current Config (RESCUE=True + Fill CE)

| Parameter | Value | Notes |
|-----------|-------|-------|
| token_roll.enable | True | Rescue active for all-fail groups |
| rescue_loss_weight | 1.0 | Equal contribution rescue vs non-rescue |
| fill_ce_beta | 1.0 | CE loss weight at forced-token position |
| fill_ce_clip | 0.28 | PPO-style ratio clip for CE loss |
| n_baseline_keep | 2 | Original rollouts kept per dead group |
| n_tokens_per_group | 3 | Distinct forced tokens per dead group |
| candidate_pool | 6 tokens | first_token_candidates_chemistry.json |
| response_prefix | `<reasoning>\n` | Forced token placed after this scaffold |
| (all other params) | same as RESCUE=False | Paper-faithful |

## Files Modified (cumulative)

| File | Change |
|------|--------|
| `verl/trainer/main_ppo.py:131` | `loss_mode in ("sdpo", "srpo")` for self_distillation_needs_ref |
| `verl/workers/fsdp_workers.py:900` | `loss_mode in ("sdpo", "srpo")` for teacher module creation |
| `verl/workers/actor/dp_actor.py:135` | `loss_mode not in ("sdpo", "srpo")` for EMA update guard |
| `verl/workers/actor/dp_actor.py:684-940` | SRPO loss: GRPO+SDPO union norm, rescue advantage amplification, fill CE loss |
| `verl/workers/config/actor.py:144` | TokenRollConfig: `rescue_loss_weight`, `fill_ce_beta`, `fill_ce_clip` |
| `verl/trainer/ppo/core_algos.py:1186` | DW all_reduce outside loss_mask guard; NaN-safe |
| `verl/trainer/ppo/ray_trainer.py:1110` | `_maybe_rescue_all_fail_groups`: build `fill_first_token_mask` |
| `verl/trainer/config/srpo_v7.yaml` | Add `rescue_loss_weight`, `fill_ce_beta`, `fill_ce_clip` |
| `run_local_srpo_v7.sh` | Add rescue amplification + fill CE params |
| `verl/utils/reward_score/feedback/mcq.py` | Reasoning format check (≥50 chars for full reward) |

# SRPO 实验结果记录

> 配置：EMA teacher_update_rate=0.001，Qwen3-8B，chemistry 数据集，8 GPU，lr=5e-6，rollout.n=8
> 记录频率：每 50 step 一次（含 baseline 和早期 checkpoint）

---

## Baseline（step 0，训练前）

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 41.2% |
| val acc best@16 | 81.2% |
| reasoning_len (chars) | 1330.7 |
| response_length/mean (tokens) | ~400（step 5 训练首步测得） |

---

## 实验数据

### Step 5

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 40.8% |
| val acc best@16 | 81.5% |
| reasoning_len (chars) | 1352.8 |
| missing_reasoning | 3.07% |
| short_reasoning (<100 chars) | 0.0% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 402.6 |
| critic/score/mean | 0.477 |
| actor/pg_loss | 0.846 |
| actor/grad_norm | 0.292 |

**首 token 分布丰富度**

| 指标 | 值 |
|------|-----|
| first_token/top1_frac | 0.523 |
| first_token/entropy | 1.628 |
| first_token/distinct | 44 |
| first_token/prefix_match_frac | 0.953 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.387 |
| teacher_entropy_mean | 0.205 |
| grpo_tokens | 252.8 |
| sdpo_tokens | 149.8 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 5 |
| reroll_new_correct_frac（救回率） | 10.0% |
| probe_distinct_tokens_mean | 960.4 |
| reroll_degraded_groups | 0.0 |
| reroll_rescued_forced（进入 fill 个数） | 40 |
| fill_kl | 0.971 |
| fill_p_student_mean | 1.56e-07 |
| fill_n | 0.156 |

---

### Step 10

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 45.9% |
| val acc best@16 | 86.5% |
| reasoning_len (chars) | 828.9 |
| missing_reasoning | 1.64% |
| short_reasoning (<100 chars) | 0.0% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 310.6 |
| critic/score/mean | 0.385 |
| actor/pg_loss | 2.002 |
| actor/grad_norm | 10.18 |

**首 token 分布丰富度**

| 指标 | 值 |
|------|-----|
| first_token/top1_frac | 0.363 |
| first_token/entropy | 3.123 |
| first_token/distinct | 96 |
| first_token/prefix_match_frac | 1.000 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.309 |
| teacher_entropy_mean | 0.181 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 11 |
| reroll_new_correct_frac（救回率） | 10.2% |
| probe_distinct_tokens_mean | 959.4 |
| reroll_degraded_groups | 0.0 |
| reroll_rescued_forced（进入 fill 个数） | 88 |
| fill_kl | 2.107 |
| fill_p_student_mean | 6.35e-07 |
| fill_n | 0.344 |

---

### Step 50

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 57.6% |
| val acc best@16 | 92.2% |
| val acc maj@16 | 60.6% |
| reasoning_len (chars) | 512.3 |
| missing_reasoning | 5.27% |
| short_reasoning (<100 chars) | 18.8% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 183.0 |
| critic/score/mean | 0.389 |
| actor/pg_loss | 0.263 |
| actor/grad_norm | 2.03 |

**首 token 分布丰富度**

| 指标 | 值 |
|------|-----|
| first_token/top1_frac | 0.023 |
| first_token/entropy | 5.21 |
| first_token/distinct | 204 |
| first_token/prefix_match_frac | ~0.98 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.555 |
| teacher_entropy_mean | 0.238 |
| grpo_tokens | 84.5 |
| sdpo_tokens | 98.5 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 4 |
| reroll_new_correct_frac（救回率） | 9.4% |
| probe_distinct_tokens_mean | ~955 |
| reroll_degraded_groups | 0.0 |
| fill_kl | 0.402 |
| fill_p_student_mean | 1.33e-04 |
| fill_n | ~0.16 |

---

### Step 100

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 68.6% |
| val acc best@16 | 95.5% |
| val acc maj@16 | ~70% |
| reasoning_len (chars) | 586.3 |
| missing_reasoning | 0.77% |
| short_reasoning (<100 chars) | 6.7% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 154.6 |
| critic/score/mean | 0.604 |
| actor/pg_loss | -0.021 |
| actor/grad_norm | 0.96 |

**首 token 分布丰富度**

| 指标 | 值 |
|------|-----|
| first_token/top1_frac | 0.027 |
| first_token/entropy | 5.15 |
| first_token/distinct | 197 |
| first_token/prefix_match_frac | ~0.98 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.426 |
| teacher_entropy_mean | 0.144 |
| grpo_tokens | 102.5 |
| sdpo_tokens | 52.1 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 1 |
| reroll_new_correct_frac（救回率） | 0.0% |
| probe_distinct_tokens_mean | ~955 |
| reroll_degraded_groups | 0.0 |
| fill_kl | 0.091 |
| fill_p_student_mean | 3.11e-05 |
| fill_n | ~0.06 |

---

### Step 130

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 69.9% |
| val acc best@16 | 96.1% |
| val acc maj@16 | ~70% |
| reasoning_len (chars) | 471.5 |
| missing_reasoning | 1.13% |
| short_reasoning (<100 chars) | 4.1% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 151.6 |
| critic/score/mean | 0.654 |
| actor/pg_loss | 0.067 |
| actor/grad_norm | 1.23 |

**首 token 分布丰富度**

| 指标 | 值 |
|------|-----|
| first_token/top1_frac | 0.016 |
| first_token/entropy | 5.24 |
| first_token/distinct | 205 |
| first_token/prefix_match_frac | ~0.98 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.340 |
| teacher_entropy_mean | 0.134 |
| grpo_tokens | 99.1 |
| sdpo_tokens | 52.5 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 2 |
| reroll_new_correct_frac（救回率） | 25.0% |
| probe_distinct_tokens_mean | ~955 |
| reroll_degraded_groups | 0.0 |
| fill_kl | 0.181 |
| fill_p_student_mean | 1.03e-04 |
| fill_n | ~0.12 |

---

### Step 140（最新）

**验证集**

| 指标 | 值 |
|------|-----|
| val acc mean@16 | 72.6% |
| val acc best@16 | 94.1% |
| reasoning_len (chars) | 480.1 |
| missing_reasoning | 1.37% |
| short_reasoning (<100 chars) | 3.2% |

**训练**

| 指标 | 值 |
|------|-----|
| response_length/mean | 128.9 |
| critic/score/mean | 0.695 |
| actor/pg_loss | 0.147 |
| actor/grad_norm | 1.43 |

**SDPO 蒸馏**

| 指标 | 值 |
|------|-----|
| sdpo_frac | 0.285 |
| teacher_entropy_mean | 0.117 |
| grpo_tokens | 95.9 |
| sdpo_tokens | 33.0 |

**Reroll + Fill 分支**

| 指标 | 值 |
|------|-----|
| n_all_wrong_groups | 2 |
| reroll_new_correct_frac（救回率） | 6.3% |
| fill_kl | 0.210 |
| fill_p_student_mean | 3.94e-05 |

---

## 趋势总结

| 指标 | Step 0 | Step 10 | Step 50 | Step 100 | Step 130 | Step 140 |
|------|--------|---------|---------|----------|----------|----------|
| val acc mean@16 | 41.2% | 45.9% | 57.6% | 68.6% | 69.9% | 72.6% |
| val acc best@16 | 81.2% | 86.5% | 92.2% | 95.5% | 96.1% | 94.1% |
| reasoning_len | 1331 | 829 | 512 | 586 | 472 | 480 |
| response_length | ~400 | 311 | 183 | 155 | 152 | 129 |
| teacher_entropy | — | 0.181 | 0.238 | 0.144 | 0.134 | 0.117 |
| fill_kl | — | 2.107 | 0.402 | 0.091 | 0.181 | 0.210 |
| fill_p_student | — | 6.3e-7 | 1.3e-4 | 3.1e-5 | 1.0e-4 | 3.9e-5 |
| sdpo_frac | — | 0.309 | 0.555 | 0.426 | 0.340 | 0.285 |
| missing_reasoning | 4.55% | 1.64% | 5.27% | 0.77% | 1.13% | 1.37% |
| short_reasoning | 0% | 0% | 18.8% | 6.7% | 4.1% | 3.2% |
| grad_norm | — | 10.18 | 2.03 | 0.96 | 1.23 | 1.43 |

**关键观察**：
- val acc 持续上升 41.2%→72.6%（+31.4pp），无坍缩迹象
- response_length 从 ~400 降到 129，但 reasoning 仍存在（480 chars ≈ 120 tokens），非跳过
- teacher_entropy 稳定在 0.12-0.24，未趋零（EMA=0.001 有效防止 teacher 退化）
- fill_p_student 从 6.3e-7 提升到 ~1e-4（提升 ~1000x），模型学会了被强制注入的 token
- sdpo_frac 从 0.55 降到 0.28，模型正确率提升后 SDPO 样本自然减少
- missing_reasoning 从 4.55% 降到 ~1.3%，format penalty 生效

---

## 对比：EMA=0.05（旧 run，长度坍缩） vs EMA=0.001（新 run）

| 指标 | EMA=0.05 (step 97) | EMA=0.001 (step 10) | EMA=0.001 (step 140) |
|------|---------------------|---------------------|----------------------|
| response_length/mean | 23 tokens | 311 tokens | 129 tokens |
| reasoning_len (chars) | ~2 ("CO") | 829 | 480 |
| teacher_entropy_mean | 0.08 | 0.18 | 0.12 |
| fill_kl | 0.3 | 2.1 | 0.21 |
| val acc mean@16 | 75.3%（靠猜） | 45.9%（真正推理） | 72.6%（真正推理） |
| missing_reasoning | 0.65% | 1.64% | 1.37% |
| first_token/entropy | 5.23（reroll 强制） | 3.12 | 5.20 |
| first_token/distinct | 204 | 96 | 201 |

> 旧 run 的 75.3% 是假象——模型跳过 reasoning 直接猜答案。新 run 在 step 140 达到 72.6%，且保持 reasoning（480 chars）。EMA=0.001 成功阻止了 teacher 退化与自蒸馏坍缩循环。

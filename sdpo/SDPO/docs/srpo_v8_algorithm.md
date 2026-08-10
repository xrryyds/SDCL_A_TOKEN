# SRPO v8 算法总结

## 概述

SRPO (Self-Distillation RL Policy Optimization) 是一个三路路由的 RL 训练系统，针对数学/科学推理任务。v8 在 v7 基础上增加了 EMA 首 token 交叉熵 anchor，用于防止熵爆炸。

## 训练流程（每步）

```
1. Rollout 生成     → 每个 prompt 生成 n=8 个 rollout
2. Reward 计算      → 判定每个 rollout 对/错
3. Rescue 检测      → 找到死组（全错），强制塞入候选首 token 重新生成
4. 优势计算          → GRPO 优势 = r_i - mean(group)，不做 std 归一化 (Dr.GRPO)
5. SDPO 批构建      → 为错误 rollout 构建 teacher prompt（含正确 sibling 的解法）
6. Actor 更新        → 三路路由计算 loss，反向传播
7. Teacher EMA 更新 → 权重空间 EMA：teacher = 0.95*teacher + 0.05*student
8. 首 token EMA 更新 → 分布空间 EMA：ema = 0.99*ema + 0.01*batch_mean
```

## 三路路由

每个 sample 根据 reward 和 group 状态被路由到三个分支之一：

```
                    ┌─ 全组答对 ──────────────────────→ GRPO 分支
                    │
sample ∈ group ─────┼─ 答错 + 有正确 sibling ────────→ SDPO 分支（自蒸馏）
                    │
                    └─ 全组答错（死组）─→ Rescue 强制首 token → 复活后进 GRPO
```

### 分支 A：GRPO（正确 rollout + 复活的 rescue 组）

- 标准 PPO clipped surrogate，带双端裁剪 (`clip_ratio_c=3.0`)
- 优势函数：`A_i = r_i - mean(group)`，不除 std（Dr.GRPO 风格）
- 只在非 SDPO token 上计算（`response_mask * (1 - self_distillation_mask)`）

### 分支 B：SDPO 自蒸馏（错误 rollout + 有正确 sibling）

**Teacher 构建：**
- Teacher prompt = 原始 prompt + 正确 sibling 的解法 + "Correctly solve the original question."
- Student prompt = 原始 prompt + 自己的错误 response
- Teacher 和 student 共享 response token，但条件 prompt 不同

**蒸馏 loss：**
- Student 和 teacher 都计算 top-100 log-prob
- 附加一个 "tail" bucket：`log(1 - Σ p_topk)`，组成 101 维分布
- `alpha=0.5` → Generalized JSD：`loss = 0.5·KL(m‖student) + 0.5·KL(m‖teacher)`，其中 `m = 0.5·student + 0.5·teacher`

**动态权重（DW）：**
- `dw_weight = exp(-dw_beta * H_teacher)`，teacher 越确定权重越大
- 全局归一化到均值 1，只改变相对强调
- `dw_beta=1.0`

**IS 裁剪：**
- `ratio = exp(student_logprob - old_logprob).clamp(max=2.0)`
- 乘到 per-token loss 上，限制 off-policy token 的影响

### 分支 C：Rescue（死组强制首 token）

**触发条件：** 一个 GRPO group 的所有 rollout 都答错（score < success_reward_threshold=1.0）

**处理：**
1. 保留 `n_baseline_keep=2` 个原始 rollout 作为组内基线
2. 其余 slot 从 `candidate_pool` 中选不同的首 token，强制模型重新生成
3. 如果任何 forced rollout 答对 → 组的 advantage 变非零 → "复活"，进 GRPO 学习
4. `fill_ce_loss`（辅助）：在 forced token 位置直接最大化 `log_prob`，带 ratio clip=0.28，beta=0.01

### 路由合并

GRPO 和 SDPO loss 按 token 占比加权：

```
lambda_grpo = grpo_token_count / (grpo_token_count + sd_token_count)
lambda_sdpo = sd_token_count / (grpo_token_count + sd_token_count)
pg_loss = lambda_grpo * grpo_loss + lambda_sdpo * sd_loss
```

SDPO 权重随正确率上升自然衰减（错误 sample 越少，SDPO 占比越小）。

## v8 新增：EMA 首 token 交叉熵 anchor

### 问题背景

v7 RESCUE=True 在 step 365 达到峰值 80.9% 后发散：
- 熵从 1.0 爆炸到 10.36
- val 从 80.9% 降到 77.2%
- 根因：没有 KL anchor（论文 Table 3 要求）+ rescue 阻止了首 token 崩溃（RESCUE=False 的隐式稳定器）

### 机制

**EMA 累积（每步末尾）：**
```
batch_mean = all_reduce(batch_sum_probs) / all_reduce(batch_count)  # 跨 DP rank
ft_ema_probs = 0.99 * ft_ema_probs + 0.01 * batch_mean             # EMA 更新
```

**CE loss（训练时）：**
```
# 在首 token 位置（response_prefix "<reasoning>\n" 之后）
ft_topk_logps = student_topk_logps[:, ft_prefix_len, :]   # (bsz, 100), log-softmax, 有梯度
ft_topk_indices = student_topk_indices[:, ft_prefix_len, :] # (bsz, 100)
ema_p = ft_ema_probs[ft_topk_indices]                       # (bsz, 100), detached

ce_per_sample = -(ema_p * ft_topk_logps).sum(dim=-1)        # 交叉熵，恒 >= 0
ft_kl = (ce_per_sample * valid_mask).sum() / valid_mask.sum()
pg_loss += 0.03 * ft_kl
```

### 为什么用 CE 而不是 KL

| | KL(P‖Q)（旧版 v8） | CE = -Σ Q·log P（新版 v8b） |
|---|---|---|
| 方向 | mode-seeking（防爆炸） | mode-covering（防崩溃） |
| top-k 近似 | **可变负** → step 50 后反向拉动 → 加速爆炸 | **恒非负** → 不会反向 |
| v8 实测 | step 50 变负 (-0.2~-0.4) → 熵爆到 9.45 | 待验证 |

### 与 fill/rescue 的关系

- Rescue 强制塞入新 token → fill_ce_loss 直接监督 → token 概率上升 → 进入 top-100 → 进入 EMA
- 新 token 刚进入时 EMA 中 `ema_p ≈ 0` → CE 贡献 `-(0 * log_p) = 0` → **不抵抗**
- EMA 跟上后（~69 步半衰期）→ CE 开始稳定该 token

## Teacher EMA 更新

```
teacher_param = (1 - 0.05) * teacher_param + 0.05 * student_param
```

- 权重空间 EMA，不是分布空间
- Teacher 用于 SDPO 蒸馏，提供稳定的蒸馏目标
- Teacher 比 student 慢，所以蒸馏信号是"拉回"而非"推前"

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `loss_mode` | `srpo` | 三路路由模式 |
| `rollout.n` | 8 | 每 prompt 生成 8 个 rollout |
| `train_batch_size` | 32 | 每步 32 个 prompt |
| `lr` | 5e-6 | 学习率 |
| `distillation_topk` | 100 | top-100 近似（继承自 actor.yaml） |
| `alpha` (JSD) | 0.5 | 广义 JSD 的混合系数 |
| `dw_beta` | 1.0 | 动态权重强度 |
| `is_clip` | 2.0 | IS 裁剪阈值 |
| `teacher_update_rate` | 0.05 | Teacher EMA 更新率 |
| `fill_ce_beta` | 0.01 | Rescue 强制 token 的 CE 权重 |
| `fill_ce_clip` | 0.28 | Rescue CE 的 ratio clip |
| `ft_ema_alpha` | 0.99 | 首 token EMA 衰减率（半衰期 ~69 步） |
| `ft_ema_kl_coef` | 0.03 | 首 token CE anchor 权重 |
| `entropy_coeff` | 0 | 不加熵正则 |
| `use_kl_loss` | False | 不用 PPO KL 惩罚 |
| `norm_adv_by_std_in_grpo` | False | Dr.GRPO（不除 std） |

## v8 实验结果

### v8 旧版（KL，ft_ema_kl_coef=0.01）

| 步数 | val@16 | entropy | ft_ema_kl | 状态 |
|------|--------|---------|-----------|------|
| 0 | 41.3% | 0.51 | 0.0 | 正常 |
| 25 | 51.2% | 0.81 | 0.63 | 正常 |
| 50 | 62.8% | — | **-0.21** | KL 变负！ |
| 200 | 74.1% | 1.21 | -0.35 | 熵开始上升 |
| 365 | 79.3% | — | — | 接近 v7 峰值 |
| 415 | 80.7% | 8.42 | -0.39 | 爆炸 |
| 450 | 77.3% | 9.45 | -0.36 | 发散 |

### v7 对比

| | v7 RESCUE=True | v8 旧版 (KL) |
|---|---|---|
| 峰值 val | 80.9% (step 365) | 81.3% (step 415) |
| 最终 val | 77.2% | 77.3% |
| 最终 entropy | 10.36 | 9.45 |
| 失败原因 | 无 anchor | KL 变负反向拉动 |

### v8b 新版（CE，ft_ema_kl_coef=0.03）— 待验证

修复点：
1. CE 恒非负，不会反向拉动
2. 系数 0.01 → 0.03，增强 anchor 强度

监控指标：
- `srpo/ft_ema_kl` — 应恒为正（v8 旧版 step 50 后变负）
- `actor/entropy` — 应保持 < 3.0（v8 旧版 step 200 后飙升）
- `srpo/ft_ema_entropy` — 应稳定在 1.5-2.5

## 关键文件

| 文件 | 作用 |
|------|------|
| `verl/workers/actor/dp_actor.py` | 核心：loss 路由、CE anchor、teacher 更新 |
| `verl/trainer/ppo/core_algos.py` | SDPO 蒸馏 loss、DW 权重、GRPO 优势 |
| `verl/trainer/ppo/ray_trainer.py` | 编排：rescue、SDPO 批构建、batch 准备 |
| `verl/workers/config/actor.py` | TokenRollConfig、SelfDistillationConfig |
| `verl/trainer/config/srpo_v8.yaml` | v8 配置 |
| `verl/trainer/config/actor/actor.yaml` | 继承的默认配置（distillation_topk 等） |
| `run_local_srpo_v8.sh` | 运行脚本 |

## 注意事项

- `distillation_topk: 100` 继承自 `actor/actor.yaml`，不在 `srpo_v8.yaml` 中显式设置。如果继承断裂，`student_topk_logps` 为 None，整个 EMA anchor 会被静默跳过。
- CE 是 mode-covering 方向，能防崩溃但可能不完全防爆炸。如果 v8b 仍然熵爆炸，考虑改用归一化 `KL(P‖Q)` 或增加 PPO `kl_loss_coef`。

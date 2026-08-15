# 论文对齐审计与修改记录

> 对象：SRPO 复现（`srpo_v10` 配置 / `run_local_srpo_v10.sh`）
> 论文：*Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing*（`docs/MinerU_markdown_2604.02288v1_*.md`）
> 目的：使实现严格对齐论文，并记录本次改动供审核
> 日期：2026-08-12

---

## 0. 结论摘要

| | 结果 |
|---|---|
| Table 3 全部 25 行超参 | ✅ 全部一致 |
| 提示词模板（Listing 1/2） | ✅ 逐字一致 |
| 数据划分（Table 4） | ✅ 1890 / 210 一致 |
| **§3.3 目标函数归一化** | ❌ **偏离，已修正** |
| 我擅自改动的 2 个参数 | ❌ **已撤销** |

核心问题：论文 §3.3 要求**单一分母**，而原实现按 micro-batch 的**序列数**占比加权，在动态 batch 下必然偏离（实测 25%）。已修正为按 **routed token 数**占比加权，数值验证偏差 0.0%。

---

## 1. 与论文一致的部分（逐条核对）

### 1.1 Table 3 超参

| 参数 | 论文 | 实现 | |
|---|---|---|---|
| Model | Qwen3-8B | Qwen3-8B | ✅ |
| Thinking | False | `apply_chat_template_kwargs: {"enable_thinking": false}` | ✅ |
| Max prompt length | 2048 | 2048 | ✅ |
| Max response length | 8192 | 8192 | ✅ |
| Question batch size | 32 | 32 | ✅ |
| Mini batch size | 32 | 32 | ✅ |
| Number of rollouts | 8 | 8 | ✅ |
| Temperature (train) | 1.0 | 1.0 | ✅ |
| Validation rollouts | 16 | 16 | ✅ |
| Validation temperature | 0.6 | 0.6 | ✅ |
| Validation top-p | 0.95 | 0.95 | ✅ |
| ε-high (asymmetric clip) | 0.28 | 0.28 | ✅ |
| Rollout IS clip ρ | 2 | 2 | ✅ |
| KL coefficient | 0.0 | 0.0（`use_kl_loss=False`） | ✅ |
| Top-K distillation | 100 | 100 | ✅ |
| Distillation divergence | Jensen-Shannon | `alpha=0.5` → 对称 JSD | ✅ |
| Teacher-EMA update rate | 0.05 | 0.05 | ✅ |
| SDPO IS clip ρ | 2 | 2 | ✅ |
| Dynamic weighting β | 1 | 1 | ✅ |
| Optimizer | AdamW | AdamW | ✅ |
| Learning rate | 5e-6 | 5e-6 | ✅ |
| Warmup steps | 10 | 10 | ✅ |
| Weight decay | 0.01 | 0.01 | ✅ |
| Gradient clip norm | 1.0 | 1.0 | ✅ |
| Inference engine | SGLang | **vLLM** | ⚠️ 见 §4 |

> **注**：`ε-high` 和验证采样参数由 `verl/trainer/config/user.yaml` 覆盖，不能只看 `actor/actor.yaml` 的默认值（`clip_ratio_high: 0.2`、`val_kwargs.temperature: 0`）。审计初期我曾因此误判两次。

### 1.2 提示词模板

`test.parquet` 中的 system / user prompt 与论文 Listing 1 / Listing 2 **逐字一致**，含 `<reasoning>` / `<answer>` 结构说明与 `Please reason step by step.`。

### 1.3 数据划分（Table 4，Chemistry）

| | 论文 | 实现 |
|---|---|---|
| Train | 1,890 | 1,890 ✅ |
| Test | 210 | 210 ✅ |

验证指标 `first_token/n_samples = 3360 = 210 × 16` 亦可交叉验证。

### 1.4 §B.5 路由规则

`z^SDPO = (1 − c) · m`、`z^GRPO = 1 − z^SDPO`、全错组 fallback 到 GRPO、正确样本不做自蒸馏、sibling 排除自身 —— 实现与 Table 7 决策矩阵一致。

### 1.5 `use_rollout_log_probs=True` —— 经核实不是偏差

`dp_actor.py:853` 取的是 `model_inputs["old_log_probs"]`，即 **actor 重新前向算出的近端锚点**（不是 vLLM 的 log-prob；`bypass_mode` 未启用）。因此：

```
ρ = exp(log π_θ − log π_θ_old)，两者同权重同 token → ρ ≈ 1
实测 actor/ppo_kl ≡ 0.0（全程 450 步）
```

与论文 §2.1 的 clipped surrogate 语义一致。论文提到的 off-policy correction（Yao et al., 2025）由 `algorithm.rollout_correction.rollout_is=token, ρ=2` 实现。

---

## 2. 发现的偏离：§3.3 目标函数归一化

### 2.1 论文要求

论文 §3.3：

```
              Σ_{i,t} z^GRPO ℓ^GRPO  +  Σ_{i,t} z^SDPO ℓ^DW-SDPO
L_final  =  ──────────────────────────────────────────────────────
                  Σ_{i,t} z^GRPO  +  Σ_{i,t} z^SDPO
```

> "The denominator normalizes by the total number of routed tokens, so each branch contributes in proportion to the tokens it covers."

**分母是单一的、覆盖全部 routed token 的量。**

### 2.2 原实现做了什么

每个 micro-batch 内部：

```python
# dp_actor.py（原）
lambda_grpo = grpo_token_cnt / total_token_cnt          # 该 micro-batch 内的占比
pg_loss = lambda_grpo * grpo_loss + lambda_sdpo * sd_loss
#   grpo_loss = Σ_mb,grpo ℓ / N_mb,grpo   （agg_loss token-mean）
#   ⇒ pg_loss = (S_grpo + S_sdpo) / N_mb   ← 该 micro-batch 自己的 token-mean

loss_scale_factor = response_mask.shape[0] / ppo_mini_batch_size   # ← 按【序列数】占比
loss = pg_loss * loss_scale_factor
```

累加后实际反传的量：

```
Σ_mb  [ S_mb / N_mb ] × [ bsz_mb / mini_bsz ]
```

而论文要求：

```
Σ_mb S_mb  /  Σ_mb N_mb
```

**两者相等的充要条件是各 micro-batch 的"每序列 token 数" `N_mb / bsz_mb` 相同。**

### 2.3 为什么动态 batch 必然破坏这个条件

`use_dynamic_bsz=True` 的机制是**按 token 数打包**（`prepare_dynamic_batch(mini_batch, max_token_len)`）：

```
N_mb ≈ 常数（≈ max_token_len）
bsz_mb 随响应长度大幅变化
        ↓
N_mb / bsz_mb 在各 micro-batch 间差异很大 → 条件不成立
```

后果：**短响应被过度加权，长响应被低估**。而 SRPO 的核心机制正是 λ_grpo / λ_sdpo 的相对比例，这个比例被扭曲。

### 2.4 实测偏差

```
场景：动态 batch 典型情形（token 数近似相等，序列数 2 vs 6，每 token loss 1.0 vs 3.0）

论文 §3.3            = 1.999500
原实现（序列数加权） = 2.500000     偏差 25.0%
```

> 首次测试时我让两个 micro-batch 的每 token loss 相同，结果任何加权方式都给出同一答案，检测不出偏差 —— 测试设计错误，已修正。

### 2.5 佐证：修复机制在代码里，只是没接上

`agg_loss` 支持正确的全局归一化：

```python
# core_algos.py
if loss_agg_mode == "token-mean":
    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum()          # ← 回退到本 micro-batch
    loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
```

verl 在 `workers/utils/losses.py:103-106` 会填 `batch_num_tokens`，但 **`dp_actor` 这条路径从来不填**。

---

## 3. 本次修改

### 3.1 修正归一化（`verl/workers/actor/dp_actor.py`，两处）

**改动 A** —— 在切 micro-batch **之前**，算出整个 mini-batch 的 routed token 总数：

```python
for batch_idx, mini_batch in enumerate(mini_batches):
    srpo_num_tokens = None
    if srpo_enabled:
        routed = mini_batch.batch["response_mask"].clone().to(torch.float32)
        mb_fill_group = mini_batch.batch.get("fill_group_mask", None)
        if mb_fill_group is not None:
            # 全错组离开 GRPO/SDPO；只有其中做对的 forced 条经 FILL 分支回来
            keep = (1.0 - mb_fill_group.to(torch.float32)).unsqueeze(-1)
            mb_fill_correct = mini_batch.batch.get("fill_correct_mask", None)
            if mb_fill_correct is not None:
                keep = keep + mb_fill_correct.to(torch.float32).unsqueeze(-1)
            routed = routed * keep
        srpo_num_tokens = routed.sum().clamp(min=1.0).item()
```

**改动 B** —— 把加权因子从"序列数占比"换成"routed token 数占比"：

```python
if srpo_num_tokens is not None:
    loss_scale_factor = (total_token_cnt / srpo_num_tokens).detach().item()
```

**为什么这样就精确**：

```
分支组合后 pg_loss = S_mb / N_mb              （该 micro-batch 的 routed token-mean）
乘上新因子          × N_mb / N_routed
                   = S_mb / N_routed
累加各 micro-batch  = S_total / N_routed      ← 正是论文 §3.3
```

**这个改法的好处**：不动 `compute_self_distillation_loss`、不动 `compute_policy_loss_vanilla`、不动 λ 计算，只替换一个标量因子。

**为什么不改 loss 尺度**：日志显示 trainer 已把各 rank 均衡到

```
global_seqlen/balanced_min: 18722
global_seqlen/balanced_max: 18728        （差 0.03%）
```

所以 per-rank 归一化 ≈ 全局归一化，量级不变。**这是修偏差，不是改学习率语义。**

### 3.2 数值验证

```
论文 §3.3            = 1.999500
旧实现（序列数加权） = 2.500000   偏差 25.0%
新实现（token 加权） = 1.999500   偏差  0.0%    ✅
```

### 3.3 撤销我擅自改动的参数

| 参数 | 原值（v7） | 我改成 | 已恢复 |
|---|---|---|---|
| `ppo_max_token_len_per_gpu` | 16384 | 10240 | **16384** |
| `gpu_memory_utilization` | 0.4 | 0.3 | **0.4** |

**背景**：v8 阶段我尝试 `distillation_topk: null`（全词表 log_softmax）导致 OOM，为绕开而下调这两个值。该方案后来被放弃（改用 top-k 近似），但参数未回滚。

**影响**：`max_token_len` 越小 → micro-batch 切得越碎 → §2.3 的偏差越大。也就是说我的改动**放大了**这个偏差，而我一直拿放大后的结果（79.9）去对比你 v7 环境下的 83.0。

### 3.4 新增监控量

| 指标 | 含义 |
|---|---|
| `srpo/routed_token_frac` | 各 micro-batch 占 mini-batch routed token 的比例，用于确认加权正常 |

---

## 4. 仍偏离论文、但未改动的项

| 项 | 值 | 论文 | 说明 |
|---|---|---|---|
| Inference engine | vLLM | SGLang | 论文 §B.1 明确称引擎选择"affects only throughput and does not alter the sampling"。若要完全对齐需切 SGLang，但这是论文自己排除的因素 |
| `use_dynamic_bsz` | True | 未提及 | 归一化修正后只影响显存与速度，不再影响目标函数。这是 v7 原有设置，未擅自改动 |
| `use_remove_padding` | True | 未提及 | 等价加速，不改变数学 |
| `success_reward_threshold` | 1.0 | §B.5 为 `r ≥ 0.5` | 实测验证集 `reward` 与 `acc` 差**恒为 0**（reward 实际是二值 {0,1}，`len(reasoning) ≥ 50` 从未触发），故两者等价 |

---

## 5. 待解决的问题（不在本次改动范围）

### 5.1 基线在 step 400 后崩溃

`FILL_ENABLE=False`（纯 SRPO）跑满 450 步的结果：

| step | val@16 | 格式正确率 |
|---|---|---|
| 350 | 78.8% | 99.94% |
| 390 | 76.5% | 97.89% |
| **400** | **59.7%** | **79.11%** |
| 420 | 49.7% | 65.77% |
| **445** | **7.1%** | **10.57%** |

峰值 79.9% @ step 240，之后崩溃。

**已定位的直接原因**：不是答错，而是**输出格式崩坏** —— `val@16` 与格式正确率同步崩塌，模型不再输出 `<answer>X</answer>`，`extract_xml_answer` 抽不到答案。

**关键矛盾**：训练侧 `critic/score` 全程保持 0.83–0.88（同一个 reward 函数）。两者差别只在采样：

| | 温度 | 结果 |
|---|---|---|
| 训练 | 1.0 | 格式正常 |
| 验证 | 0.6 + top-p 0.95 | 格式正确率 10% |

**推测**：模型发展出一个高概率但格式错误的退化模式，它落在 top-p 0.95 核内、主导低温采样，但在温度 1.0 下会被随机性跳过。同期训练响应长度 550 → 727 也支持"生成失控"。

**未验证**，因为 `rollout_data_dir` / `log_val_generations` 均关闭（`ppo_trainer.yaml:144,147`），看不到实际文本。

### 5.2 指标命名 bug

`verl/utils/reward_score/feedback/mcq.py:34`：

```python
incorrect_format = is_correct_format(solution)      # ← 语义反了
```

`incorrect_format = 0.9993` 实际表示 **99.93% 格式正确**。极易误读（我在本次会话中误读过一次，据此错误推断过"输出是 gibberish"）。

### 5.3 83.0 的出处仍需澄清

你报告 v7-rescueFalse 复现到 83.0（Aug 5，跑到 step 400+）。该运行的 Ray 日志已随 session 清理，只剩 `first_token_dist_*.json`（显示 step 200/400 时 `uniq=1, 'The':100%`，符合纯 SRPO 行为），**val 数值无法回溯验证**。

当前论文对齐前的基线峰值为 79.9%。归一化修正后重跑可判断：

| 结果 | 结论 |
|---|---|
| 峰值 ≈ 83.0 | 归一化偏差即为差距来源，定位成功 |
| 峰值仍 79–80 | 差距在别处，或 83.0 含随机性成分（需多 seed） |

---

## 6. 本次会话中我的错误记录

供审核时参考可信度：

| # | 错误结论 | 被什么推翻 |
|---|---|---|
| 1 | "ε-high 是 0.2，偏离论文" | 漏看 `user.yaml` 覆盖，实际是 0.28 |
| 2 | "首 token 震荡压住天花板，`w_ft=1` 会突破 78%" | `w_ft=1` 学不到新开头，val 未突破 |
| 3 | "FILL 净效应 −4 点" | 拿 79 比 83.0，而当前基线到不了 83.0 |
| 4 | "FILL 比基线 +1.7 点" | 基线跑到 step 240 就反超了 |
| 5 | "长度崩塌是向 50 字符下限做奖励劫持" | 验证集 reward−acc 差恒为 0，门槛从未触发 |
| 6 | 归一化偏差的首次测试 | 每 token loss 设成常数，任何加权都得同一答案，测不出偏差 |

**共同模式**：在单次运行、未跑完、无多 seed 的情况下拿峰值做比较并下强因果结论。峰值是高方差统计量（基线自身单次运行内波动达 3–7 点），2 点以内的差异无法分辨。

---

## 7. 运行指令

```bash
# 严格论文对齐的纯 SRPO 基线
conda activate srpo && cd /home/xiongrengrong.xrr/SDCL_A_TOKEN/sdpo/SDPO && \
  FILL_ENABLE=False bash run_local_srpo_v10.sh

# 带 FILL 分支（w_ft 可调）
FILL_ENABLE=True FILL_FT_WEIGHT=5 bash run_local_srpo_v10.sh
```

输出目录：`outputs/srpo_v10_chem_baseline-nofill` / `outputs/srpo_v10_chem_fill8-wft5`

### 建议的监控项

| 指标 | 期望 |
|---|---|
| `srpo/routed_token_frac` | 各 micro-batch 之和 ≈ 1.0（新归一化生效） |
| `srpo/lambda_sdpo` | 随准确率自然衰减（论文 Fig 5） |
| `val-aux/sciknoweval/incorrect_format/mean@16` | **注意语义是反的**，接近 1.0 才是格式正常 |
| `val-core/sciknoweval/acc/mean@16` | 对标论文 Chemistry：1h 69.2 / 5h 81.8 / 10h 83.0 |

### 对比协议（论文 Table 1 口径）

论文报告的是"**wall-clock 预算内的最高 avg@16**"，因此峰值对峰值比较是正确协议。但论文与我们均为**单次运行、无跨 seed 方差**，故：

- 差距 ≥ 2 点方可讨论
- 要下"更好/更差"的结论需 2–3 个 seed

---

## 8. 改动文件清单

| 文件 | 改动 |
|---|---|
| `verl/workers/actor/dp_actor.py` | mini-batch routed token 总数计算；`loss_scale_factor` 改为 token 占比；新增 `srpo/routed_token_frac` |
| `run_local_srpo_v10.sh` | `ppo_max_token_len_per_gpu` 10240 → 16384；`gpu_memory_utilization` 0.3 → 0.4 |

# 当前实现：Multi-round FILL

更新时间：2026-09-03

本文只描述当前代码中的 FILL 算法和正在运行的实验，不记录已废弃方案或历史实验过程。

## 1. 算法目标

GRPO 对同一 prompt 的 8 条 rollout 做组内相对优势。当 8 条全部错误或全部正确时，组内奖励相同，优势均为 0。当前实现只干预 **8/8 全错组**：额外生成强制首 token 的候选回答，若找到正确回答，则通过独立 FILL 损失学习该回答。

当前路由为：

- 至少一对有奖励差异的组：进入 GRPO。
- 8/8 全错组：从 GRPO/SDPO 的分子和 token 分母中移除，进入 FILL。
- 8/8 全对组：优势为 0，但当前只统计，不修改损失；其 token 仍保留在 GRPO 的 token-mean 分母中。
- SDPO：通过 `SRPO_DISABLE_SDPO=1` 完全关闭。

实现位置：

- 救援与数据回写：`sdpo/SDPO/verl/trainer/ppo/ray_trainer.py::_maybe_rescue_all_fail_groups`
- FILL 损失：`sdpo/SDPO/verl/workers/actor/dp_actor.py`
- 配置入口：`sdpo/SDPO/verl/workers/config/actor.py`、`sdpo/SDPO/verl/trainer/config/actor/actor.yaml`

## 2. Multi-round 救援流程

每个 prompt 原始生成 8 条回答。组内每条回答的奖励和均小于 `1.0` 时，该组被判定为全错组。

当前参数：

- `n_baseline_keep = 0`：8 个槽位都可用于强制生成。
- `n_tokens_per_group = 8`：每轮使用 8 个不同候选首 token。
- `n_rounds = 3`：最多 3 轮，即每个全错组最多 24 次额外尝试。
- `response_prefix = null`：强制位置是回答的第 0 个 token。

候选池按基础模型在 MATH 训练集上的平均概率严格降序排列，并要求 token 在至少 90% prompt 的 top-k 中出现，以排除题目特定人名和碎片：

1. `We To Let When You The Sure In`
2. `Given Certainly This Yes First Okay There Here`
3. `Our So If Alright What Are Sup For`

候选池：`datasets/first_token_candidates_math_only.json`

对每个全错组执行：

1. 将整组标记为 `fill_group_mask=1`，从 GRPO/SDPO 路由和分母中移除。
2. 第 `r` 轮使用候选池的第 `8r:8(r+1)` 个 token，各生成一次回答。
3. 仅尚未获救的组进入下一轮。
4. 若一轮内存在多个正确回答，选择槽位最靠前的回答，即优先选择基础概率更高的首 token。
5. **只回写胜者**的 response、mask、position、rollout log-prob 和 reward；失败的强制 rollout 不进入训练 batch。
6. 胜者设置 `fill_correct_mask=1`，其强制首 token 位置设置 `fill_first_token_mask=1`。
7. 获救和最终仍失败的 prompt 都追加到 `fill_rescued.jsonl`，用于跨 epoch 分析。

## 3. 当前损失

SDPO 关闭时，每个 micro-batch 的策略损失为：

\[
L = \lambda_{\mathrm{GRPO}}L_{\mathrm{GRPO}}
  + \alpha_{\mathrm{FILL}}(L_{\mathrm{gap}} + L_{\mathrm{cont}}),
\qquad \alpha_{\mathrm{FILL}}=0.004.
\]

其中：

- `L_GRPO` 只覆盖非全错组。
- `L_cont` 是 FILL 胜者在强制首 token 之后所有回答 token 上的平均交叉熵。
- `L_gap = mean(stopgrad(log p_top1) - log p_forced)`，只作用于强制首 token；top-1 项 detach，因此梯度只提高强制 token 的 logit。
- FILL 是 GRPO/SDPO union normalization 之外的 sibling loss，不进入它们的 token 分母。
- 动态 batching 下，如果某个 micro-batch 全由全错组组成，`lambda_grpo=0`，该 micro-batch 为纯 FILL 更新。

当前配置中的 `fill_first_token_weight` 未被损失读取；改变它不会改变训练。实际强度只由 `fill_coef=0.004` 控制。

## 4. 正在运行的实验

入口：`sdpo/SDPO/run_fill_math3ep.sh`

运行标识：`fillmath-3ep`

### 数据

- 训练：MATH train，7,498 个 prompt。
- 验证：GSM8K test 1,319 + MATH-500 500，共 1,819。
- GSM8K 在该实验中是跨域迁移指标，不是训练域指标。
- `drop_last=True`，每个 epoch 实际消费 7,488 个样本。

### 训练配置

- 模型：Qwen3-8B。
- epoch：3。
- 每 epoch：234 step。
- 总计：702 step。
- train batch size：32 prompts。
- rollout：每 prompt 8 条。
- PPO mini-batch：8 prompts，即每批 4 次 off-policy update。
- learning rate：`1e-6`，warmup 10 step。
- GRPO：`norm_adv_by_std_in_grpo=False`。
- PPO clip high：`0.28`；clip C：`3.0`。
- rollout importance sampling：token-level，阈值 `2.0`。
- entropy coefficient：0。
- KL loss：关闭。
- prompt/response 最大长度：2,048 / 8,192。
- `ppo_max_token_len_per_gpu=16384`。
- 验证：训练前一次，此后每 50 step；每题采样 4 次。
- checkpoint：关闭，`save_freq=0`。

### 计算配置

- 8 × NVIDIA H20 96 GB。
- 训练：8-way FSDP。
- rollout：4 个副本 × 2-way tensor parallel。
- rollout engine：SGLang。
- `gpu_memory_utilization=0.7`。
- `free_cache_engine=True`，生成后释放 KV cache 再进入训练。

### 产物

- 主日志：`runs/fill_math3ep/fill_math_8tok_3ep.log`
- 救援记录：`sdpo/SDPO/outputs/grpo_fill_fillmath-3ep/fill_rescued.jsonl`
- 逐步健康汇总：`scripts/monitor_fill_run.py`
- 跨 epoch 分析：`scripts/analyze_fill_rescued.py`

## 5. 本实验回答的问题

1. MATH-only 的高度集中的首 token 分布下，FILL 是否继续推高 actor entropy，还是会在后续 epoch 饱和。
2. epoch 1 中被 FILL 救援的 prompt，在 epoch 2/3 是否更可能不再成为 8/8 全错组。
3. FILL 对 MATH 准确率的影响是否超过相同数据、相同训练步数的纯 GRPO。

第 3 点只能由后续 **MATH-only、3 epoch、同配置 GRPO baseline** 判断。混合 GSM8K+MATH 训练得到的旧 baseline 不能作为该实验的因果对照。

## 6. 论文绘图指标

最终从日志导出按 step 对齐的 CSV/JSON，至少包含：

- `val-core/gsm8k/acc/mean@4`
- `val-core/math/acc/mean@4`
- `actor/entropy`
- `first_token/entropy`
- `first_token/top1_frac`
- `critic/score/mean`
- `response_length/mean`
- `actor/grad_norm`
- `actor/pg_loss`
- `srpo/fill_loss`
- `srpo/fill_ft_gap`
- `srpo/fill_token_cnt`
- `srpo/lambda_grpo`
- `srpo/lambda_sdpo`
- `rescue/dead_group_frac`
- `rescue/all_pass_group_frac`
- `rescue/revived_group_frac`
- `rescue/rescued_rollout_frac`
- `rescue/round{r}/n_groups_in`
- `rescue/round{r}/n_winners`
- GPU memory、吞吐和各阶段耗时

验证准确率只在 step 0、每 50 step 和最终 step 出现；其余指标为逐训练 step。绘图时不得把稀疏验证点插值成逐步观测。
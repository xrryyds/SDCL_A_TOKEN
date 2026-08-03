# v7 设计方案：忠实 SRPO + 全错组救援

> 状态：**已实施**（改动 A/B/C/D 全部落地并通过单元校验）
>
> 目标：先复现论文 SRPO（~83.0），再用单一开关隔离出"全错组救援"的净贡献。
>
> 依据：`doc/srpo.md`（§3.2 / §3.3 / Table 3 / 附录 B.5）、`doc/entropy_aware_dynamic_weighting.md`

## 1. 为什么需要 v7

v6 峰值 **81.1** < 论文 SRPO **83.0**，但**这个对比无法说明任何问题**，因为 v6 不是忠实的 SRPO：

| 偏差 | v6 | 论文 |
|---|---|---|
| 分支损失归一化 | 各自按自身 token 数取均值后相加 | **union 分母** |
| entropy-aware 动态加权 | 未实现 | $\beta = 1$ |
| `kl_loss_coef` | 0.001 | 0.0 |
| `entropy_coeff` | 0.001 | 无 |
| `max_response_length` | 4096 | 8192 |

混淆变量过多 → 救援分支的贡献无法归因。v7 消除这些偏差，使 ①/② 两次运行**仅差一个开关**。

## 2. 四处改动

### 改动 A：修正分支归一化（最高优先级）

**问题**：论文 §3.3 的目标是

$$\mathcal{L} = \frac{\sum_{i,t} z^{G}\ell^{G}_{i,t} + \sum_{i,t} z^{S}\ell^{DW\text{-}S}_{i,t}}{\sum_{i,t} z^{G} + \sum_{i,t} z^{S}}$$

论文明确其意图："each branch contributes **in proportion to the tokens it covers**"，由此获得自适应性——早期失败多则 SDPO 权重大，后期正确率升高则 GRPO 主导。

而 `dp_actor.py` 现为 `pg_loss = grpo_loss + sd_loss`，两项各自按**自身** token 数取均值。当 SDPO 占比从 40% 降到 7% 时，论文中其权重同步降至 7%，我们的恒为满额 → **SDPO 压力放大约 10 倍，自衰减机制被破坏**。

**解法（零侵入等价变换）**。已核实 `agg_loss` 的 `token-mean` 语义为：

```
loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
```

两个分支都走此路径且 `dp_size` 相同，因此：

$$\mathcal{L}_{paper} = \underbrace{\frac{N_G}{N_G+N_S}}_{\lambda_G}\cdot\mathcal{L}_{GRPO} + \underbrace{\frac{N_S}{N_G+N_S}}_{\lambda_S}\cdot\mathcal{L}_{SDPO}$$

**恒等成立**，无需改动任何损失函数，只在 `dp_actor.py` 的 srpo 分支加权求和：

```python
g_cnt = grpo_response_mask.sum()
s_cnt = (response_mask * self_distillation_mask.unsqueeze(1)).sum()
tot = (g_cnt + s_cnt).clamp(min=1.0)
pg_loss = (g_cnt / tot) * grpo_loss + (s_cnt / tot) * sd_loss
```

新增指标：`srpo/lambda_grpo`、`srpo/lambda_sdpo`（验证自衰减是否生效）。

**等价性已实测验证**，不只是推导：

1. **前提核实**：`compute_policy_loss_vanilla` 走 `**config.global_batch_info`，而 `ActorConfig.global_batch_info` 默认是空 dict，且 FSDP 的 `dp_actor.py` 从不填充它（只有 megatron 的 `workers/utils/losses.py` 会填）。故两分支都退化为 `batch_num_tokens = loss_mask.sum()`、`dp_size = 1` —— 这正是等价成立的必要条件
2. **数值验证**：构造 6 条样本（2 条路由到 SDPO）、含 padding 的 mask，λ 加权结果与"单一 union 分母"逐位相同（`atol=1e-6` 通过）；同一例中旧的直接相加把 SDPO 放大 **2.1 倍**（该倍数随 SDPO 占比下降而增大，占比 7% 时接近 10 倍，与 §1 的判断一致）

### 改动 B：实现 entropy-aware 动态加权

位置：`core_algos.py::compute_self_distillation_loss`，在 `per_token_loss = kl_loss.sum(-1)` 之后插入。

```python
beta = _cfg_get("dw_beta", 0.0)          # 0 = 关闭，论文默认 1
if beta > 0:
    # 教师熵（topK 近似，见下文说明）；教师侧本就 stopgrad，权重再 detach
    H = -(teacher_p * teacher_logp).sum(-1)
    w_tilde = torch.exp(-beta * H)
    w_mean = verl_F.masked_mean(w_tilde, loss_mask).clamp(min=1e-8)
    per_token_loss = per_token_loss * (w_tilde / w_mean).detach()
```

**两个已知近似，需在代码注释与论文中说明**：

1. **topK 近似**：`distillation_topk=100`（与论文一致），故 $H$ 只在 top-100 上计算，是真实熵的下界。论文同为 topK 设置，可接受
2. **归一化范围**：论文分母是 $\Omega_{sdpo}$（全部有效 SDPO token）。分布式下严格实现需跨 DP rank 规约；本方案先在 **micro-batch 内**归一化。记录 `srpo/dw_weight_std` 观察方差，若过大再改为跨 rank

新增配置项 `self_distillation.dw_beta`（默认 0，向后兼容），新增指标 `srpo/teacher_entropy_mean`、`srpo/dw_weight_std`。

> `teacher_entropy_mean` 有独立价值：可验证论文 Fig 1(c)"教师熵随训练上升"是否在我们环境复现。这是论文核心诊断，值得独立确认。

### 改动 C：对齐论文超参

| 项 | v6 | v7 | 理由 |
|---|---|---|---|
| `kl_loss_coef` | 0.001 | **0.0** | 论文 Table 3 |
| `entropy_coeff` | 0.001 | **0** | 论文未使用 |
| `max_response_length` | 4096 | **8192** | 论文 Table 3 |
| `dw_beta` | — | **1** | 论文 §3.2 默认 |

**风险须知**：B、C 是我当初为对抗 v5 熵坍缩而加的。移除后熵坍缩风险回归。但 v5 的坍缩发生在 step 95 之后，且当时长度膨胀到 4045；v7 的机制组合不同，需实测。**监控 `actor/entropy`，若跌破 0.01 且 val 连续 3 点下滑则中止**。

### 改动 D：首 token 分布多样性探针（每 100 步）

**为什么这是本方法最关键的验证**：整套方法（v1 起）的初衷是"**让模型学会更丰富的首 token 分布**"。此前只测过训练**前**的分布，从未测过训练**后**是否真的变丰富。这个探针把论证闭环——它直接回答"强制的候选 token 有没有被内化"。

**实现方式（近乎零成本）**：复用**验证阶段已有的 rollout**，不额外生成。验证时每题采 16 条、共 210×16 = **3360 条样本**，正好是充足的经验分布样本。

位置：`ray_trainer.py::_validate()`，在拿到 `test_output_gen_batch` 后解码首个有意义 token（跳过格式脚手架，复用 `collect_first_meaningful_tokens.py::is_structural` 的判定逻辑），跨全部验证批次累积后统计一次。

**记录指标**：

| 指标 | 含义 | 训练前参考值 |
|---|---|---|
| `first_token/unique` | 出现过的不同 token 数 | 探针 120 样本中仅 **2 种**（To/The）|
| `first_token/entropy` | 经验分布香农熵（nats）| 真实分布熵 **0.437** |
| `first_token/top1_frac` | 最高频 token 占比 | top-1 概率 **0.804** |
| `first_token/top5_frac` | 前 5 名累计占比 | rank1-5 累计 ≈ **0.9995** |
| **`first_token/pool_frac`** | **落在候选池内的比例**（We/Calcul/Determin/Analy/This/1）| 训练前这 6 个合计 ≈ **1.4e-2** |

`pool_frac` 是核心指标：**若救援机制按设计生效，模型应开始自发产出它被强制使用过的那些候选 token**，该值应显著上升。

另外每 100 步把完整计数 dump 到 `outputs/<exp>/first_token_dist_step{N}.json`，供事后画分布演化图（论文配图素材）。

**实施中发现并修掉的一个隐蔽 bug**：Qwen3 的 `<think>` 是 *added token* 而非普通 special token，`decode(skip_special_tokens=True)` **不会**把它剥掉，会原样返回字符串 `"<think>"`。而 `is_structural` 判定含字母的串为"有意义"，于是 `<think>` 会被当成每条样本的首个有意义 token —— 整个分布退化成单点，`pool_frac` 恒为 0，而且**不会报错**，只会静默给出"方法无效"的错误结论。修法是显式跳过 `all_special_ids ∪ get_added_vocab()`。已用构造样例回归：`<think>\nThis is...` 现在正确返回 `This`。

**须声明的口径差异**：3360 条样本的**经验熵**与训练前测的**真实分布熵**（0.437）不是同一个量，不可直接比较绝对值；但它在各步之间口径一致，**用于看趋势是有效的**。若需与 0.437 严格可比，需在训练后用 `scripts/measure_first_token_distribution.py` 对最终权重重测一次——这要求把 `trainer.save_freq` 从 v6 的 `0` 改为 `100`（否则跑完没有权重可测）。

## 3. 运行预算：10 小时 wall-clock（对齐论文报数协议）

论文 Table 1 按 **1h / 5h / 10h wall-clock 预算**报"该预算内的最佳 avg@16"，而非固定步数。为使数字可直接对比，v7 采用同一协议：

| | 设置 |
|---|---|
| `total_training_steps` | 450（上限，保证 10h 内不会提前结束）|
| 实际跑 | **满 10 小时**，观察方法上限 |
| 预计步数 | 10h ÷ ~110s ≈ **330 步** |

**1h / 5h / 10h 三档的提取方式**：日志中 `timing_s/step` 逐步累加即得 wall-clock 偏移，据此定位三档对应的 step，取该预算内的最佳 val。命令：

```bash
grep -oE "timing_s/step:[0-9.]+" <log> | sed 's/.*://' | \
  awk '{s+=$1; printf "step=%d cum_h=%.2f\n", NR, s/3600}'
```

> 注意：论文的 10h 是在其自身实现上测的（其 SRPO 报 75.8-91.5s/步）。我们的 wall-clock 含 vLLM 启动与验证开销，口径不完全一致；**故同时报"按步数对齐"的曲线**，两种口径都给出，避免单一口径引起争议。

## 4. 实验矩阵

两次运行**仅差 `token_roll.enable` 一个开关**：

| # | 配置 | 目的 |
|---|---|---|
| **①** | v7，`token_roll.enable=False` | 忠实 SRPO 复现，目标 ~83.0，验证管线可信 |
| **②** | v7，`token_roll.enable=True` | **②−① = 全错组救援的净贡献** |

判读标准：

- ① 到不了 78-83：说明仍有未识别的偏差，**不要继续 ②**，先排查
- ② > ①：方法成立，差值即贡献
- ② ≈ ①：救回样本对最终精度贡献有限 → 方法定位需转向**样本效率**（可用"达到某 val 所需步数"作为指标），而非最终精度

## 5. 成本

`max_response_length` 回到 8192 后，长度不再坍缩（预期回到基线的 1400+ 量级），单步约 **100-130s**（对照：GRPO 基线实测 105-133s；v6 因 20 token 仅 20-30s）。

| | 单步 | 450 步 |
|---|---|---|
| ① | ~110s | **~14 小时** |
| ② | ~120s（含救援开销）| **~15 小时** |

两台 8 卡机可并行，**约 15 小时出双结果**。

## 6. 改动清单（待批准后执行）

| 文件 | 改动 |
|---|---|
| `verl/workers/actor/dp_actor.py` | srpo 分支改为 λ 加权求和；新增 λ 指标 |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` 内实现 DW；新增熵/权重指标 |
| `verl/trainer/ppo/ray_trainer.py` | `_validate()` 内加首 token 分布统计（改动 D）|
| `verl/workers/config/actor.py` | `SelfDistillationConfig` 新增 `dw_beta`（默认 0） |
| `verl/trainer/config/actor/actor.yaml` | 同步 `dw_beta` 默认值 |
| `verl/trainer/config/srpo_rescue.yaml`（→ 复制为 `srpo_v7.yaml`）| `dw_beta=1`、`kl_loss_coef=0.0`、`entropy_coeff=0`、`max_response_length=8192` |
| `run_local_srpo_rescue.sh`（→ 复制为 `run_local_srpo_v7.sh`）| 上述超参；`save_freq=100`；`token_roll.enable` 提为命令行可切换 |

**不改动**：救援逻辑（`_maybe_rescue_all_fail_groups`）、`ForcedFirstTokenAgentLoop`、候选池 —— 保持与 v6 一致以便对比。

## 7. 不在本次范围内

- **DAPO dynamic sampling 对照（③）**：全错组"丢弃+重采样"是既有方案（Yu et al. 2025，论文自身引用），论文投稿需要此基线，但不阻塞 ①②
- 其余四个基准（Physics / Biology / Materials / Tool Use）
- 跨 DP rank 的 DW 严格归一化

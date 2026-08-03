# feature_v7：忠实 SRPO + 全错组救援

> 本文件是 v7 实现的完整记忆，供后续会话或协作者快速接手。
> 详细设计见 `doc/v7_design.md`，DW 详解见 `doc/entropy_aware_dynamic_weighting.md`。

## 0. 一句话目标

复现论文 SRPO（~83.0），再用单一开关 `RESCUE` 隔离出"全错组救援"的净贡献。
基准：SciKnowEval Chemistry，Qwen3-8B，8 卡 H20。

## 1. 方法起源

| 情况 | 论文 SRPO 的处理 | 问题 |
|---|---|---|
| 8 条里有对的 | 对的→GRPO，错的→SDPO（对的兄弟当 teacher）| 正常 |
| 错了但有对的兄弟 | SDPO logit 蒸馏 | 正常 |
| **整组全错** | 无 teacher → SDPO 不接；GRPO 优势 = 0-0 = 0 | **盲区：不产生梯度** |

v5 实测：全错组在训练早期占 **25-28%** 的 prompt，这些 prompt 被完全浪费。

**本方法**：全错组时从候选池抽 token 强制作为首个有意义 token，重新生成 rollout，首次成功即停。期望把这些死样本救活，产生非零优势。

## 2. v6 设计（`srpo_rescue.yaml`，`loss_mode: srpo`）

三分支路由：

| 情况 | 路由 | 机制 |
|---|---|---|
| 对的 rollout | GRPO | PPO-clip，组内优势 |
| 错但有对的兄弟 | SDPO | logit 级 KL 蒸馏，teacher = 对的兄弟 |
| 全错组 | 救援 | 强制首 token，重生成 6 条（留 2 基线），首次成功即停 |

损失组合（`dp_actor.py`）：

```python
pg_loss = grpo_loss + sd_loss   # 直接相加
```

超参：`kl_loss_coef=0.001`、`entropy_coeff=0.001`、`max_response_length=4096`，无 DW。

关键组件：
- `ForcedFirstTokenAgentLoop`（`forced_first_token_agent_loop.py`）：异步 agent loop 会从 `raw_prompt` 重建 prompt_ids 而忽略 input_ids 张量，所以强制 token 必须在 agent loop 里注入。
- `_maybe_rescue_all_fail_groups()`：在 `fit()` 里原地修改 batch 和 reward_tensor，包括拼接 forced_reward 和 rollout_log_probs（否则 `pg_loss: nan`）。

## 3. v6 结果

- 峰值 **81.1** @ step 260-265（GRPO 基线 71.6 @ step 130）
- **末段崩溃**：step 370 后从 79.2 暴跌到 34.6@395。step 350 时 `actor/entropy` 已 ~0.01、`response_length` 坍缩到 20 token。
- 即使有 KL 锚 + 熵奖励也没拦住崩溃。
- 完整曲线：0→41 → 75@50 → 81@260 → 崩溃 → 42@405

## 4. v6 两处偏差（相对论文）

### 偏差 1：分支归一化

论文 §3.3 要**单一 union 分母**：

$$\mathcal{L} = \frac{\sum z^G \ell^G + \sum z^S \ell^S}{\sum z^G + \sum z^S}$$

每个分支按它覆盖的 token 数**成比例**贡献——SDPO 占比从 40% 降到 7% 时权重同步降到 7%（**自衰减**）。

v6 直接相加，各项除以**自身** token 数：

$$\mathcal{L}_{v6} = \frac{\sum z^G \ell^G}{N_G} + \frac{\sum z^S \ell^S}{N_S}$$

SDPO 占比降到 7% 时仍给 100% 权重 → 放大 **~10×**，自衰减失效。

### 偏差 2：缺 entropy-aware 动态加权

论文 §3.2：每个 SDPO token 乘 $w = \exp(-\beta H_{teacher})$，归一化到均值 1。教师确信的权重大，不确定的小。v6 完全没实现。

## 5. v7 四处改动

### 改动 A：λ 加权 = union 归一化（`dp_actor.py`）

```python
g_cnt = grpo_response_mask.sum()
s_cnt = sd_response_mask.sum()
tot   = (g_cnt + s_cnt).clamp(min=1.0)
pg_loss = (g_cnt/tot) * grpo_loss + (s_cnt/tot) * sd_loss
```

数学等价于单一 union 分母：

$$\frac{N_G}{N_{tot}} \cdot \frac{\sum z^G \ell^G}{N_G} + \frac{N_S}{N_{tot}} \cdot \frac{\sum z^S \ell^S}{N_S} = \frac{\sum z^G \ell^G + \sum z^S \ell^S}{N_{tot}}$$

**前提已核实**：FSDP 的 `dp_actor` 从不填 `global_batch_info`（只有 megatron 的 `workers/utils/losses.py` 填），两分支都退化为 `dp_size=1`、`batch_num_tokens = loss_mask.sum()`。**数值验证** `atol=1e-6` 通过；同例旧法把 SDPO 放大 2.1×（占比越低放大越多，7% 时约 10×）。

新增指标：`srpo/lambda_grpo`、`srpo/lambda_sdpo`（验证自衰减）。

### 改动 B：实现 DW（`core_algos.py::compute_self_distillation_loss`）

```python
H_teacher = -(teacher_p * teacher_logp).sum(-1)      # topK+tail 近似，真实熵下界
w = exp(-dw_beta * H_teacher)
w = (w / masked_mean(w)).detach()                    # 归一化均值 1
per_token_loss = per_token_loss * w
```

- `dw_beta=0` 关闭（向后兼容），`=1` 对齐论文
- 两个已知近似：topK 熵下界（论文同设置）；micro-batch 内归一化（论文是全 SDPO token，跨 rank 需 all-reduce，先记录 `dw_weight_std` 看方差）
- 新增 `srpo/teacher_entropy_mean`（验证论文 Fig 1(c) 教师熵随训练上升）、`srpo/dw_weight_std`
- 指标键**无条件发射**：`reduce_metrics` 要求每个 micro-batch 的键集相同

### 改动 C：对齐论文超参（`srpo_v7.yaml`）

| 项 | v6 | v7 |
|---|---|---|
| `kl_loss_coef` | 0.001 | **0.0** |
| `entropy_coeff` | 0.001 | **0** |
| `max_response_length` | 4096 | **8192** |
| `dw_beta` | — | **1** |
| `save_freq` | 0 | **100**（事后重测首 token 分布需要 checkpoint） |

### 改动 D：首 token 分布探针（`ray_trainer.py::_validate`）

每 100 步统计验证集 210×16=3360 条 rollout 的首个有意义 token 经验分布。**这是方法初衷的第一次真正验证**——此前只测过训练前的分布，从未测训练后是否变丰富。

| 指标 | 训练前参考 |
|---|---|
| `first_token/unique` | 探针样本仅 2 种（To/The）|
| `first_token/entropy` | 真实分布 0.437 nats |
| `first_token/top1_frac` | 0.804 |
| `first_token/pool_frac` | **6 候选合计 1.4e-2**（核心：应显著上升）|

每 100 步 dump 完整计数到 `outputs/<exp>/first_token_dist_step{N}.json`。

**修掉的隐蔽 bug**：Qwen3 的 `think_open` 是 added token，`decode(skip_special_tokens=True)` 不剥，会原样返回字符串。`is_structural` 判定含字母的串为"有意义"，于是它会被当成每条样本的首 token —— 整个分布退化成单点，`pool_frac` 恒为 0，且**不报错**，只会静默给出"方法无效"的错误结论。修法：显式跳过 `all_special_ids ∪ get_added_vocab()`。

## 6. 候选池

`datasets/first_token_candidates_chemistry.json`——训练前用 `scripts/measure_first_token_distribution.py`（transformers 单 forward）测的首 token 分布，再由 `scripts/build_candidate_pool.py` 构建。

测量：To+The 占 99%，熵 0.437 nats。跳 top-2，取 rank 3-8：

| token | id | 训练前概率 | NLL（lift 成本）|
|---|---|---|---|
| We | 1654 | 8.2e-3 | 4.8 |
| Calcul | 57908 | 2.5e-3 | 6.0 |
| Determin | 92648 | 1.6e-3 | 6.4 |
| Analy | 73307 | 5.5e-4 | 7.5 |
| This | 1986 | 5.2e-4 | 7.6 |
| 1 | 16 | 3.9e-4 | 7.8 |

选 rank 3-8：仍是语义合法的首 token（推理题常见开头），但概率低，强制代价仅 1.8-4.8 nats（MATH 题同类 token 要 27 nats）。

## 7. 实验矩阵

两次运行**仅差 `RESCUE` 一个开关**：

| # | 配置 | 目的 |
|---|---|---|
| ① | `RESCUE=False` | 忠实 SRPO 复现，目标 ~83.0，验证管线可信 |
| ② | `RESCUE=True` | **②−① = 全错组救援的净贡献** |

判读：
- ① 到不了 78-83 → 还有未识别偏差，**不要继续 ②**，先排查
- ② > ① → 方法成立，差值即贡献
- ② ≈ ① → 救回样本对最终精度贡献有限 → 转向"样本效率"指标

满 10h wall-clock（对齐论文 Table 1 的 1h/5h/10h 报数协议），`total_training_steps=450` 作上限，预计 ~330 步。

## 8. 已验证

- **λ 加权 = union 归一**：数值 `atol=1e-6`
- **DW on/off**：loss 变化、梯度流通、非路由样本梯度=0、空 mask 不 NaN、指标键集稳定
- **首 token 探针**：正确跳过脚手架 / 控制符 / 全 padding；`pool_frac` 计算正确
- **Hydra 配置解析 + dataclass 实例化**（v6 崩过这步）
- **三方配置一致性**（`SelfDistillationConfig` / `actor.yaml` / `srpo_v7.yaml`）

## 9. 风险

- **v6 末段崩溃**（熵→0.01，长度→20）。v7 去掉 KL+熵奖励，崩溃风险**更高**。
- 监控 `actor/entropy`：跌破 0.01 且 val 连续 3 点下滑 → 中止
- `max_response_length` 回到 8192，单步 ~110-130s（v6 因 20 token 仅 20-30s）

## 10. 文件清单

| 文件 | 改动 |
|---|---|
| `verl/workers/actor/dp_actor.py` | A：λ 加权 + λ 指标 |
| `verl/trainer/ppo/core_algos.py` | B：DW + 熵/权重指标；删 `compute_token_roll_loss` |
| `verl/trainer/ppo/ray_trainer.py` | D：首 token 探针 + `_probe_control_token_ids` |
| `verl/workers/config/actor.py` | B：`SelfDistillationConfig.dw_beta` |
| `verl/trainer/config/actor/actor.yaml` | B：`dw_beta` 默认 0 |
| `verl/trainer/config/srpo_v7.yaml` | C：新配置 |
| `sdpo/SDPO/run_local_srpo_v7.sh` | C：启动脚本，`RESCUE` 可切换 |

**不改动**：`_maybe_rescue_all_fail_groups`、`ForcedFirstTokenAgentLoop`、候选池——保持与 v6 一致以便对比。

## 11. 启动

```bash
# ① 忠实 SRPO 复现
RESCUE=False ./sdpo/SDPO/run_local_srpo_v7.sh

# ② SRPO + 全错组救援
RESCUE=True  ./sdpo/SDPO/run_local_srpo_v7.sh
```

10h wall-clock 提取 1h/5h/10h 三档：

```bash
grep -oE "timing_s/step:[0-9.]+" <log> | sed 's/.*://' | \
  awk '{s+=$1; printf "step=%d cum_h=%.2f\n", NR, s/3600}'
```

## 12. 相关文档

- `doc/v7_design.md` — 完整设计 + 数学推导 + 成本
- `doc/entropy_aware_dynamic_weighting.md` — DW 三步公式 + Table 2 消融
- `doc/srpo.md` — 论文全文
- `doc/grpo_router.md` — 路由框架

## 13. 已跑实验汇总

| 实验 | 配置 | 最后一步 | 峰值 | 状态 |
|---|---|---|---|---|
| v6 | SRPO+救援，非忠实 | 407/450 | 81.1 @ 260 | 末段崩溃到 34.6 |
| GRPO 基线 | 论文超参 | 130/450 | 71.6 @ 130 | 被中断 |

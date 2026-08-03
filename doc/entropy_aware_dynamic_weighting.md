# Entropy-Aware 动态加权（DW-SDPO）说明

> 论文 §3.2 的机制，SRPO 相对纯 SDPO 的核心贡献之一。本文档解释它是什么、为什么需要、以及对我们实现的影响。
>
> 来源：`doc/srpo.md`（§3.2、§3.3、Fig 1c、Fig 4a、Table 2、Table 3）

## 1. 一句话

**按教师的"确信程度"给每个 token 的蒸馏损失打折：教师含糊的位置少学，教师确定的位置多学。**

## 2. 它要解决什么问题

SDPO 的教师是"看过正确答案的自己"（上下文里塞入同组一条正确 rollout），它逐 token 给出目标分布，学生用散度去对齐。

问题在于：**教师在不同位置的可靠程度差别很大**。

论文 Fig 1(c) 观测到关键现象——**教师的 token 级熵随训练上升**。原因是教师是学生的 EMA，训练久了两者越来越像，教师的 privileged context 优势被抹平，它自己也开始拿不准。此时它给出的目标从"有用的纠正"退化为"噪声"。

论文将此诊断为 SDPO 后期崩溃的两大主因之一：

| 主因 | 论文的解法 |
|---|---|
| 对已正确样本做自蒸馏引入优化歧义 | **样本路由**（只让错误样本走 SDPO）|
| **教师信号可靠性随训练退化** | **entropy-aware 动态加权** |

DW 专治第二个。

## 3. 三步公式

### 第一步：量化教师在每个位置的不确定性

$$H_{i,t} = -\sum_{v \in \mathcal{V}} q_{i,t}(v)\log q_{i,t}(v), \qquad q_{i,t}(v) = \pi_\theta(v \mid x, f_i, y_{i,<t})$$

- $q$ 是**教师**分布（条件里带 $f_i$ = 正确兄弟解答）
- $H$ 小 = 教师几乎确定下一个 token 是什么
- $H$ 大 = 教师在多个候选间摇摆

### 第二步：熵转权重（指数衰减）

$$\tilde{w}_{i,t} = \exp(-\beta H_{i,t}), \qquad \beta = 1 \text{（论文默认）}$$

数量感：

| 教师熵 $H$ | $\tilde w = e^{-H}$ | 含义 |
|---|---|---|
| 0.1 | 0.90 | 教师很确定 → 几乎全额学习 |
| 0.5 | 0.61 | |
| 1.0 | 0.37 | |
| 2.0 | **0.14** | 教师很含糊 → 损失压到约 1/7 |

$\beta$ 控制惩罚陡峭程度（$\beta$ 越大，对高熵越严厉）。

### 第三步：归一化以保持总损失尺度

$$w_{i,t} = \frac{\tilde{w}_{i,t}}{\frac{1}{|\Omega_{sdpo}|}\sum_{(j,s)\in\Omega_{sdpo}}\tilde{w}_{j,s}}$$

分母是所有 SDPO token 上 $\tilde w$ 的**均值**，因此 $w$ 的均值恒为 1 —— **只在 token 间重新分配权重，不改变整体量级**。

论文强调："does not alter the functional form of SDPO; it only modulates each token's contribution according to teacher confidence."

最终：

$$\ell^{DW\text{-}SDPO}_{i,t} = w_{i,t}\cdot\ell^{SDPO}_{i,t}$$

## 4. 效果（论文 Table 2 消融，五基准平均 avg@16）

| | 1h | 5h | 10h |
|---|---|---|---|
| SRPO | 66.9 | 75.5 | **77.4** |
| SRPO w/o DW | 66.5 | 74.8 | 75.6 |
| **增益** | +0.4 | +0.7 | **+1.8** |

**增益随训练放大**，这恰好验证其动机：教师后期才变噪声，早期加权收益有限、后期才关键。论文称它让 SRPO 能 "continue improving beyond the point where pure GRPO plateaus"。

## 5. 与损失归一化的区别（重要，别混淆）

论文里有**两个**独立的加权机制，作用层级不同：

| 机制 | 层级 | 公式 | 作用 |
|---|---|---|---|
| **DW（§3.2）** | **token 级** | $\exp(-\beta H_{i,t})$ | 同一分支内，按教师置信度重分配 |
| **union 归一化（§3.3）** | **分支级** | $\dfrac{\sum z^{G}\ell^{G} + \sum z^{S}\ell^{DW\text{-}S}}{\sum z^{G} + \sum z^{S}}$ | 让每个分支"按其覆盖的 token 数成比例"贡献 |

§3.3 那个分母是**两分支 token 的并集**，论文明确其意图是 "each branch contributes in proportion to the tokens it covers"，并说明这带来自适应：早期失败多 → SDPO 覆盖多 → 权重大；后期正确率上升 → GRPO 主导。

## 6. 对我们实现的影响

我们**两个都没做对**：

| 项 | 论文 | 我们 v6 | 后果 |
|---|---|---|---|
| DW | $\beta=1$ | **未实现** | 教师后期噪声目标被全额吸收 |
| 分支归一化 | union 分母 | `grpo_loss + sd_loss`，**各自按自身 token 数取均值后相加** | SDPO 占比降到 7% 时权重仍为满额 → **压力放大约 10 倍**，§3.3 的自衰减机制被破坏 |

### 一个反直觉的推论：DW 未必能治长度坍缩

教师看过答案后，在"该收尾了"这个位置是**低熵**的（很确定要输出 `</reasoning>`）→ DW 给它**高**权重 → 理论上**强化**缩短倾向。

所以论文 SRPO 长度能保持 moderate（Fig 4a），主要功劳大概率不在 DW，而在 **§3.3 的 union 归一化让 SDPO 权重随占比自然衰减**（40% → 7%）。而我们那个归一化 bug 恰好破坏了这个自衰减。

**优先级结论**：

- **修归一化 bug → 治长度坍缩**
- **加 DW → 治后期教师噪声**

两者对症不同、都要做；若只能做一个，**先修归一化**。

## 7. 实现要点（供 v7 参考）

1. 教师熵需要教师的**完整分布**。我们当前用 `distillation_topk=100`（与论文一致），所以 $H$ 只能在 top-100 上算 —— 这是对真实熵的**下界近似**，可接受（论文也是 topK 设置），但需在代码注释中说明
2. 归一化的分母是**当前 micro-batch 内所有 SDPO token** 的 $\tilde w$ 均值。分布式训练下严格实现应跨 DP rank 归约；简化为 micro-batch 内归一化会引入偏差，需权衡（建议先做 micro-batch 内，并记录指标观察方差）
3. 需要新增指标：`srpo/teacher_entropy_mean`（验证 Fig 1c 的上升现象是否在我们环境复现）、`srpo/dw_weight_std`（观察加权是否真的在起作用）

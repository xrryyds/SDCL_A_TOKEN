# FILL 的 baseline 选型调研

调研目标：为 FILL（首 token 强制多样化，抢救 all-fail 组）找一个**有开源代码 + 明确被 CCF-A 会议收录**的
承载 baseline，使 "baseline + FILL" 的增益可复现、可比较。

调研日期：2026-08-24。所有结论均已在本地代码库或公开页面上核实，未核实的地方明确标注。

---

## 0. 为什么 SRPO 不能继续做 baseline

- **无开源代码。** SRPO（arXiv 2604.02288）只放了论文，没有 repo。我们目前的 SRPO 是自己按论文
  §4.1/§4.2 重写的。
- **复现有系统性缺口。** 在 SciKnowEval/Biology 上，我们的复现比论文低 5–7 分且早期饱和：

  | 运行 | 1h | 5h | 10h | peak | 末尾长度 | 末尾 VALID% |
  |---|---|---|---|---|---|---|
  | 我们的 SRPO（lr 5e-6, mb 32, k-gate） | 53.4 | 58.6 | — | 58.6 | 279 | 61.2 |
  | 我们的 GRPO（lr 1e-6, mb 8, 840 步跑完） | 41.6 | 63.5 | 59.5 | **63.5** | 876 | 29.8 |
  | 论文 GRPO | 46.9 | 68.1 | 70.6 | — | — | — |
  | 论文 SRPO | 55.8 | 68.3 | 72.8 | — | — | — |

  论文的**定性结论我们复现出来了**（SRPO 早期更快、GRPO 后期反超），但绝对值对不上。
  在一个自己都没对齐的 baseline 上叠加 FILL，涨点无法归因。

---

## 1. 硬性筛选条件

FILL 要能挂上去，baseline 必须同时满足：

1. **组内采样**（一个 prompt 采 n 个 rollout）—— FILL 的作用对象是"整组全错"
2. **可验证奖励**（rule-based / verifier）—— 需要判定 forced-token rollout 是否救活
3. **开源代码**，最好基于 verl（我们整套代码在 verl 上）
4. **明确处理或明确回避 all-fail 组** —— 这样 FILL 才有可对比的直接对手
5. **CCF-A 收录**

CCF-A（人工智能方向）名单：**AAAI、NeurIPS、ACL、CVPR、ICCV、ICML、IJCAI**。
注意：**ICLR 不在 CCF 名单里**，COLM 也不在。这直接排掉了几个热门候选。

---

## 2. 唯一满足全部条件（除年份）的候选：DAPO

| 项 | 内容 |
|---|---|
| 论文 | *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* |
| arXiv | https://arxiv.org/abs/2503.14476 |
| 收录 | **NeurIPS 2025**（CCF-A）proceedings 已上线 |
| 代码 | https://github.com/BytedTsinghua-SIA/DAPO （约 1.9k★，**基于 verl**） |
| 与 FILL 的关系 | **正面相关：DAPO 的 Dynamic Sampling 就是在处理 all-fail / all-correct 组** |

**年份不符**：用户要求 2026，DAPO 是 NeurIPS **2025**。这是本次调研没能满足的唯一一条。
但它是唯一"开源 + verl + 组采样 + 直面 all-fail"四项全中的工作。

### DAPO 与 FILL 的对比故事（这是最锋利的一点）

DAPO 的 Dynamic Sampling 面对全对/全错组的做法是：**过采样，然后把这些组整个丢掉**，
用 `max_num_gen_batches` 反复重采直到凑满一个 batch。也就是说：

- DAPO：全错组 = 噪声 → **丢弃**，代价是重复生成，step 时间被拉长
- FILL：全错组 = 未开发的梯度 → **抢救**，用强制首 token 换一条新推理路径

可以直接量化的三个指标：

1. **准确率**（主指标）
2. **gen-batches per step**（DAPO 因为丢组会 >1，FILL 应保持 1）
3. **有效梯度 prompt 占比**（DAPO 丢掉的组贡献 0，FILL 把其中一部分救回来）

FILL 的动机在这里也能立住：Qwen3-8B 在 MATH 上首 token 熵只有 **0.029 nats**（"We" 占 97.8%），
所以"全错"往往是**开局被锁死**，而不是模型没能力。换首 token 就是换开局。

### 本地状态核实

- `verl/trainer/config/algorithm.py:43` 有 `FilterGroupsConfig`（含 `max_num_gen_batches: int = 0`），
  并在 `AlgoConfig.filter_groups`（line 611）暴露。
- **但 `verl/trainer/ppo/ray_trainer.py` 里完全没有引用 `filter_groups`**，`recipe/` 目录是空的。

⇒ **DAPO 的 dynamic sampling 逻辑本地没有实现**，只有一个悬空的配置类。要么从上游 verl 拉，
要么自己写（在 `fit()` 的采样循环里加过采样 + 丢组 + 重采）。这是上 DAPO 路线的主要工作量。

---

### 实测：DAPO 在 chemistry 上的真实开销（chem-dapo1，2026-08-24）

移植完成后实测（`train_batch_size=32`, `rollout.n=8`, SciKnowEval chemistry）：

| 指标 | 实测值 |
|---|---|
| 每个生成批次存活组数 | 19–24 / 32 |
| `dapo/dropped_group_frac` | 0.28–0.38 |
| `dapo/num_gen_batches` | **恒定 2** |
| 生成开销 | **2×** |

**一个必须诚实面对的结论**：这 28–38% 的同质组里绝大部分是**全对组**，不是全错组。
chemistry 训练 reward 后期达到 0.889，8 连对的概率约 `0.889^8 ≈ 0.39`；而全错组只占 5%（早期 13%）。

FILL 只能救全错组。即使把全错组全部救活，每批存活也只从 ~23 升到 ~27，仍然 < 32 ——
**`num_gen_batches` 不会从 2 降到 1**。

⇒ **"FILL 帮 DAPO 省算力"这条叙事在 chemistry 上立不住。** 能改善的是 `dropped_group_frac`
（28% → ~24%）和准确率。论文的卖点必须是准确率，把效率当次要证据，否则会被审稿人直接击穿。

（在 biology 上全错组占 16.5%，省算力的故事相对成立一些，但 biology 只有 50 题验证集。）

## 3. 直接竞争者（必须知道它的存在）

| 项 | 内容 |
|---|---|
| 论文 | *Advantage Collapse in Group Relative Policy Optimization* |
| 方法 | **AVSPO** —— 给同质组（全对/全错）注入**虚拟奖励样本**，让优势不再为 0 |
| 收录 | **ICML 2026**（CCF-A）|
| 代码 | https://github.com/hexixiang/Advantage-Collapse-Rate —— **未放出代码（0★）** |

**这是和 FILL 同一个问题域的工作**，而且年份正好是用户要的 2026。好消息是它没开源，
先发空间还在。

FILL 相对 AVSPO 的差异化论点：

- AVSPO 注入的是**虚拟奖励**，不产生新轨迹 —— 模型没有见过任何新的解法
- FILL 做的是**真实重采样 + 首 token 多样化**，会真的生成新的推理路径并从成功者身上学习
- 支撑证据就是那个 0.029 nats 的首 token 熵测量：问题不在奖励信号，在探索被首 token 锁死

如果要写论文，AVSPO 应该作为 related work 里必须正面比较的对象。

---

## 4. 被排除的候选及原因

| 方法 | 排除原因 |
|---|---|
| **GMPO** | ICLR 2026 —— **ICLR 不在 CCF 名单** |
| **GSPO** | 仅 arXiv，无会议收录 |
| **CISPO** (MiniMax-M1) | 仅技术报告 |
| **SAPO** | 仅 arXiv |
| **REINFORCE++** | 仅 arXiv |
| **VAPO** | 仅 arXiv |
| **Dr.GRPO** | 收录情况未核实，且不基于 verl |

---

## 5. 本地已有的零成本正交性表格

我们的 verl 里已经注册了 8 个 policy loss（`verl/trainer/ppo/core_algos.py`）：

```
vanilla (1421)  gspo (1515)  sapo (1591)  gpg (1676)  clip_cov (1712)
kl_cov (1817)   geo_mean (1978)  cispo (2064)  bypass_mode (2407)
```

外加 `examples/` 下的 `reinforce_plus_plus, remax, rloo, otb, prefix_grouper,
rollout_correction, gmpo_trainer, cispo_trainer, gspo_trainer, sapo_trainer, gpg_trainer`。

**这些方法一个都不处理 all-fail 组** —— 它们改的全是 importance ratio / clipping 的形状。
所以 "X" vs "X + FILL" 在这些方法上都应该涨，而且改一个 config 就能跑。

这张正交性表是**对抗 AVSPO 竞争的最好防守**：证明 FILL 不是某一个算法的补丁，
而是一个和 policy-loss 设计正交的、可叠加的增益。

---

## 6. 建议路线

1. **主线：DAPO + FILL**（接受 NeurIPS 2025 的年份），先把 dynamic sampling 实现出来，
   跑 DAPO / DAPO+FILL 两条线，报准确率 + gen-batches/step + 有效梯度占比
2. **正交性表**：在本地已有的 `gspo / geo_mean / cispo / sapo` 上各跑一对 ±FILL，成本极低
3. **Related work**：正面对比 AVSPO（ICML 2026），论点是"真实轨迹 vs 虚拟奖励"

如果年份是硬约束，那么当前没有满足全部条件的 baseline —— 需要重新定义可接受范围
（例如接受 ICLR 2026 的 GMPO，或接受 NeurIPS 2025 的 DAPO）。

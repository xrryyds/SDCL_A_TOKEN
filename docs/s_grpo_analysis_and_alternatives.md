# S-GRPO 算法分析与可借鉴方向

> **背景**:截至 2026-05-30,所有历史实验作废,思路重置。当前 baseline 在 MATH train 7496 题上 acc=72.92%(mistake=2030,corr=5466,DeepSeek-R1-Distill-Qwen-7B,2048+4096,greedy 单次)。
>
> 这篇文档:(1) 完整解构 S-GRPO 的算法和它和 vanilla GRPO 的区别 (2) 抽出我们这边可借鉴的"机制"而非"方法名" (3) 列出 2025 同期 GRPO 圈其他可借鉴的工作。

---

## 1. S-GRPO 全名与一句话定位

**Serial-Group Decaying-Reward Policy Optimization** (arXiv 2505.07686v2, 2025-05)。

它**不是**一个通用的 RL 算法替代品 —— 是一个**专攻"long-CoT 模型 overthinking"**的 GRPO 变体。它的目标不是把 acc 拉得更高,而是**在保持/微涨 acc 的前提下,把生成长度砍掉 35%–61%**。

S-GRPO 在 R1-Distill-Qwen-7B 上 MATH-500: vanilla 85.8 / 5590 tok → S-GRPO 92.4 / 2252 tok(+6.6pp / -60% tok)。它的卖点首先是 **token 效率**,其次是 acc。

---

## 2. 算法核心(从 vanilla GRPO 一步步改)

### 2.1 Vanilla GRPO 的 group 是怎么构造的

对每个 prompt:
1. 用 π_θ_old 并行 rollout `n` 条完整路径 `o_1, ..., o_n`
2. 给每条算 outcome reward `r_i`
3. group baseline = mean(r),advantage `A_i = (r_i − mean) / std`
4. PPO clip + token-level广播 advantage 算 loss

**问题**:奖励只看最终对错,中间推理过程的"啰嗦/简洁"完全没信号。模型一旦答对,中间多绕路也照样高 reward → response 越训越长。

### 2.2 S-GRPO 的 group 怎么构造

**只 rollout 一次完整路径**,然后**在这条路径上沿时间轴抽 m 个位置提前终止 thinking**,共得到 `m+1` 个候选答案(原始路径 + m 个早退路径):

```
完整 rollout:  T_1 T_2 ... T_n </think> C_0          ← 原版回答
位置 P_1 早退: T_1 ... T_{P_1} </think> C_1          ← 强行 inject "</think>" 让模型直接答
位置 P_2 早退: T_1 ... T_{P_2} </think> C_2
...
位置 P_m 早退: T_1 ... T_{P_m} </think> C_m
```

`P_i ~ Uniform(1, n)`,**随机长度截断**。注入退出的 token 是固定字符串:
```
"Time is limited, stop thinking and start answering.\n</think>\n\n"
```

### 2.3 衰减奖励 (decaying reward)

这是 S-GRPO 的核心创新。按位置从早到晚遍历 `(P_1, P_2, ..., P_m, P_0=full)`:

```
N_right = 0
for i in (1..m, 0):
    if C_i 正确:
        N_right += 1
        r_i = 1 / 2^(N_right - 1)        # 第一个对的→1, 第二个→0.5, 第三个→0.25...
    else:
        r_i = 0                          # 错的一律 0
```

直觉:**越早能在简短 CoT 下答对,reward 越高**。错的不给"长度奖励"。

### 2.4 Advantage 与 Loss

```
A_i = r_i − mean(r)         # 注意:不除以 std (与 vanilla GRPO 不同)
A_{i,t} = A_i               # token 级广播
```

Loss 形式仍是 PPO clip:

```
J = E[ 1/G · Σ_i 1/|o_i| · Σ_t  min( ratio · A,  clip(ratio, 1-ε, 1+ε) · A ) ]
```

**没有显式 KL penalty / reference model**(论文式 2 里看不到)。

### 2.5 与 vanilla GRPO 的差异表

| 维度 | vanilla GRPO | S-GRPO |
|---|---|---|
| Group 构造 | n 条**并行**完整 rollout | 1 条完整 + m 条**串行早退** |
| Reward | 终点对错 0/1 | 早退点对错 + 指数衰减位置奖励 |
| Advantage 归一 | (r − mean)/std | r − mean(去 std) |
| 调控对象 | 仅最终结果 | 隐式调控"思考是否充分" |
| Rollout 成本 | n × 全长 | 1 × 全长 + m × 答案段(共享前缀) |
| 显式 KL | 论文典型有 β·KL | 无 |
| Length bias | 普遍越训越长 | 显式抑制(简洁正确高奖) |

### 2.6 Ablation 揭示的真实关键

论文 Table 2 (Qwen3-8B):

| 设定 | Acc | Tokens |
|---|---|---|
| **S-GRPO 完整版** | 84.26 | 4,922 |
| 去掉衰减,只奖励最短的对(Shortest-1) | 81.70 (-2.56) | 4,390 |
| 去掉衰减,所有对都给 1(All-1) | 83.00 (-1.26) | 8,179 (+66%) |
| 去掉 serial group(= vanilla GRPO) | 82.30 (-1.96) | 8,150 (+66%) |

**关键启示**:
- "All-1" 跟 vanilla GRPO 几乎一样 —— 说明**没有衰减,长度就刹不住**(All-1 token 还涨了 66%)
- "Shortest-1" 太极端 —— **只奖励最短的会过拟合短答案,acc 掉**
- **指数衰减**是 S-GRPO 的真正引擎,serial group 只是承载衰减的容器

---

## 3. 我们能借鉴什么?

⚠️ 先界定 S-GRPO 的适用边界 —— 它假设 **base model 已经是 long-CoT 模型**(R1-Distill 系列、Qwen3 thinking 模式),且 **acc 已经接近天花板**(MATH-500 85+),问题是"啰嗦"。**我们的 baseline 在 MATH train 是 72.92%**,甚至没到 long-CoT 模型该有的水平 —— 直接套 S-GRPO 不一定对症。

但可以拆出几个"机制"级别的灵感:

### 借鉴点 A:**截断式 rollout 替代并行 rollout**(强烈推荐研究)

**Why**:S-GRPO 用 1 条完整 rollout + m 段共享前缀的早退,采样成本远低于 vanilla GRPO 的 n 条独立 rollout。对**显存紧 / vLLM 单卡 colocate**的场景特别友好。

**怎么用**:即使我们不做 length penalty,也可以在每个 mistake 题上跑 1 条完整 rollout,再在不同位置截断重续,多得"答案族"做 contrastive。比纯 n-parallel 节省 60% 显存和时间。

### 借鉴点 B:**位置加权的奖励信号**(直接可移植)

**Why**:S-GRPO 的衰减是"位置 × 对错"的二维 reward,比纯 0/1 outcome reward 信息量大。它给优化器一个"在多个对答案中偏向哪一条"的方向。

**怎么用**:我们的 mistake 池上,如果一题 rolling-K 救回率 = 5/8,纯 outcome reward 给这 5 条都是 1 分,模型不知道学哪条好。可以引入**置信度加权**(student 自己 rollout 时 logprob 高的对答案给更高 reward),或者**首 token 多样性加权**(把奖励倾斜给"和 anchor 不同首 token 但答对"的路径,鼓励探索)。

### 借鉴点 C:**"answer-segment-only" loss 而不是 full-CoT loss**(很值得试)

**Why**:S-GRPO 早退路径里 `T_1...T_P` 是共享的、不应该参与梯度(它来自 rollout 模型 π_θ_old),只有早退之后的 `</think> + C_i` 才是新生成、要参与 PPO ratio 的。这等价于**只对答案段算 loss**。

**怎么用**:我们之前的 a_token / soft-teacher fill 方案是**全 span KL**,prompt 之外的所有 token 都要算。可以试**只对 boxed{...} 之前的最后 N token 算 loss**,或者**只对 fill 之后的首 K token 算 loss**,看是否能在不掉 acc 的前提下显著降训练时间。

### 借鉴点 D:**去掉 advantage 的 std 归一化**(几行代码就能验证)

**Why**:S-GRPO 论文明说"为了稳定性"去 std。Dr.GRPO 同期工作 (arXiv 2503.20783) 也独立指出 vanilla GRPO 的 std 归一化是 "difficulty bias" 来源 —— 难题(reward 方差小)被人为放大梯度,简单题反之。

**怎么用**:如果以后跑任何 GRPO-like 方法,**默认从去 std 开始**,而不是从 vanilla GRPO 开始。这是个零成本的小优化。

### 不太适合借鉴的点

- ❌ **指数衰减 + early-exit inducer**:对我们 7B baseline 还在 72% 的状态,先解决"答对率"再考虑"啰嗦"。
- ❌ **完整 PPO clip + on-policy**:工程量很大(rollout 引擎 + LoRA hot-swap + DDP barrier),我们之前的 GRPO 3 池就栽在这里。

---

## 4. 同期(2025)其他值得借鉴的算法

### 4.1 DAPO — Decoupled Clip + Dynamic Sampling (arXiv 2503.14476)

R1-Zero 复现圈最重要的开源工作。Qwen2.5-32B base 上 AIME 2024 = 50。四个核心改动:

1. **Clip-Higher**:把 PPO 的 clip ε 从对称 `[1-ε, 1+ε]` 改成 `[1-ε_low, 1+ε_high]`,**上裁剪放大**。Why:vanilla 对称 clip 抑制了"低概率 token 突然变好"的探索 —— 因为高概率 token 1.0×ε 就饱和,低概率 token 同样 ε 却允许更大相对变动 → 高概率 token 反而长不上去。Clip-higher 给低概率 token 更大上行空间。
2. **Dynamic Sampling**:reward 全 1 或全 0 的 group(group baseline 方差为 0,梯度=0)直接**重采样**,不浪费 batch slot。
3. **Token-level loss**:vanilla GRPO loss 是 sequence-level `1/|o_i|` 归一化(每条序列等权),长序列里每个 token 权重小;改成全 batch token-level 平均,**长序列每 token 权重正常**。
4. **Overlong reward shaping**:超长但正确的样本不直接判错,而是用 soft length penalty。

**对我们的借鉴价值**:**clip-higher 和 dynamic sampling 是无脑收益**,任何 GRPO-like 训练都该开。token-level loss 在长 CoT 场景能让长样本学得更稳。

### 4.2 Dr.GRPO — Bias-Free GRPO (arXiv 2503.20783)

7B base + minimalist recipe,AIME 2024 = 43.3 SOTA。诊断了 vanilla GRPO 的两个 bias 来源:

1. **Length bias**:sequence-level `1/|o|` 让长错误回答被弱化、长正确回答被弱化 —— 在 group baseline 下,**错回答比对回答平均更长**这一现象会被这个归一化放大成"训练时模型倾向于继续生成长错回答"。
2. **Difficulty bias**:`/std` 让难题(group 内 reward 方差小)梯度被放大 —— 在难题上,几个偶然答对的样本被赋予巨大权重,可能学到的是噪声。

**修正**:**两个归一化都去掉**,直接用 mean-centered reward 做 advantage,token-level 全局平均。

**对我们的借鉴价值**:**结论很硬核 —— 直接照抄"去两个归一化"**。即使不动其他设计,这一改就消除两个已知 bias。零工程成本。

### 4.3 VinePPO — Step-level credit assignment (arXiv 2410.01679)

针对 PPO critic 在推理任务上几乎随机的问题,用 **MCTS-style branching MC rollout** 给中间步骤估 value。MATH/GSM8K 上比 PPO 快 3× wall-clock,acc 更高。

**对我们的借鉴价值**:**思路启发**多于直接套用。如果以后想做"step-level 奖励"(比如对每个 reasoning step 单独打分),这是参考实现。但工程量在 GRPO 流派之上,优先级靠后。

### 4.4 RLOO — REINFORCE with leave-one-out baseline (NeurIPS 2024)

不是 2025 工作,但**值得提**。Group rollout n 条,advantage 用 leave-one-out:

```
A_i = r_i − mean(r_{j≠i})
```

省掉 critic,比 PPO 简单,在 instruction tuning 上跟 PPO/DPO 打平甚至更好。

**对我们的借鉴价值**:如果以后做 GRPO,**leave-one-out 是 group baseline 的更好默认**。比 vanilla `(r − mean)/std` 数学性质更干净(无偏估计 advantage)。

### 4.5 Self-Rewarding 路线(综述性提及)

2024-2025 有一系列工作让模型用自己的 logprob / 自己的 verifier 头给 rollout 打分(Self-Rewarding LLM、Process-Reward Model)。**对我们 baseline 72.92% 的状态**,这些方法的前提"模型自己的判断比规则强"还不成立 —— 我们 boxed{} 字符串相等已经是足够准的 reward。**暂不考虑**。

---

## 5. 给当前节点(72.92% baseline)的建议

按"工程成本 / 预期收益"排序:

| # | 方向 | 工程量 | 预期 | 备注 |
|---|---|---|---|---|
| 1 | **Dr.GRPO 修正:去 std + token-level loss**(任何 GRPO 训练前置) | 几行代码 | 一致性收益 | 零风险 |
| 2 | **DAPO clip-higher + dynamic sampling** | 中 | 中等 acc 收益 | 工程稳定 |
| 3 | **截断式 rollout + 答案段-only loss**(从 S-GRPO 借) | 中 | 节省 50%+ rollout 成本 | 让 GRPO 能在 4 卡 H800 跑得动 |
| 4 | 完整 S-GRPO(serial + decaying reward) | 高 | 主要降长度,acc 副作用未知 | **不建议优先做** —— 我们目标是涨 acc,不是降长度 |
| 5 | VinePPO step-level credit | 极高 | 不确定 | 研究性,优先级最低 |

**最务实路线**:先 #1(去归一化)+ #2(clip-higher / dynamic sampling) → 拿到一个"修正版 GRPO"打底 → 再决定要不要叠 #3(截断 rollout 降成本)。

#1 + #2 加起来代码改动 < 100 行,不需要额外工程基础设施(LoRA hot-swap/colocate engine 这些还是要,但 reward 端是干净的)。

---

## 6. 引用

- S-GRPO: [arXiv 2505.07686v2](https://arxiv.org/abs/2505.07686) — Serial-Group Decaying-Reward PO
- DAPO: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) — Decoupled Clip + Dynamic Sampling
- Dr.GRPO: [arXiv 2503.20783](https://arxiv.org/abs/2503.20783) — Understanding R1-Zero-Like Training
- VinePPO: [arXiv 2410.01679](https://arxiv.org/abs/2410.01679) — MC-based step-level value
- DeepSeek-R1: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) — R1 / R1-Zero 原文

# SRPO + FILL：算法流程与损失公式

## 1. 总体

- **baseline = 原论文 SRPO**，未做任何改动
- **开启 FILL 后**：全错组（组内 8 条 rollout 全部答错）路由到 FILL 分支，baseline 仍是原论文的 baseline
- baseline 与 FILL **同级，一个样本只走其中一个**

---

## 2. 算法流程

```
                ┌──────────────────────────────────────┐
                │  一步：32 prompts × 8 rollouts         │
                └───────────────────┬──────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   算 reward (二值)  │
                          └─────────┬─────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  按 prompt(uid) 分组    │
                        └───────────┬───────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
   非全错组                                                  全错组
   (baseline，原论文)                                        (走 FILL)
        │                                                       │
        │                                          ┌────────────▼────────────┐
        │                                          │ 按候选池固定顺序 fill 8 个│
        │                                          │ 首 token，然后正常生成    │
        │                                          └────────────┬────────────┘
        │                                                       │
        │                                          ┌────────────▼────────────┐
        │                                          │      重新算 reward       │
        │                                          └────────────┬────────────┘
        │                                                       │
        │                                          ┌────────────▼────────────┐
        │                                          │ 选序号最小的正确那一条   │
        │                                          │ (winner)，其余全部丢弃   │
        │                                          └────────────┬────────────┘
        │                                                       │
  ┌─────▼─────────────────────┐                    ┌────────────▼────────────┐
  │ 答对          → GRPO       │                    │  winner → FILL          │
  │ 答错+有teacher → SDPO      │                    │  (整组不进 GRPO/SDPO)   │
  └─────┬─────────────────────┘                    └────────────┬────────────┘
        │                                                       │
        └───────────────────────┬───────────────────────────────┘
                                │
                  ┌─────────────▼─────────────┐
                  │  L = L_baseline + L_FILL   │
                  └───────────────────────────┘
```

**FILL 分支细节**

- 候选池按原生概率降序排列（chemistry：`To, The, We, Calcul, Determin, Analy, This, 1`）
- slot $j$ 强制候选 $j$，即 slot 序号 = 优先级
- winner = **序号最小的那条答对的**，每组最多 1 条
- 该组其余 7 条不进任何损失

---

## 3. 损失公式

### 3.1 记号

| 符号 | 含义 |
|---|---|
| $i$ | rollout 下标 |
| $t$ | token 位置 |
| $\tau$ | 首 token 位置（`len("<reasoning>\n")`） |
| $c_i$ | $\mathbf{1}[\text{答对}]$ |
| $m_i$ | $\mathbf{1}[\text{组内有正确兄弟}]$ |
| $\rho_{i,t}$ | $\pi_\theta(y_{i,t})/\pi_{\text{old}}(y_{i,t})$ |
| $A_i$ | GRPO 组相对优势 $r_i-\bar r$ |
| $M_{i,t}$ | `response_mask` |
| $W$ | winner 集合（每个全错组 1 条） |

### 3.2 路由

$$z_i^{\text{SDPO}}=(1-c_i)\,m_i,\qquad z_i^{\text{GRPO}}=1-z_i^{\text{SDPO}}$$

全错组 $g$（$\forall i\in g:\ c_i=0 \Rightarrow m_i=0$）：

$$z_i^{\text{GRPO}}=z_i^{\text{SDPO}}=0\quad \forall i\in g$$

$$z_i^{\text{FILL}}=\mathbf{1}\big[\,i=\min\{j\in g:\ \text{fill 后答对}\}\,\big]$$

### 3.3 baseline（原论文）

**GRPO 分支**

$$\ell_{i,t}^{\text{GRPO}}=-\min\Big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},\,1-0.2,\,1+0.28)\,A_i\Big)$$

**SDPO 分支**

$$\ell_{i,t}^{\text{DW-SDPO}}=w_{i,t}\cdot\operatorname{JSD}_{\alpha=0.5}\Big(\pi_\theta(\cdot\mid x,y_{<t})\ \big\|\ \pi_\theta(\cdot\mid x,f_i,y_{<t})\Big)\cdot\min(\rho_{i,t},2)$$

动态加权（论文 §3.2）：

$$w_{i,t}=\frac{\exp(-\beta H_{i,t})}{\frac{1}{|\Omega_{\text{sdpo}}|}\sum_{(j,s)\in\Omega_{\text{sdpo}}}\exp(-\beta H_{j,s})},\qquad \beta=1$$

$H_{i,t}$ 是 teacher 分布的熵；teacher $=\pi_\theta(\cdot\mid x,f_i,y_{<t})$，即**同一套权重**加特权上下文 $f_i$（原 prompt + 正确兄弟的解答），只重新打分学生已有轨迹，不重新生成。

**并集归一化（论文 §3.3）**

$$N_b=\sum_{i,t}z_i^{b}M_{i,t},\qquad
\lambda_{\text{GRPO}}=\frac{N_{\text{GRPO}}}{N_{\text{GRPO}}+N_{\text{SDPO}}},\qquad
\lambda_{\text{SDPO}}=\frac{N_{\text{SDPO}}}{N_{\text{GRPO}}+N_{\text{SDPO}}}$$

$$\mathcal{L}^{b}=\frac{\sum_{i,t}z_i^{b}M_{i,t}\,\ell_{i,t}^{b}}{N_b},\qquad b\in\{\text{GRPO},\text{SDPO}\}$$

$$\mathcal{L}_{\text{baseline}}=\lambda_{\text{GRPO}}\,\mathcal{L}^{\text{GRPO}}+\lambda_{\text{SDPO}}\,\mathcal{L}^{\text{SDPO}}$$

### 3.4 FILL

**首 token 损失** = 当前最大概率 token 的 log 概率 − 要 fill 的这个 token 的 log 概率：

$$\mathcal{L}_{\text{FILL}}^{\text{ft}}
=\frac{1}{|W|}\sum_{i\in W}\Big(\underbrace{\log\max_v \pi_\theta(v\mid x,y_{i,<\tau})}_{\text{stopgrad}}\ -\ \log \pi_\theta(y_{i,\tau}\mid x,y_{i,<\tau})\Big)$$

- 等于两者的 **logit 之差**（$\log p = z - \log Z$，$\log Z$ 相消）
- $\max_v$ 一侧 stopgrad，梯度只推高被 fill 的那个 token
- 取值 $\ge 0$，无上界；被 fill 的 token 越冷门则越大
- 对被 fill token 的 logit 求导恒为 $-1$，不随 $p_{\text{forced}}$ 衰减

**后续 token 损失** = 正常 CE，只在 $t\neq\tau$ 上取均值：

$$\mathcal{L}_{\text{FILL}}^{\text{cont}}
=\frac{\sum_{i\in W}\sum_{t\neq\tau}M_{i,t}\big(-\log \pi_\theta(y_{i,t})\big)}
{\sum_{i\in W}\sum_{t\neq\tau}M_{i,t}}$$

**FILL 总损失**

$$\mathcal{L}_{\text{FILL}}=\mathcal{L}_{\text{FILL}}^{\text{ft}}+\mathcal{L}_{\text{FILL}}^{\text{cont}}$$

### 3.5 最终目标

$$\boxed{\ \mathcal{L}=\lambda_{\text{GRPO}}\,\mathcal{L}^{\text{GRPO}}+\lambda_{\text{SDPO}}\,\mathcal{L}^{\text{SDPO}}+\mathcal{L}_{\text{FILL}}\ }$$

FILL 与 baseline 同级，直接相加，不参与 $\lambda$ 归一化。

---

## 4. 代码位置

| 内容 | 文件 : 行 |
|---|---|
| 全错组识别、fill、选 winner | `verl/trainer/ppo/ray_trainer.py` : 1123-1359 |
| 三分支掩码与损失合并 | `verl/workers/actor/dp_actor.py` : 940-1105 |
| 首 token 损失 | `verl/workers/actor/dp_actor.py` : 1035-1050 |
| SDPO 损失与动态加权 | `verl/trainer/ppo/core_algos.py` : 1085-1225 |

---

## 5. 关键配置

| 参数 | 值 |
|---|---|
| `train_batch_size` / `rollout.n` | 32 / 8 |
| `ppo_mini_batch_size` | 32（每卡全量 → 每步 1 次更新，on-policy） |
| `lr` / `lr_warmup_steps` | 5e-6 / 10 |
| GRPO `clip_ratio_low` / `high` | 0.2 / 0.28 |
| SDPO `top-K` / `α` / `β` / EMA | 100 / 0.5 / 1 / 0.05 |
| `n_baseline_keep` / `n_tokens_per_group` | 0 / 8 |
| `response_prefix` | `"<reasoning>\n"` |
| `success_reward_threshold` | 1.0 |

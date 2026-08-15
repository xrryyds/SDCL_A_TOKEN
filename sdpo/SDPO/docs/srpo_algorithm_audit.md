# 当前算法设计复述与论文对照（SRPO + FILL）

对照文档：`docs/MinerU_markdown_2604.02288v1_2083097477240627200.md`（SRPO, arXiv 2604.02288）
代码基：`sdpo/SDPO`（= [lasgroup/SDPO](https://github.com/lasgroup/SDPO) 的副本）
审计日期：2026-08-16

---

## 一、一句话概述

每步对 32 个 prompt 各采样 8 条 rollout，按**每条 rollout 的学习状态**分流到不同的监督信号：答对的走 GRPO（序列级 reward 优势），答错且组内有正确兄弟的走 SDPO（logit 级蒸馏纠错），全错组回落 GRPO（优势恒 0，等于不学）。FILL 是我们的扩展：把全错组这部分"浪费掉"的样本，用强制首 token 重新生成来抢救。

---

## 二、SDPO 分支：何时用、怎么用

### 2.1 何时用（路由）

论文 §3.1 定义两个二值指示量：

- `c_i = 1[y_i 正确]`
- `m_i = 1[y_i 有可用的 teacher 信息]`

```
z_i^SDPO = (1 - c_i) · m_i
z_i^GRPO = 1 - z_i^SDPO
```

论文 Table 7 的完整决策矩阵，与我们实现的对应关系：

| c_i | m_i | teacher prompt 内容 | 论文路由 | 我们的实现 | 一致 |
|---|---|---|---|---|---|
| ✓ | ✓ | question + 兄弟解答 | GRPO | `(seq_scores < 1.0)` 为 0 → mask=0 | ✅ |
| ✓ | ✗ | question only | GRPO | 同上 | ✅ |
| ✗ | ✓ | question + 兄弟解答 | **SDPO** | `solution_strs[i] is not None` 且答错 → mask=1 | ✅ |
| ✗ | ✗ | question only | GRPO（回落） | 全错组无正确兄弟 → `solution_strs[i]=None` → mask=0 | ✅ |

代码位置 `verl/trainer/ppo/ray_trainer.py:794-810`：

```python
self_distillation_mask = torch.tensor(
    [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)], ...
)
if loss_mode == "srpo":
    seq_scores = reward_tensor.sum(dim=-1)
    self_distillation_mask = self_distillation_mask * (seq_scores < success_threshold)
```

两个论文明确要求的细节也已实现：

- **唯一正确者不能当自己的 teacher**：`dont_reprompt_on_self_success=True` → `_get_solution` 中 `solution_idxs = [j for j in solution_idxs if j != idx]`（`ray_trainer.py:663`）
- **兄弟选取**：同 `uid` 组内取第一个正确的（`solution_idxs[0]`）

### 2.2 怎么用（teacher 分布的构造）

论文定义 teacher 分布为

```
q_{i,t}(v) = π_θ(v | x, f_i, y_{i,<t})
```

三个要点，逐一核对：

**① teacher 是当前策略本身，不是独立权重的副本**

论文中 `π_θ` 就是学生自己；差异**完全来自上下文 `f_i`**。

我们的实现（`dp_actor.py:895`）：
```python
teacher_model = self.teacher_module or self.actor_module
```
由于 `use_kl_loss=False` → `need_reference_policy()` 返回 False → ref 分支不实例化 → `teacher_module` 从未赋值 → **回落为 `actor_module`（当前策略）**。

结果**符合论文**。但要注意：配置里的 `teacher_regularization='ema'` 和 `teacher_update_rate=0.05` 是**死配置，代码路径从未走到**。（我在早期核对 Table 3 时误将其记为"已对齐 EMA=0.05"，那个勾打错了——不过方向上无害，因为论文本就不要独立 EMA teacher。）

**② `f_i` = 原 prompt + 正确兄弟的解答**

`ray_trainer.py:735-760`：
```
"{prompt}"
"\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
"\n\nCorrectly solve the original question.\n"
```
与论文附录一致。

**③ teacher 不生成新回答，只重新打分学生已有轨迹**

`ray_trainer.py:782`：
```python
teacher_input_ids = torch.cat([teacher_prompt["input_ids"], responses], dim=1)
```
直接把**学生自己的 `responses`** 拼在特权 prompt 后做一次 forward（`torch.no_grad()`）。符合论文 "the self-teacher does not generate a new response; it re-scores the student's existing trajectory"。

### 2.3 SDPO 损失本身

`core_algos.py:1085-1225`。散度计算：

- 取 student / teacher 各自的 **top-100** log-prob（`distillation_topk=100`）
- 加一个 **tail bucket**（`distillation_add_tail=True`）：用 `log(1-Σp_i)` 把截断掉的尾部概率质量补成第 101 个"桶"，保证是合法分布
- `alpha=0.5` 走 **Jensen-Shannon**：先算混合分布 `m = (1-α)·p_s + α·p_t`（在 log 空间用 `logsumexp`），再 `lerp(KL(m‖p_s), KL(m‖p_t), α)`
- `per_token_loss = kl_loss.sum(-1)`

与论文 Table 3（top-K=100 / Jensen-Shannon）一致。

### 2.4 动态加权（§3.2）

论文：`w̃_{i,t} = exp(−β·H_{i,t})`，其中 `H` 是 teacher 熵；再归一化到在 `Ω_sdpo` 上均值为 1。

我们的实现（`core_algos.py:1186-1203`）：
```python
teacher_probs = teacher_distill_log_probs.exp()
teacher_entropy = -(teacher_probs * teacher_distill_log_probs).sum(-1)
dw_weight = torch.exp(-dw_beta * teacher_entropy)
# 跨 GPU all_reduce 求均值后归一化
dw_weight = (dw_weight / dw_mean).detach()
per_token_loss = per_token_loss * dw_weight
```

`β=1`，且归一化用了 `all_reduce`（全局均值，而非单卡），这比单卡归一更贴近论文的 `Ω_sdpo` 定义。

> **⚠️ 已知偏差（唯一一处实质性偏差）**
>
> 论文 §3.2 的 `H_{i,t}` 是**全词表**熵：`H = -Σ_{v∈V} q log q`。
> 我们算的是 **top-100 + tail 桶**上的熵（代码注释自己也标了 "this is a lower bound on the true teacher entropy"）。
>
> 影响：这是真实熵的**下界**，系统性偏低。因为 `w̃ = exp(−β·H)`，熵偏低 → 权重偏高且**权重之间的差异被压缩**。归一化到均值 1 之后，绝对尺度无影响，但**区分度被削弱**——即"抑制高熵不可靠目标"这个机制的力度弱于论文。
>
> 实测该效应在训练后期变得严重：`dw_weight_std` 从 0.440 → 0.103（最低 0.028），即动态加权到后期**几乎退化为无差别加权**，论文声称的"10h 时贡献 +1.8 分"这一项在我们这里基本失效。

### 2.5 额外的 IS 修正

`is_clip=2.0`（`sdpo.yaml` 官方值）：
```python
ratio = torch.exp(student_log_probs - old_log_probs).clamp(max=2.0)
per_token_loss = per_token_loss * ratio
```
另有 `rollout_is=token` / `rollout_is_threshold=2.0` 的 rollout 修正权重。两项均与官方 `sdpo.yaml` 一致。

---

## 三、GRPO 分支

- 优势：`A_i = (r_i - r̄) / (σ_r + ε)`，但 `norm_adv_by_std_in_grpo=False` → **不除标准差**（与官方 `sdpo.yaml` 一致）
- 损失：标准 PPO clip，`ε_low=0.2`、`ε_high=0.28`（论文 Table 3 的 asymmetric clip）
- 掩码：`grpo_response_mask = response_mask × (1 − sd_mask) × (1 − fill_group_mask)`

**关于 clip 恒不生效**：`ppo_mini_batch_size=32` × `rollout_n=8` ÷ 8 卡 = 每卡 32 条 = 该卡全部样本 → 每步仅 1 次梯度更新 → `ρ ≡ 1` → `pg_clipfrac ≡ 0`、`ppo_kl ≡ 0` 全程。
这**符合论文**——论文摘要反复强调 SRPO 是 "unified **on-policy** framework"。ε_high=0.28 在这种配置下是不起作用的装饰参数。

---

## 四、三分支合并（§3.3）

论文的统一目标是**单一并集分母**：

```
L = [ Σ_{i,t} z^GRPO·ℓ^GRPO + Σ_{i,t} z^SDPO·ℓ^DW-SDPO ] / [ Σ_{i,t} z^GRPO + Σ_{i,t} z^SDPO ]
```

关键含义：分母是**所有被路由 token 的总数**，所以当 SDPO 样本比例随训练下降时，SDPO 项的权重**自动衰减**，不需要额外的混合超参。

我们的实现（`dp_actor.py:1038-1079`）用 λ 加权等价复现：

```python
grpo_token_cnt = grpo_response_mask.sum()
sd_token_cnt   = sd_response_mask.sum()
total_token_cnt = grpo_token_cnt + sd_token_cnt + fill_token_cnt
lambda_grpo = grpo_token_cnt / total_token_cnt
lambda_sdpo = sd_token_cnt   / total_token_cnt
lambda_fill = fill_token_cnt / total_token_cnt
pg_loss = lambda_grpo*grpo_loss + lambda_sdpo*sd_loss + lambda_fill*fill_loss
```

因为每个分支的 `agg_loss` 已是各自 token 的均值，乘以 token 份额后相加，即还原成并集分母的加权平均。

**自衰减实测生效**：`lambda_sdpo` 从 0.33（step 1）降到 0.10（step 200+），随答对率上升自动让位给 GRPO，与论文 Figure 5(a)(b) 描述的趋势一致。

---

## 五、FILL 分支（我们的扩展，非论文内容）

论文的 SRPO 对全错组（`m_i=0` 对所有 i）只能回落 GRPO，而此时组内 reward 全为 0 → 优势恒 0 → **这批样本完全不产生梯度，被浪费**。FILL 就是回收这部分。

流程（`ray_trainer.py`）：

1. 识别全错组（`dead_groups`）
2. 对该组的 8 个 slot，按**候选池固定顺序**各强制一个首 token 重新生成
   - 池顺序：前 2 位是该数据集上原生高概率的开头（chemistry: `To`/`The`），后 6 位是低概率但有语义引导性的
   - 强制方式：把 token 拼在 `response_prefix="<reasoning>\n"` 之后，让模型从这里续写
3. 重新算 reward，**答错的全部丢弃**
4. 每组只学**序号最小的那条正确 rollout**（`fill_correct_mask`）
5. 损失：`fill_per_token = -clamp(ratio, 1±0.28) × ft_scale`，其中 `ft_scale = 1 + (w_ft−1)·fill_first_token_mask`
   - 当前 `w_ft=1`，即首 token 不额外放大（早期试过 5 和 9，判定为纯扰动税）

**性质说明**：强制 token 不是策略采样出来的，且 `old_log_probs` 在 rescue 后被重算导致 `ρ≈1`、真实重要性权重（`π_θ/π_forced ≈ 0.002`）被丢弃。所以这一支实质是**带 clip 的加权最大似然（RFT/STaR 风格）**，不是严格的 policy gradient。在人为构造的组上做组相对优势没有意义，因此这里用单位权重而非优势。

---

## 六、与论文的偏差清单（截至本次审计）

### 已核实完全一致

| 类别 | 内容 |
|---|---|
| 超参 Table 3 | lr 5e-6 / mini-batch 32 / batch 32 / rollout n=8 / ε_high 0.28 / ρ clip 2 / KL coef 0 / entropy_coeff 0 / β=1 / top-K 100 / JSD α=0.5 / warmup 10 / wd 0.01 / grad clip 1.0 |
| 数据集 | Chemistry 1890/210、Biology 450/50（与论文 Table 4 逐字相同，同源 SDPO 官方 split） |
| 采样 | 训练 T=1.0/top_p=1；验证 n=16/T=0.6/top_p=0.95；`enable_thinking=False` |
| SDPO 路由 | Table 7 全四行 |
| teacher 构造 | `π_θ` 本身 + 特权上下文；不重新生成，只重打分 |
| §3.3 并集归一化 | λ 加权等价实现，自衰减实测生效 |
| lr 调度 | warmup 10 步后恒定 5e-6，无衰减 |
| 环境 | torch 2.8.0（论文 B.1 指定）、SGLang 引擎、8×H20 |
| 官方脚本对照 | `experiments/generalization/run_sdpo_all.sh` + `sdpo.yaml`：`norm_adv_by_std_in_grpo=False`、`is_clip=2.0`、`rollout_is=token/2.0`、`max_model_len=18944`、`calculate_log_probs=True` 全部一致 |

### 存在偏差

| # | 偏差 | 严重度 | 说明 |
|---|---|---|---|
| 1 | **动态加权的 teacher 熵用 top-100+tail 而非全词表** | **中** | 是真实熵的下界，导致权重区分度被压缩；实测 `dw_weight_std` 后期降到 0.028，该机制几乎失效。论文称此机制在 10h 贡献 +1.8 分 |
| 2 | CUDA 12.8 / 驱动 580 vs 论文 12.4 / 550 | 低 | 无法降级，torch 自带 runtime |
| 3 | 代码含 FILL/entropy-band/neg-cov 等额外分支 | 无 | baseline 模式下实测全惰性（`fill_group_n=-1`、`λ_fill≡0`、两 coef=0） |
| 4 | 归一化修正（`SRPO_UNION_NORM`） | 无 | 实测数值等价：单 micro-batch 时两种算法都=1.0，双 micro-batch 时都≈0.5 |

### 无法核实

- SRPO 论文**无开源代码**（arXiv 页无 code 链接，正文称 "plan to release implementation details"）
- 公开的 SDPO W&B 日志只有 **Olmo-3-7B-Instruct** 的 run，没有 Qwen3-8B，无法直接比对熵/长度的绝对量级

---

## 七、实验结果与论文对照

按论文的 wall-clock 预算口径（Table 1 报"预算内最高 avg@16"），Chemistry / Qwen3-8B：

| | 1h | 5h | 10h |
|---|---|---|---|
| 论文 base | 41.1 | 41.1 | 41.1 |
| 论文 GRPO | 62.1 | 75.9 | 78.9 |
| 论文 SDPO | 71.6 | 80.6 | 80.6 |
| **论文 SRPO** | **69.2** | **81.8** | **83.0** |
| 我们 baseline（纯 SRPO） | **74.6** | 77.9 | 77.9 |
| 我们 + FILL | 74.2 | 78.2 | **80.3** |

**关键观察**

1. **1h 我们反超论文 5.4 分**（74.6 vs 69.2），说明实现无硬伤
2. **83.0 是 10h 的值**；450 步 ≈ 5h，对应基准是 **81.8**。所以 baseline 差 3.9 分、FILL 差 1.5 分（早先把差距说成 12.9 分是用错了基准）
3. **baseline 在 step ~220 熵坍缩**，FILL 不坍缩：

| step | baseline 熵 / 长度 | FILL 熵 / 长度 |
|---|---|---|
| 1 | 0.515 / 448 | 0.532 / 458 |
| 121 | 0.338 / 251 | 0.431 / 329 |
| 201 | 0.259 / 312 | 0.496 / 338 |
| 241 | **0.066 / 1431** | 0.420 / 341 |
| 最新 | **0.053 / 1734** | 0.506 / 447 |

坍缩后的输出退化为空洞重复模板（`**Option A** is scientifically accurate, numerically precise...` 重复十余遍），首 token 分布坍缩为单点（`The:1.00` → `Option:1.00`），`grad_norm` 降到 0.01，速度慢 5 倍。

**坍缩链条**：二值 reward 对 reasoning 内容零梯度 + `entropy_coeff=0` + 无 KL 锚 → 学生塌向单一高奖励模板 → **teacher 与学生同权重，只差上下文，学生一自信 teacher 也自信** → 两分布 JSD 趋零 → SDPO 梯度消失；同时答对率高导致 53% 的 micro-batch 无错样本可路由（`empty_target_batch=0.531`）→ 实质退化成纯 GRPO 在已坍缩策略上继续锐化。

**FILL 为何能阻断**：每步强制注入少量非原生首 token，打断"熵下降 → teacher/学生同步自信 → JSD 趋零"这个自我强化环。这也解释了它在 Biology（全错组占 14.7%，料更多）上领先 baseline 达 +10 分。

---

## 八、下一步可查方向

1. **修正偏差 #1**：把动态加权的 teacher 熵改为全词表计算，验证是否恢复论文所述的后期增益（这是目前唯一已知的实质偏差，且方向上正好对应"后期涨不上去"）
2. `chemistry_filtered`：原作者 W&B 显示他们用的数据集名为 `sciknoweval/chemistry_filtered`，我们用 `chemistry`。本地无该变体，`data/load_dataset.py` 中的 `filtered` 只是切片变量名、非内容过滤，具体筛选规则未知
3. 步数预算：若要对标 10h，需把 `total_training_steps` 从 450 提到约 900

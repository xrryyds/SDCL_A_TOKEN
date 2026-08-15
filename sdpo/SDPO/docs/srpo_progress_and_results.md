# SRPO 复现 + FILL 扩展：进度、算法设计与实验结果

最后更新：2026-08-16
对照论文：`docs/MinerU_markdown_2604.02288v1_2083097477240627200.md`（SRPO, arXiv 2604.02288）
代码基：`sdpo/SDPO` = [lasgroup/SDPO](https://github.com/lasgroup/SDPO) 的副本
详细算法审计见：`sdpo/SDPO/docs/srpo_algorithm_audit.md`

---

## 1. 目标

1. **忠实复现** SRPO 论文在 SciKnowEval 上的结果（Qwen3-8B）
2. **扩展 FILL 分支**：论文的 SRPO 对"全错组"（组内 8 条 rollout 全答错）只能回落 GRPO，此时组内 reward 全 0 → 优势恒 0 → **这批样本不产生任何梯度、被完全浪费**。FILL 用强制首 token 重新生成来回收这部分信号。

---

## 2. 算法设计

### 2.1 SRPO 主干（严格按论文）

每步 32 个 prompt × 8 条 rollout。按**每条 rollout 的学习状态**路由：

```
z_i^SDPO = (1 - c_i) · m_i          c_i = 1[答对]，m_i = 1[有 teacher 信息]
z_i^GRPO = 1 - z_i^SDPO
```

| c_i | m_i | 路由 | 我们的实现 |
|---|---|---|---|
| ✓ | ✓ | GRPO | `seq_scores >= 1.0` → mask=0 |
| ✓ | ✗ | GRPO | 同上 |
| ✗ | ✓ | **SDPO** | 答错 且 `solution_strs[i] is not None` |
| ✗ | ✗ | GRPO（回落，A≡0） | 全错组无正确兄弟 → `solution_strs[i]=None` |

**SDPO 的 teacher**（论文 `q = π_θ(v | x, f_i, y_{i,<t})`）：

- teacher **就是当前策略本身**，不是 EMA 副本。差异**只来自特权上下文 `f_i`**
- `f_i` = 原 prompt + `"\nCorrect solution:\n\n{正确兄弟的解答}\n\n"` + `"Correctly solve the original question."`
- teacher **不生成新回答**，只在特权上下文下**重新打分学生自己的轨迹**（`torch.cat([teacher_prompt, responses])`）
- 散度：top-100 + tail 桶，JSD（α=0.5）
- 唯一正确者不当自己的 teacher（`dont_reprompt_on_self_success=True`）

**动态加权（§3.2）**：`w̃ = exp(−β·H_teacher)`，β=1，跨 GPU all_reduce 归一化到均值 1。

**三分支合并（§3.3 单一并集分母）**，用 λ 加权等价实现：
```python
lambda_grpo = grpo_token_cnt / total_token_cnt
lambda_sdpo = sd_token_cnt   / total_token_cnt
lambda_fill = fill_token_cnt / total_token_cnt
pg_loss = λ_grpo·grpo_loss + λ_sdpo·sd_loss + λ_fill·fill_loss
```
自衰减实测生效：`λ_sdpo` 从 0.33（step 1）→ 0.10（step 200+）。

**on-policy**：`mini_batch=32 × rollout_n=8 ÷ 8卡` = 每卡全量 → 每步 1 次更新 → `ρ≡1`、`clipfrac≡0`。符合论文"unified on-policy framework"。

### 2.2 FILL 分支（我们的扩展，非论文内容）

1. 识别全错组
2. 该组 8 个 slot 按**候选池固定顺序**各强制一个首 token 重新生成（前 2 位是该数据集原生高概率开头，后 6 位是低概率但有语义引导性的）
3. 答错的**全部丢弃**
4. 每组只学**序号最小的那条正确 rollout**
5. 损失：`-clamp(ratio, 1±0.28) × ft_scale`，`w_ft=1`（早期试过 5/9，判定为纯扰动税）

**性质**：强制 token 非策略采样，`old_log_probs` 重算后 `ρ≈1`，真实重要性权重（≈0.002）被丢弃 → 实质是**带 clip 的加权最大似然（RFT/STaR 风格）**，不是严格 policy gradient。故用单位权重而非组相对优势。

---

## 3. 实验结果

### 3.1 Chemistry（vLLM 环境，450 步跑完）

| | peak | last5 均值 |
|---|---|---|
| baseline（纯 SRPO） | **80.3** @400 | 78.7 |
| fill（w_ft=5） | 78.9 @330 | 77.4 |

**fill 输 1.4 分。** 原因：chemistry 全错组只占 **5.2%**，`λ_fill≈0.3%`（舍入误差级），fill 无料可榨却要付 `w_ft=5` 的强制扰动税。

### 3.2 Biology（更难，base 30.5）

| | peak | last5 均值 |
|---|---|---|
| baseline | 59.9 @420 | 57.5 |
| **fill（biology 专属池，w_ft=1）** | **69.9** @355 | **65.2** |

**fill 赢 +10.0 分。** 全错组占 **7.5%（峰值 28.1%）**，远高于 chemistry。为 biology 单独生成候选池后强制命中率从 4.2% → 5.4%（`440/8096`），FILL winner 从每步 0-1 增至 2-3。

**验证了核心假设**：FILL 的收益取决于"有多少全错组可榨"。

### 3.3 Chemistry（SGLang + torch 2.8，对齐论文 B.1 环境）

| | peak | 当前/末值 |
|---|---|---|
| baseline | 77.9 @165 | **68.4**（坍缩后被终止于 step 252） |
| fill（w_ft=1） | **80.3** @250 | 77.2 |

**baseline 在 step ~220 模式坍缩，fill 未坍缩：**

| step | baseline 熵 / 长度 | fill 熵 / 长度 |
|---|---|---|
| 1 | 0.515 / 448 | 0.532 / 458 |
| 121 | 0.338 / 251 | 0.431 / 329 |
| 201 | 0.259 / 312 | 0.496 / 338 |
| 241 | **0.066 / 1431** | 0.420 / 341 |
| 末 | **0.053 / 1734** | 0.506 / 447 |

坍缩后输出退化为空洞重复模板（`**Option A** is scientifically accurate, numerically precise...` 重复十余遍），首 token 分布坍缩为单点（`The:1.00` → `Option:1.00`），`grad_norm` 降至 0.01，速度慢 5 倍（41s → 218s/step）。

### 3.4 与论文对照（Chemistry / Qwen3-8B，wall-clock 口径）

| | 1h | 5h | 10h |
|---|---|---|---|
| 论文 base | 41.1 | 41.1 | 41.1 |
| 论文 GRPO | 62.1 | 75.9 | 78.9 |
| 论文 SDPO | 71.6 | 80.6 | 80.6 |
| **论文 SRPO** | **69.2** | **81.8** | **83.0** |
| 我们 baseline (SGLang) | **74.6** | 77.9 | 77.9 |
| 我们 fill (SGLang) | 74.2 | 78.2 | **80.3** |

**关键结论**

1. **1h 我们反超论文 5.4 分**（74.6 vs 69.2）→ 实现无硬伤
2. **83.0 是 10h 的值**。450 步 ≈ 5.2h，对应基准是 **81.8**。baseline 差 3.9 分、fill 差 1.5 分
   （早前把差距说成 12.9 分是**用错了基准**，已纠正）
3. **论文不按步数报告**，全部按 wall-clock。`total_training_steps=450` 是我们自己设的，论文未给步数
4. 我们的 s/step（41-45s）显著快于论文推算值（约 120s/step），**说明两者对比的不是同一训练阶段**

---

## 4. 环境对齐（论文附录 B.1）

| 项 | 论文 | 我们 | |
|---|---|---|---|
| GPU | 8× H20 NVLink | 8× H20 | ✅ |
| PyTorch | **2.8.0** | **2.8.0+cu128** | ✅ |
| Rollout 引擎 | **SGLang** | **SGLang 0.5.2** | ✅ |
| 框架 | verl + FSDP | verl | ✅ |
| CUDA / 驱动 | 12.4 / 550.144.03 | 12.8 / 580.105.08 | ⚠️ 不可降 |

环境搭建踩的坑（已解决）：
- vLLM 0.12.0（为 torch 2.9 编译）与 torch 2.8 ABI 冲突 → `std::bad_alloc`，需卸载
- flash-attn 安装失败真因是**它去 GitHub 下预编译 wheel 被内网拦截**，非编译错误 → `FLASH_ATTENTION_FORCE_BUILD=TRUE` 强制本地编译
- SGLang 每卡起两个进程（`WorkerDict` 训练 + `sglang` 推理服务，后者占 60-76GB），kill 后须等显存真正释放

---

## 5. 本轮代码改动

**唯一实质偏差已修复**（开关 `SRPO_EXACT_DW_ENTROPY`，默认开启）：

**问题**：论文 §3.2 的 `H` 是**全词表**熵 `−Σ_{v∈V} q log q`；我们从 **top-100 + tail 桶**反推，是真实熵的**下界**（代码注释自己标了 "lower bound"）。

**后果（实测）**：熵被系统性低估 → `exp(−β·H)` 取值范围被压缩 → `dw_weight_std` 从 0.440 降到 **0.103**（最低 0.028），**动态加权后期几乎退化为无差别加权**。论文称该机制在 10h 贡献 **+1.8 分**，正好对应"后期涨不上去"。

**改动**：
- `dp_actor.py`：`dw_beta>0` 时 teacher forward 改 `calculate_entropy=True`，用 verl 的 `entropy_from_logits`（`logsumexp(logits) − Σ softmax·logits` 恒等式，不物化全词表，显存可控）
- `core_algos.py`：新增 `teacher_entropy_exact` 参数，优先用精确熵；取不到才回落，并新增 `srpo/dw_entropy_exact` 指标区分路径
- 该指标放在无条件初始化处，避免 `reduce_metrics` 因 key 缺失报错

**验证判据**：`srpo/dw_entropy_exact` 应为 `1.0`；`teacher_entropy_mean` 应显著高于旧版 0.487；`dw_weight_std` 后期不应再退化到 0.03。

其他工具性改动（不影响算法）：`DATA_PATH` / `CANDIDATE_POOL_PATH` / `RUN_TAG` / `ROLLOUT_ENGINE` 改为环境变量可覆盖；修掉输出目录硬编码的 `srpo_v10_chem_` 前缀。

---

## 6. 已核实一致的项（详见审计文档）

超参 Table 3 全项、数据集划分（Chemistry 1890/210、Biology 450/50）、SDPO 路由 Table 7 全四行、teacher 构造、JSD/top-K/tail、§3.3 并集归一化、lr 调度（warmup 10 + 恒定 5e-6）、采样参数（训练 T=1.0；验证 n=16/T=0.6/top_p=0.95）、以及原作者官方脚本 `experiments/generalization/run_sdpo_all.sh` + `sdpo.yaml` 的全部设置（`norm_adv_by_std_in_grpo=False`、`is_clip=2.0`、`rollout_is=token/2.0`、`max_model_len=18944`）。

**已纠正的两个自身错误**：
1. `teacher_regularization='ema'` / `teacher_update_rate=0.05` 是**死配置**——`use_kl_loss=False` → ref 不实例化 → `teacher_module` 未赋值 → teacher 回落为 `actor_module`。这**恰好符合论文**（论文 teacher 就是 π_θ 本身），但早期核对时看到配置值就打了勾，那个勾是错的
2. 归一化修正（`SRPO_UNION_NORM`）**数值等价**、非差距来源：单 micro-batch 时两算法都=1.0，双 micro-batch 时都≈0.5

---

## 7. 未复现的部分（重要缺口）

**论文 Table 2 的消融，一个都没做：**

| 论文消融 | 状态 |
|---|---|
| SRPO（完整，含 DW） | 部分——DW 曾因 top-k 近似而失效，刚修复 |
| **SRPO w/o dynamic weighting**（`dw_beta=0`） | ❌ 从未跑 |
| **Advantage Mix**（λ=0.9 优势级混合） | ❌ 代码未实现 |
| **五 benchmark 平均** | ❌ 只有 chemistry + biology，缺 physics / materials / tooluse |

注意：论文 Table 2 的数值是**五项平均**，不能与我们的 chemistry 单项直接对照。此前的对比口径不严谨。

---

## 8. 当前进度

**正在跑**（chemistry，SGLang + torch 2.8，均为修复后代码）：
- 本机：`sg2-baseline`（`FILL_ENABLE=False`）
- 另一机器：`sg2-fill`（`FILL_ENABLE=True, FILL_FT_WEIGHT=1`）

两条同代码同环境，唯一差别 `FILL_ENABLE`，为单变量对照。

**下一步（按价值排序）**

1. **`dw_beta=0` 对照**（改一个参数）——直接验证 DW 是否真在起作用，并对上论文 Table 2 第二组。若它与当前 baseline 接近，则坐实"top-k 近似令 DW 失效"
2. **实现 Advantage Mix（λ=0.9）**——验证论文核心主张"sample routing 优于优势级混合"
3. 补齐 physics / materials / tooluse，才能对齐五项平均
4. `chemistry_filtered`：原作者 W&B 显示其数据集名为 `sciknoweval/chemistry_filtered`，我们用 `chemistry`；本地无该变体，筛选规则未知

**待解释的现象**：论文声称 SRPO 长期稳定、"moderate response lengths"，而我们的 baseline 在 step 220 坍缩。已排除超参/数据/路由/teacher 构造/环境/lr 调度，也确认原作者官方脚本同样**没有**长度或重复惩罚。SRPO 无开源代码（作者称 "plan to release"），公开的 SDPO W&B 只有 Olmo-3-7B 的 run（熵 2.6-3.3、长度 166-423，但模型不同不可直接比），故暂无法定论。

**FILL 的价值已被两组实验支持**：Biology 上 +10.0 分；Chemistry SGLang 上 baseline 坍缩而 fill 不坍缩（80.3 vs 77.9）。机制解释是强制首 token 每步注入少量不可预测性，打断了"熵下降 → teacher 与学生同步自信 → JSD 趋零 → SDPO 停摆 → 纯 GRPO 在坍缩策略上继续锐化"这一自我强化环。

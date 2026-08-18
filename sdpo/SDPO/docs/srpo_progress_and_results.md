# SRPO 复现 + FILL 扩展：当前进展

最后更新：2026-08-18
对照论文：`docs/MinerU_markdown_2604.02288v1_2083097477240627200.md`（SRPO, arXiv 2604.02288）
算法设计与损失公式见：`docs/fill_branch_design_current.md`

---

## 1. 目标与现状

**目标**：① 复现论文 SRPO 在 chemistry 上的 83.0；② 用 FILL 分支回收全错组，在此基础上加分。

**现状**：

| | 最好成绩 | 论文 |
|---|---|---|
| baseline（纯 SRPO） | **82.7** @step440 | **83.0**（10h） |
| FILL | **81.8** @step480 | — |

83.0 差 0.3 点已经摸到，但**不可稳定复现**——同配置多次运行方差达 9.4 点。

---

## 2. 论文对标口径（已厘清）

论文按 wall-clock 预算报告，不给步数。由论文 §4.4 的每步耗时反推出对应步数：

| 预算 | 论文每步耗时 | **对应步数** | 论文 chemistry |
|---|---|---|---|
| 1h | 83.4s | **43** | 69.2 |
| 5h | 78.3s | **230** | 81.8 |
| **10h** | 75.8s | **475** | **83.0** |

我们每步 33-71s（比论文快，因响应更短），所以 `total_training_steps` 已从 450 提到 **800**，可覆盖 475 步的 10h 位置。

---

## 3. 全部实验结果

| run | 类型 | 熵实现 | 止于 | peak | 末长度 | 末熵 | 结局 |
|---|---|---|---|---|---|---|---|
| **sg2-baseline** | baseline | top-k | 450 | **82.7** | 317 | 0.427 | **稳定** |
| chem_baseline-nofill | baseline | top-k(旧) | 450 | 80.3 | 1144 | 0.440 | 长度爆炸 |
| sg-baseline | baseline | top-k(旧) | 250 | 77.9 | 1325 | 0.078 | 长度爆炸 |
| dw0-baseline | baseline | top-k | 100 | 73.3 | 1027 | 0.086 | 长度爆炸 |
| dw1-baseline | baseline | 全词表 | 225 | 77.8 | 327 | 0.638 | 稳定（被误杀） |
| v11-baseline | baseline | 全词表(抢卡) | 285 | 74.5 | 733 | 0.300 | 长度爆炸 |
| f800-baseline | baseline | 全词表 | 225 | 77.0 | 1183 | 0.389 | **OOM 崩溃** |
| dwk0-baseline | baseline | top-k | 在跑 | — | — | — | — |
| chem_fill8-wft5 | fill | 旧loss | 450 | 78.9 | 729 | 0.793 | 长度爆炸 |
| **sg2-fill** | fill | 旧loss | 450 | **81.4** | 295 | 0.273 | 稳定 |
| v11b-fill | fill | 新loss | 450 | 78.3 | 501 | 0.527 | 稳定 |
| **f800-fill** | fill | 新loss | 700 | **81.8** | 69 | 3.100 | **熵爆炸** |
| bio2-fill | fill(bio) | 旧loss | 420 | 69.9 | 141 | 0.444 | 稳定 |

---

## 4. 核心发现

### 4.1 长度爆炸是复现失败的主因，且随机触发

**7 条 baseline 里 5 条以长度爆炸收场**（末长度 733-1325，正常应为 230-330）。

崩溃时的输出内容已确认是退化模板，`Option A is the most accurate...` 重复 6-7 遍：

```
<reasoning>
The target molecule "Nc1ccn(...)" includes a fluorinated aromatic ring...
- Option A correctly reflects the fluorinated ring...
Option A accurately captures the fluorinated ring...
Option A is the most accurate and comprehensive representation...
Option A is the only option that accurately reflects...
Option A is the most accurate and correct reactant...     ← 重复 6-7 遍
</reasoning>
<answer> A </answer>     [score] 1.0
```

reward 是纯二值只看 `<answer>`，reasoning 段写什么、多长、是否重复**完全没有梯度**，所以重复填充不受惩罚。

`f800-baseline` 更进一步：长度涨到 1183、`len_max` 撞 8192 上限 → 单 micro-batch 需 23GB 显存 → **CUDA OOM 崩溃**。

**分水岭在 step 80-120**：

| step | sg2（成功） | 崩掉的那几条 |
|---|---|---|
| 40 | 291 | 311 |
| 80 | **227** | 300-303 |
| 120 | **228** | **361+** |

`sg2` 早期就把长度压到 227 并全程稳定在 182-317；崩掉的那些从 300 一路往上涨。

论文 Figure 4(a) 明确说 SRPO "yields moderate response lengths"，我们复现不出这个稳定性。**这是与论文最明显的行为差异。**

### 4.2 动态加权的熵实现：两种都会崩，不是差异来源

论文 §3.2 定义 `H = −Σ_{v∈𝒱} q log q` 是**全词表**熵。代码原先从 top-100 + tail 反推（下界），我修正为全词表（`SRPO_EXACT_DW_ENTROPY`，默认 1）。

统计结果显示两者无系统差异：

| 熵实现 | 稳定 | 崩溃 | peak 范围 |
|---|---|---|---|
| top-k 近似 | 1 | 3 | **73.3 - 82.7** |
| 全词表（论文） | 1 | 2 | 74.5 - 77.8 |

**top-k 同配置下 peak 跨度 9.4 点**（73.3 ↔ 82.7）。所以 82.7 是 4 条 top-k run 里唯一没崩的那条，属于运气。

（记录一次判断反复：我曾先后得出"全词表更差 8.2 点"和"全词表抗坍缩更好"两个相反结论，都是基于单次运行、且其中一次被资源争抢污染。正确结论是：样本太少、方差太大，区分不出来。）

### 4.3 FILL 的价值：抗坍缩，但会走向另一种失效

FILL 从未出现 baseline 那种"熵坍缩 + 长度爆炸"，因为它强制注入首 token 多样性、熵水平持续更高。

但 `f800-fill` 暴露了相反的失效模式：

| step | val | 长度 | 熵 |
|---|---|---|---|
| 450 | 79.6 | 704 | 2.58 |
| **480** | **81.8**（peak） | — | — |
| 550 | 81.2 | 434 | 3.71 |
| 650 | 75.9 | 107 | 4.83 |
| 700 | **65.7** | **61** | 3.28 |

**熵爆到 4.83、长度崩到 61 token**（已无法输出完整格式）。原因是首 token loss 用无上界的对数差、梯度恒为 1，长程训练下持续把冷门 token 往上推，`fill_coef=0.005` 的阻尼不足。

两种失效模式对照：

| | 熵 | 长度 | 结果 |
|---|---|---|---|
| baseline 失效 | → 0.08-0.39 | → 1027-1325 | 重复模板 / OOM |
| FILL 失效 | → 4.83 | → 61 | 格式崩坏 |

### 4.4 FILL 确实改变了首 token 分布

只有新版 loss（对数差）做到了，旧版（`−clamp(ρ)`，梯度 ∝ 概率）全程 `novel_frac ≡ 0`：

| run | 首 token loss | `novel_frac` | 种类 |
|---|---|---|---|
| 旧版 fill（w_ft=1） | `−clamp(ρ)` | **0.000** | 2 |
| 概率相减 | `p_top1 − p_forced` | 预期 0（梯度 4e-8） | — |
| **新版 fill** | **`log p_top1 − log p_forced`** | **0.43 - 0.75** | **7-9** |
| baseline（对照） | — | 0.000 | 2 |

原因：概率空间的 loss 梯度 `∝ p_forced`，冷门 token（p≈4e-8）推不动；对数空间梯度恒为 `1 − p_forced ≈ 1`。

实测 gap（64 条 chemistry prompt 逐条计算）：`To`/`The` 约 1.0（47%/63% 的 prompt 上为 0），冷门 slot **9-17 nats**。

---

## 5. 与论文的核对结论

### 已核实一致

超参 Table 3 全项、数据集划分（Chemistry 1890/210、Biology 450/50）、SDPO 路由 Table 7 全四行、teacher 构造（`π_θ` 本身 + 特权上下文，只重打分不重生成）、JSD/top-K/tail、§3.3 并集归一化、lr 调度、采样参数、以及原作者官方脚本 `experiments/generalization/run_sdpo_all.sh` + `sdpo.yaml` 的全部设置。

环境已对齐论文 B.1：**torch 2.8.0**、**SGLang 0.5.2**、8×H20。（CUDA 12.8 vs 论文 12.4，驱动 580 vs 550，不可降级。）

### 无法核实

- SRPO **无开源代码**（arXiv 无 code 链接，正文称 "plan to release"）
- 公开的 SDPO W&B 日志只有 Olmo-3-7B 的 run（熵 2.6-3.3、长度 166-423），模型不同不可直接比
- 原作者 W&B 显示其数据集名为 `sciknoweval/chemistry_filtered`，我们用 `chemistry`；本地无该变体，筛选规则未知

### 未复现的消融（论文 Table 2）

| 论文消融 | 状态 |
|---|---|
| SRPO w/o dynamic weighting（`dw_beta=0`） | ❌ 从未跑 |
| Advantage Mix（λ=0.9） | ❌ 代码未实现 |
| 五 benchmark 平均 | ❌ 只有 chemistry + biology |

注意论文 Table 2 的数值是**五项平均**，不能与我们的 chemistry 单项直接对照。

---

## 6. 代码改动记录

| 改动 | 开关 | 默认 |
|---|---|---|
| 动态加权用全词表熵（论文 §3.2） | `SRPO_EXACT_DW_ENTROPY` | 1（开） |
| §3.3 按 token 份额归一化 | `SRPO_UNION_NORM` | 1（开） |
| FILL 首 token 损失 = 对数差 | — | — |
| FILL 续写段 = CE，只在 t≠τ 取均值 | — | — |
| FILL 整体系数 | `fill_coef` | 1.0（实用 0.005） |
| 训练步数可覆盖 | `TOTAL_STEPS` | 800 |
| 数据集 / 候选池 / 引擎 / tag 可覆盖 | `DATA_PATH` / `CANDIDATE_POOL_PATH` / `ROLLOUT_ENGINE` / `RUN_TAG` | — |

**`fill_coef` 的定标依据**：去掉 `λ_fill` 后 fill 的 `grad_norm` 达 18.7（baseline 0.088），梯度裁剪（norm 1.0）会丢弃 95% 的更新。取 1/213 ≈ 0.005 使两分支梯度同量级，实测 `grad_norm` 回到 0.097-0.178。

---

## 7. 运维教训

1. **同机跑两条 run 会严重污染结果**：`v11-baseline` 与 `v11-fill` 相差 10 秒在同机启动，`time/step` 从 30s 涨到 133s，最终长度失控。SGLang 每卡起两个进程（`WorkerDict` + `sglang` server，后者占 60-76GB）。
2. **清理必须带 tag 限定**：`pkill -9 -f "main_ppo.*具体tag"`。裸的 `pkill -9 -f main_ppo` 或 `pkill -f run_local_srpo_v10.sh` 会杀掉同机所有 run —— `dw1-baseline` 就是这样在 step 233 被误杀的。
3. **启动后必须等显存回到 4 MiB 再起新 run**，否则 OOM。
4. **`nohup` 在 `cd && mkdir && nohup ... &` 链式调用中重定向会失效**，日志被丢弃。改用 `setsid` + 绝对路径。
5. 两台机器共享 JuiceFS 但各自独占 8 卡，**跨机器不互相影响**（只读模型/数据 + 极小日志写入）。

---

## 8. 待解决的核心问题

**长度为什么会失控？** 这是复现不到 83.0 的主因（5/7 条 baseline 因此崩溃），且随机触发。已排除：超参、数据集、路由逻辑、teacher 构造、环境、lr 调度、熵实现。原作者官方脚本同样**没有**任何长度或重复惩罚。

一个未查的方向：`max_response_length=8192` 是硬截断，需确认被截断的样本如何计算 reward 和优势——若截断样本没有受到惩罚，可能形成"越长越不被惩罚"的正反馈。

---

## 9. 正在进行

| | 机器 | 配置 | 目的 |
|---|---|---|---|
| `dwk0-baseline` | notebook-83040d10e701 | `EXACT_DW_ENTROPY=0`、800 步 | 验证 82.7 能否复现 |

另一台机器已回收，fill 对照暂停。

**判断依据**：step 80-120 的长度。若能压到 230 附近则有望复现；若往 350+ 走则大概率重演长度爆炸。按历史统计（4 条 top-k 中 1 条成功），成功率约 1/4。

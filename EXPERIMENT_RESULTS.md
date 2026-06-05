# SDCL_A_TOKEN 实验结果汇总（MATH 数据 / 2 卡）

> **本文件范围**:在 **MATH 训练集**(2 卡 DDP)上的实验记录,数据池小,fill_correct 占训练集 ~20%。
> **DeepMath-103K(100k 量级)/ 4 卡** 的实验另起一份记录:[`EXPERIMENT_RESULTS_DEEPMATH.md`](./EXPERIMENT_RESULTS_DEEPMATH.md)。两份各自独立、互不覆盖,断开会话后只读对应文件即可续上对应实验线的上下文。
>
> **2026-05-28 V2 干净口径(覆盖 V1)**:数据池重建为 `mistake_DS_MATH_pool.json` (**2025** 题) + `corr_DS_MATH_pool.json` (**5471** 题) = 7496 题。重建协议:max_prompt_length=6144 (vLLM 总窗口=2048+4096) / max_new_tokens=4096,baseline DeepSeek-R1-Distill-Qwen-7B 无 LoRA take_exam → teacher 判分 → 拆 mistake/corr。旧 V1 池(2079/5417)是 1024+4096 协议生成,含 ~54 道 prompt 截断假冤,已弃用,本次重建剔除。
> V1 训练集:`a_token_train_data.json` = corr_5417 + fill_correct_1390 = 6807 条样本(旧)
> V2 训练集:`a_token_train_data.json` = corr_5471 + fill_correct_1264 = **6735 条样本**(新,救回率 1264/2025=62.42%)
> Baseline V1(旧池+8k 评测):mistake 0/2079 = 0.00%,corr 5417/5417 = 100.00%,math500 ≈ 73.4%,all 5417/7496 = 72.27%
> Baseline V2(新池+4k 评测):mistake **13.83%**(280/2025),corr **94.30%**(5159/5471),math500 greedy 74.6%,**math500 roll-8 pass@1 = 73.98%**,all 72.56%(5439/7496)
> ⚠️ V1 / V2 口径不可横向比:V1 评测 max_new=8192 模型有 2× 推理空间,V2 max_new=4096 压缩到一半,长链推理题被截。V2 是"严格 4k 限制下 a_token 真实增量"的干净评测线。

### Baseline 论文口径 roll-8(2026-05-27)

> ⚠ 命令陷阱:`--max_prompt_length` 是 vLLM **总窗口(prompt+gen)**,不是 prompt 单独预算。
> 正确写法:`--max_prompt_length = prompt_budget + max_new_tokens`。math500 实际最长 prompt ~1300+,推荐 prompt_budget=2048。

| 配置 | max_new_tokens | --max_prompt_length | pass@1 avg | any@8 | all@8 | 备注 |
|---|---|---|---|---|---|---|
| **baseline**(无 LoRA),T=0.6 / top_p=0.95 / K=8 | **8192** | **10240** (2048+8192) | **85.95%** | 91.80% | 75.00% | ✅ **复现论文 85.8**,几乎完美对齐 |
| **β=0.7 ckpt**(方案 C),同口径 | 8192 | 10240 | **85.90%** | 91.40% | 74.40% | 与 baseline 差 −0.05/−0.40/−0.60pp,**统计上无显著差异**(roll-8 SE ~0.5pp) |
| 🔥 **β=0.0 ckpt**(方案 C 退化为纯 teacher KL),同口径 | 8192 | 10240 | **85.35%** | **92.60%** | 72.80% | 与 baseline −0.60/+0.80/−2.20pp;pass@1 持平,**any@8 涨**(覆盖更广)、**all@8 跌**(一致性弱) |
| ~~baseline @ "8k"~~(作废) | 8192 | 5120(被夹) | ~~78.88%~~ | ~~89.40%~~ | ~~62.80%~~ | 总窗口被夹到 5120,实际生成只有 ~4k,**作废** |
| baseline(待重跑) | 32768 | 33792 期望 | 89.90% (归属待核实) | 94.20% | 81.20% | 此前跑的 32k 命令未核实,可能也踩了 max_prompt 陷阱 |

> **关键结论(2026-05-28 更新)**:
> 1. 真·baseline @ 8k = 85.95,与论文 85.8 完美对齐,**论文复现正式成立**,后续 ckpt 都以 85.95 为基准。
> 2. **a_token 方案 C(β=0.7)在 math500 上是中性**,不涨不掉(差 −0.05pp 在 roll-8 噪声里)。
> 3. 🔥 **β=0.0(纯 teacher KL,无 anchor)在 math500 上也基本持平**,但在 mistake/corr 池上**大幅碾压**β=0.5/0.7/0.8(详见 §2.2)→ **anchor(fill_token 硬塞)非但不必要,反而拖后腿**。
> 4. 此前 89.90(声称 32k baseline)更可疑——既然训练在 8k 上中性,32k 不该突然涨 4pp。可能 32k 那次命令也踩了 max_prompt 陷阱,需用 `--max_prompt_length 34816 --max_new_tokens 32768` 重跑确认。

---

## 一、Loss 路径分类

| 路径 | 触发条件 | 说明 |
|---|---|---|
| **legacy CE + EMA** | `--beta_fill -1`(或代码不带 beta 字段) | fill 首 token 走 `F.cross_entropy(student_logits, fill_token_id)`,后段 KL,total loss = EMA(λ_ce·CE + λ_kl·KL_rest) |
| **方案 C(soft-teacher fill)** | `--beta_fill ≥ 0`(默认 0.5) | fill 首 token 走 `KL(student ‖ q')`,q'=(1-β)·teacher + β·onehot(fill_token_id);整段 loss 全是 KL |

---

## 二、实验对照(评测 = mistake_DS_MATH_pool + corr_DS_MATH_pool + math500)

### 2.1 Baseline 与 legacy CE 路径

| 实验 | β | epoch | mistake | corr | math500 | all | 备注 |
|---|---|---|---|---|---|---|---|
| baseline(无 LoRA) | — | — | 0.00% (0/2079) | 100.00% (5417/5417) | ~73.4% | 72.27% (5417/7496) | 基座参考 |
| legacy CE + EMA(各种 λ_ce/λ_kl 调参,代表轮次) | — | 3 | **21.84%** | 84.98% | 73.40% | — | corr 掉得严重(-15pp) |

> legacy CE 时代尝试过多组 `--lambda_ce / --lambda_kl`(0.95/0.05、0.5/0.5 等),mistake 大致都在 18-22% 区间,corr 普遍掉到 84-86%,math500 基本持平 baseline。这条线已被方案 C 取代。

### 2.2 方案 C(soft-teacher fill,纯 KL)

> 共同配置:DDP 2 卡,batch_size=6,gradient_accumulation_steps=3,effective_batch=36,lr=1e-5,gradient_checkpointing=on,max_answer=4096,seed 默认。
> 训练数据:6807 条(corr 5417 + fill 1390)。
> ⚠️ **训练侧 max_prompt_length 不同**:旧 β=0.5/0.7/0.8 用 1024,新 β=0.0 / β=0.7-new 用 2048(2026-05-28 同口径重训)。
> ⚠️ **评测侧口径不同**:旧 β=0.5/0.7/0.8 用 max_new_tokens=4096 / max_prompt=5120;新 β=0.0 / β=0.7-new 用 8192 / 10240(2026-05-28 同口径)。

| 实验目录                                       | β       | epoch | 训练 max_prompt | 评测 max_new | mistake               | corr                   | math500 (greedy)     | math500 pass@1 (T=0.6,n=8) | math500 any@8 | all                    | 训练 ep1/ep2 loss |
| ---------------------------------------------- | ------- | ----- | --------------- | ------------ | --------------------- | ---------------------- | -------------------- | -------------------------- | ------------- | ---------------------- | ----------------- |
| 🆕 `a_token_betaC_b07_math_20260527_214846`   | **0.7** | 2     | **2048(新)**   | **8192(新)** | **41.46%** (862/2079) | **97.05%** (5257/5417) | **82.4%** (412/500)  | **85.48%**                 | 92.40%        | **81.63%** (6119/7496) | — / 1.007         |
| 🔥 `a_token_betaC_b00_math_20260527_164851`   | **0.0** | 2     | **2048(新)**   | **8192(新)** | **43.34%** (901/2079) | **97.54%** (5284/5417) | **84.0%** (420/500)  | **85.35%**                 | **92.60%**    | **82.51%** (6185/7496) | — / 1.001         |
| `a_token_betaC_b05_20260525_110133`(旧)        | 0.5     | 2     | 1024(旧)        | 4096(旧)     | 19.87% (413/2079)     | 93.21% (5049/5417)     | 71.20% (356/500)     | 73.32%                     | 85.60%        | 72.87% (5462/7496)     | —                 |
| `a_token_betaC_b07_20260525_070630`(旧)        | 0.7     | 2     | 1024(旧)        | 4096(旧)     | 19.77% (411/2079)     | 93.69% (5075/5417)     | 75.60% (378/500)     | 73.32%                     | 85.60%        | 73.19% (5486/7496)     | 1.326 / 1.007     |
| `a_token_betaC_b08_4gpu_20260525_133031`(旧)   | 0.8     | 2     | 1024(旧)        | 4096(旧)     | 18.66% (388/2079)     | 92.95% (5035/5417)     | 74.40% (372/500)     | 73.48%                     | 85.60%        | 72.35% (5423/7496)     | —                 |

> **2026-05-28 重大发现 — 颠覆性修正**:β=0.7 同口径重训(prompt=2048 / 评测 8k)后,**几乎追平 β=0.0**(all 81.63 vs 82.51,差 0.88pp 在 noise 内)。
>
> **真正的关键变量是训练时 `--max_prompt_length`**:
> - 旧 β=0.7 (max_prompt=1024) → all 73.19%
> - 新 β=0.7 (max_prompt=2048) → all **81.63%**(+8.44pp)
> - 同样 β=0.7,只换训练 prompt 长度,从 73 → 81(+8pp 巨幅提升)
>
> **β 在 [0, 0.7] 区间对最终性能几乎无影响**(noise 范围内),anchor 设计的实际价值被高估。
>
> **新认知**:a_token 框架的真正价值不是 anchor 也不是 β 软化的 teacher KL,而是:
> 1. **fill 数据增强**:对 mistake 题用 fill prefix 引出教师正确推理后做 KD
> 2. **训练 prompt 充分长**:2048 vs 1024 决定了 ~8pp 性能差距
> 3. β 参数可以省略(走纯 teacher KD,代码更简单)

### 2.3 关键对比

**legacy CE(λ_ce=0.95/0.05,3ep)vs 方案 C(β=0.7-new,2ep,新口径)**

| 指标 | legacy 21.84/84.98 | 方案 C β=0.7-new(同口径) | 变化 |
|---|---|---|---|
| mistake | 21.84% | **41.46%** | **+19.62pp** ⬆⬆ |
| **corr** | 84.98% | **97.05%** | **+12.07pp** ⬆⬆ |
| math500 (greedy) | 73.40% | **82.4%** | **+9.00pp** ⬆ |
| **all 综合** | 65.65%* | **81.63%** | **+15.98pp** ⬆⬆ |

> *legacy 的 all 综合按 mistake_2079 + corr_5417 反算 ≈ 67.46% (不同时期池子有差异,数值仅供方向参考)

**β=0.0 vs β=0.7-new(同口径) — 修正后** ⭐

| 指标 | β=0.7-new | β=0.0 | β=0.0 - β=0.7-new |
|---|---|---|---|
| mistake | 41.46% | 43.34% | +1.88pp |
| corr | 97.05% | 97.54% | +0.49pp |
| math500 (greedy) | 82.4% | 84.0% | +1.60pp |
| math500 pass@1 (roll-8) | **85.48%** | 85.35% | -0.13pp(β=0.7 略高) |
| math500 any@8 | 92.40% | 92.60% | +0.20pp |
| **all 综合** | 81.63% | 82.51% | +0.88pp |

**结论**:同口径下 **β=0.0 vs β=0.7 几乎无显著差异**(roll-8 SE ~0.5pp)。**β 这个超参对最终性能不敏感**,a_token 设计的真正贡献来自数据流(fill 数据)和训练长度(prompt=2048),不是 β 项。

**口径影响估算(β=0.7 旧 → 新)**

| 指标 | 旧口径 | 新口径 | 变化 | 来源 |
|---|---|---|---|---|
| mistake | 19.77% | 41.46% | +21.69pp | 主要训练 prompt 1024→2048,部分评测 max_new 4096→8192 |
| corr | 93.69% | 97.05% | +3.36pp | 训练域更长,corr 推理更稳 |
| math500 greedy | 75.60% | 82.4% | +6.80pp | 评测 max_new 4096→8192 占大头 |
| math500 pass@1 | 73.32% | 85.48% | +12.16pp | 评测 max_new 4096→8192 决定性 |
| all 综合 | 73.19% | 81.63% | +8.44pp | 训练 prompt 长度为主 |

### 2.4 V2 干净口径对照(2026-05-28,新池 2025/5471 + 评测 6144+4096)

> **背景**:此前所有实验都在 V1 池(1024+4096 协议生成,含 prompt 截断假冤)+ 评测口径不一致(旧 max_new=4096 / 新 max_new=8192)中跑。2026-05-28 重建池(rebuild_math_pool.py)+ 训练评测全部统一在 max_prompt=2048 / max_new=4096 / max_model_len=6144 协议下。
>
> **训练命令**:`run_a_token_sdcl_train.py --max_prompt_length 2048 --max_answer_length 4096 --num_epochs 2 --batch_size 6 --gradient_accumulation_steps 3 --beta_fill <β>`(其余同上)
> **评测命令**:`main.py eval --max_prompt_length 6144 --max_new_tokens 4096 --math500_roll_k 8 --math500_roll_temperature 0.6 --math500_roll_top_p 0.95`

| 实验 | β | epoch | mistake (n/2025) | corr (n/5471) | math500 greedy | math500 pass@1 (roll-8) | math500 any@8 | math500 all@8 | all (n/7496) |
|---|---|---|---|---|---|---|---|---|---|
| **baseline V2**(无 LoRA) | — | — | 13.83% (280) | 94.30% (5159) | 74.6% | **73.98%** | 85.0% | 57.2% | 72.56% (5439) |
| 🆕 `a_token_betaC_b00_mathV2_20260528_115209` | **0.0** | 2 | **18.91%** (383) | 93.57% (5119) | 75.6% | 73.55% | 85.8% | 56.2% | **73.40%** (5502) |

**β=0.0-V2 vs baseline-V2(同口径,唯一变量是 LoRA)**:

| 指标 | Δ (LoRA − baseline) | 解读 |
|---|---|---|
| mistake | **+5.08pp** ✅ | LoRA 在"模型真错"子集多救 5pp,**方案 C 在 4k 限制下确实有效** |
| corr | −0.73pp | 几乎无污染,**对照 V1 旧 β=0.5/0.7/0.8 都掉 5-7pp 是质变** |
| all | **+0.84pp** ✅ | 净增量小但正向 |
| math500 greedy | +1.0pp | 单 trace 噪声大,仅参考 |
| math500 pass@1 | −0.43pp | **noise 内打平**,方案 C 在 math500 上仍然中性(与 V1 8k 口径下结论一致) |

**关键发现**:
1. **方案 C 在 mistake 池上有效是真实的**(+5.08pp),不是评测口径错觉
2. **β=0.0 corr 几乎无污染**(-0.73pp),印证"anchor=0 关闭后 corr 路径干净"——延续 V1 的 β=0.0 dominant 结论
3. **方案 C 在 math500 上中性**(-0.43pp roll-8)从 V1 8k 到 V2 4k 都成立
4. **方案 C 真实增量随推理预算增大**:V1 8k 口径下 LoRA 比 baseline 增量 +10-39pp,V2 4k 口径下只 +5pp。**长上下文是方案 C 真正发挥价值的场景**
5. **V2 mistake 池更硬**:baseline 在 V1 旧池 4k 评测能救 17.32%,V2 新池只能救 13.83%——重建剔除了"prompt 截断假冤"那部分容易救的题



### 优先级 0(2026-05-28 新增):**V2 干净口径补 β=0.7 对照**

V2 池 + V2 协议下,β=0.0 已跑(+5.08pp on mistake / -0.73pp on corr / -0.43pp on math500 roll-8)。需补 β=0.7-V2 验证:V1 时代"β 在 [0, 0.7] 不敏感"结论在 V2 干净池上是否仍成立。
- 若 β=0.7-V2 ≈ β=0.0-V2(差 < 1pp)→ β 不重要,写死 0.0 即可
- 若 β=0.7-V2 反超 β=0.0-V2 → V1 的"β=0.0 dominant"是 V1 池脏带来的特例,需重新评估 anchor 价值

### 优先级 1:**补扫低 β 区间**(2026-05-28 新优先级)

β=0.0 大幅 dominant 全部已测点(β=0.5/0.7/0.8),需确认最优 β 在 [0, 0.3]:

| β | 状态 | 目的 |
|---|---|---|
| **0.0** | ✓ 已完成,新最优 | 对照下界,意外 dominant |
| **0.1** | 待跑(高优先级) | 看 β 是否真正单调:β=0.1 ≈ β=0.0 → anchor 完全无效;β=0.1 < β=0.0 → anchor 即使弱量也是负作用 |
| **0.2** | 待跑 | 与 β=0.1 一起定 β-性能曲线左端形状 |
| 0.3 | 视 0.1/0.2 结果决定 | 如曲线在 [0.0, 0.2] 单调下降,可跳 |

### 优先级 2:更激进——彻底脱离 anchor

如 β=0.0 验证 anchor 无价值,试:
- **fill 数据完全不用 fill_token prefix**(只保留 mistake prompt + teacher 续写做 KL),看能否再涨
- **只用 corr 数据训(不要 fill)**:看 fill 贡献占比

### 优先级 3:epoch 扫描(β=0.0 锁定后)

- β=0.0 / 1 epoch:看是否欠拟合(2ep 已经很好,1ep 可能也够,可省一半时间)
- β=0.0 / 3 epoch:看 corr 会否过拟合掉头(legacy CE 3ep 栽过)

### 优先级 4:DeepMath 4 卡线对照

- 当前 DeepMath β=0.5 训练正在跑(2026-05-28),**很可能也是次优**
- 等评测出来,若同样发现 β=0.0 > β=0.5,DeepMath 线下一步必须补 β=0.0
- 详见 [`EXPERIMENT_RESULTS_DEEPMATH.md`](./EXPERIMENT_RESULTS_DEEPMATH.md)

### 优先级 5(旧):fill 数据增量

当前 fill_correct = 1390 / mistake_pool = 2079,救回率 66.9%,剩 689 道未救回。
若 β 扫完 mistake 卡在 20%,需要回去:
```bash
python main.py pipeline --skip-train --fill_epoch 5
```
继续扩 fill_correct.json,理论上每多 100 道 mistake 评测分能再涨 ~5pp。

---

## 四、文件 / 目录索引

| 路径 | 含义 |
|---|---|
| `datasets/exam/mistake_DS_MATH_pool.json` | **1419** 题(2026-06-04,2048+8192 口径重建),模型原本错的题。历史:V2 6144+4096 口径 2025 题 / V1 2079 题 |
| `datasets/exam/corr_DS_MATH_pool.json` | **6077** 题(2026-06-04,2048+8192 口径重建),模型原本对的题。历史:V2 5471 题 / V1 5417 题 |
| `datasets/exam/fill_multi_pool.json` | **1221** 题救回(2026-06-05,2048+8192 逐个 fill,救回率 1221/1419 = 86.05%),每题 candidates avg 138.75 / median 126。历史 V2 fill_correct=1264(62.42%) |
| `datasets/exam/fill_multi_unresolved.json` | **198** 题(376 首 token 全试仍做不对的硬题) |
| `datasets/exam/a_token_train_data.json` | **6735** 条(corr 5471 + fill 1264),V2 训练数据 |
| `scripts/rebuild_math_pool.py` | **2026-05-28 新增**:一次性脚本,用 6144+4096 干净口径重建 mistake/corr 池(take_exam → teacher 判分 → 备份+覆盖) |
| `scripts/rebuild_math_pool_8k.py` | **2026-06-04**:用 2048+8192 口径重建 mistake/corr 池(同流程,max_new=8192) |
| `scripts/build_fill_pool_token.py` | 对 mistake/unresolved 池逐个强制 fill 整个首 token 池(376),T=0 greedy,收 boxed 对的 candidates |
| `datasets/first_tokens_train.json` | 训练集首 token 统计:7496 solutions / 299 unique tokens(top3:`We` 1356 / `The` 1035 / `Let` 869) |
| `output/a_token_betaC_b07_20260525_070630/` | β=0.7 训练产物(checkpoint + train.log + step_metrics.jsonl + beta_fill_log.jsonl) |
| `output/a_token_betaC_b05_20260525_110133/` | β=0.5 训练产物(运行中) |
| `output/eval_lora_<ts>/summary.json` | 评测汇总(mistake/corr/math500/all) |

---

## 五、变更历史

| 日期 | 内容 |
|---|---|
| 2026-05-24 之前 | legacy CE + EMA 多轮调参(λ_ce/λ_kl),mistake 18-22% / corr 84-86% |
| 2026-05-24 | 方案 C 实现并合入,引入 `--beta_fill`(默认 0.5),fill 首 token 切到 KL,加 `beta_fill_log.jsonl` 探针日志 |
| 2026-05-25 早 | β=0.7 / 2ep 训完,首次方案 C 完整结果 mistake 19.77 / corr 93.69 / all 73.19 |
| 2026-05-25 中 | 启动 β=0.5 / 2ep 对照实验(2 卡) |
| 2026-05-28 早 | prompt 长度统一 1024 → 2048(11 处),β=0.0 对照训完,口径分解后发现 β 不敏感、训练 prompt 长度才决定性能 |
| 2026-05-28 晚 | **V2 干净口径重做**:rebuild_math_pool.py 用 6144+4096 重建 mistake/corr 池(2025/5471),fill_correct=1264,train_data=6735;baseline V2 + β=0.0-V2 训完评完(同口径,LoRA 净增量 +0.84pp on all,+5.08pp on mistake,roll-8 持平 baseline) |
| 2026-06-04 | **2048+8192 新口径重建 corr/mistake 池**:rebuild_math_pool_8k.py(take_exam max_prompt=10240=2048+8192, max_new=8192 → teacher 判分)。结果 corr=6077 / mistake=1419 / total=7496 acc=81.07%(对比 4096 口径 corr 5456 / mistake 2040,加倍 gen 后 621 题从 mistake 转入 corr) |
| 2026-06-05 | **mistake 池逐个 fill 首 token(2048+8192)**:build_fill_pool_token.py 对 1419 题 mistake 池每题 × 376 首 token 池(first_tokens_test.json)逐个强制 fill,T=0 greedy 续写 ≤8192,boxed 对的收 candidates。结果 **fill 救回 1221/1419 = 86.05%**,unresolved 198(13.95%);candidates/题 min=1 max=374 avg=138.75 median=126;耗时 877 min(4 卡)。产物覆盖 fill_multi_pool.json / fill_multi_unresolved.json |

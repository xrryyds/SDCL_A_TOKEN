# SDCL_A_TOKEN 实验结果汇总（MATH 数据 / 2 卡）

> **本文件范围**:在 **MATH 训练集**(2 卡 DDP)上的实验记录,数据池小,fill_correct 占训练集 ~20%。
> **DeepMath-103K(100k 量级)/ 4 卡** 的实验另起一份记录:[`EXPERIMENT_RESULTS_DEEPMATH.md`](./EXPERIMENT_RESULTS_DEEPMATH.md)。两份各自独立、互不覆盖,断开会话后只读对应文件即可续上对应实验线的上下文。
>
> 数据池:`mistake_DS_MATH_pool.json` (2079 题) + `corr_DS_MATH_pool.json` (5417 题) = 7496 题
> 训练集:`a_token_train_data.json` = corr_5417 + fill_correct_1390 = 6807 条样本
> Baseline (基座 DeepSeek-R1-Distill-Qwen-7B,无 LoRA):mistake 0/2079 = 0.00%,corr 5417/5417 = 100.00%,math500 ≈ 73.4%,all 5417/7496 = 72.27%
> 注意:旧版 RUN_COMMANDS.md 写的 baseline `all=52.51%` 对应分母 `corr=3936`,那是 corr_answer_4096.json 时代的数据,现在 corr 池换成 5417 之后 baseline 全量 = 5417/7496 = **72.27%**

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

> 共同配置:DDP 2 卡,batch_size=6,gradient_accumulation_steps=3,effective_batch=36,lr=1e-5,gradient_checkpointing=on,max_prompt=1024(β=0.0 用 2048),max_answer=4096,seed 默认。
> 训练数据:6807 条(corr 5417 + fill 1390)。

| 实验目录                                       | β       | epoch | mistake               | corr                   | math500 (greedy)     | math500 pass@1 (T=0.6,n=8) | math500 any@8 | all                    | 训练 ep1/ep2 loss |
| ---------------------------------------------- | ------- | ----- | --------------------- | ---------------------- | -------------------- | -------------------------- | ------------- | ---------------------- | ----------------- |
| 🔥 `a_token_betaC_b00_math_20260527_164851`   | **0.0** | 2     | **🔥 43.34%** (901/2079) | **🔥 97.54%** (5284/5417) | **🔥 84.0%** (420/500) | **85.35%**                 | **92.60%**    | **🔥 82.51%** (6185/7496) | — / 1.001         |
| `a_token_betaC_b05_20260525_110133`            | **0.5** | 2     | 19.87% (413/2079)     | 93.21% (5049/5417)     | 71.20% (356/500)     | 73.32%                     | 85.60%        | 72.87% (5462/7496)     | —                 |
| `a_token_betaC_b07_20260525_070630`            | **0.7** | 2     | 19.77% (411/2079)     | 93.69% (5075/5417)     | 75.60% (378/500)     | 73.32%                     | 85.60%        | 73.19% (5486/7496)     | 1.326 / 1.007     |
| `a_token_betaC_b08_4gpu_20260525_133031`       | **0.8** | 2     | 18.66% (388/2079)     | 92.95% (5035/5417)     | 74.40% (372/500)     | 73.48%                     | 85.60%        | 72.35% (5423/7496)     | —                 |

> **β=0.0 是新最优(2026-05-28)**:全面 dominant,mistake 翻倍(19.77→43.34)、corr 涨(93.69→97.54)、greedy math500 涨(75.60→84.0)、all 综合涨 +9.32pp(73.19→82.51)。pass@1 / any@8 与 baseline 持平,但 all@8 略低(72.80 vs 75.00)说明输出一致性弱、多样性高。

### 2.3 关键对比

**legacy CE(λ_ce=0.95/0.05,3ep)vs 方案 C(β=0.7,2ep)**

| 指标 | legacy 21.84/84.98 | 方案 C β=0.7 19.77/93.69 | 变化 |
|---|---|---|---|
| mistake | 21.84% | 19.77% | -2.07pp |
| **corr** | 84.98% | **93.69%** | **+8.71pp** ⬆ |
| math500 (greedy) | 73.40% | 75.60% | +2.20pp |
| **all 综合** | 65.65%* | **73.19%** | **+7.54pp** ⬆ |

> *legacy 的 all 综合按 mistake_2079 + corr_5417 反算 = (2079·0.2184 + 5417·0.8498)/7496 ≈ 67.46% (不同时期池子有差异,数值仅供方向参考)

**β=0.0(无 anchor) vs β=0.7(有 anchor)** ⭐ 2026-05-28 重要新发现

| 指标 | β=0.7 | **β=0.0** | 变化 |
|---|---|---|---|
| mistake | 19.77% | **43.34%** | **+23.57pp** ⬆⬆ |
| corr | 93.69% | **97.54%** | **+3.85pp** ⬆ |
| math500 (greedy) | 75.60% | **84.0%** | **+8.40pp** ⬆ |
| math500 pass@1 (roll-8) | 85.90% | 85.35% | -0.55pp(noise 内) |
| **all 综合** | 73.19% | **82.51%** | **+9.32pp** ⬆⬆ |

**结论**:**anchor(fill_token 硬塞)是负作用源**,不是 carry。
- 旧认知错误:"必须强塞 fill_token 才能救回 mistake 题"
- 新认知:**纯 teacher KL on fill 数据**才是真正起作用的部分,β>0 反而干扰 student 学教师软分布
- 可能机制:(1) fill_token 不同题不同,强塞引入位置敏感的、与上下文无关的 anchor 行为,污染 student 语言模型分布;(2) anchor 通过共享 LoRA 权重泄漏到 corr 路径,β>0 时 corr 也从 100→93;(3) β=0.5 时 q_mix 是分裂的"双峰",哪边都拟不好

---

## 三、待跑实验队列(按优先级)

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
| `datasets/exam/mistake_DS_MATH_pool.json` | 2079 题,模型原本错的题(评测 mistake 集 / 训练 fill 输入) |
| `datasets/exam/corr_DS_MATH_pool.json` | 5417 题,模型原本对的题(评测 corr 集 / 训练 corr 输入) |
| `datasets/exam/fill_correct.json` | 1390 题,fill 后能救回的题 |
| `datasets/exam/a_token_train_data.json` | 6807 条(corr 5417 + fill 1390),训练数据 |
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

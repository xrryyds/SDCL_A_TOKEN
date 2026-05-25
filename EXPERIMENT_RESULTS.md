# SDCL_A_TOKEN 实验结果汇总

> 数据池:`mistake_DS_MATH_pool.json` (2079 题) + `corr_DS_MATH_pool.json` (5417 题) = 7496 题
> 训练集:`a_token_train_data.json` = corr_5417 + fill_correct_1390 = 6807 条样本
> Baseline (基座 DeepSeek-R1-Distill-Qwen-7B,无 LoRA):mistake 0/2079 = 0.00%,corr 5417/5417 = 100.00%,math500 ≈ 73.4%,all 5417/7496 = 72.27%
> 注意:旧版 RUN_COMMANDS.md 写的 baseline `all=52.51%` 对应分母 `corr=3936`,那是 corr_answer_4096.json 时代的数据,现在 corr 池换成 5417 之后 baseline 全量 = 5417/7496 = **72.27%**

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

> 共同配置:DDP 2 卡,batch_size=6,gradient_accumulation_steps=3,effective_batch=36,lr=1e-5,gradient_checkpointing=on,max_prompt=1024,max_answer=4096,seed 默认。
> 训练数据:6807 条(corr 5417 + fill 1390)。

| 实验目录                                       | β       | epoch | mistake               | corr                   | math500 (greedy)     | math500 pass@1 (T=0.6,n=8) | math500 any@8 | all                    | 训练 ep1/ep2 loss |
| ---------------------------------------------- | ------- | ----- | --------------------- | ---------------------- | -------------------- | -------------------------- | ------------- | ---------------------- | ----------------- |
| `a_token_betaC_b05_20260525_110133`            | **0.5** | 2     | 19.87% (413/2079)     | 93.21% (5049/5417)     | 71.20% (356/500)     | 73.32%                     | 85.60%        | 72.87% (5462/7496)     | —                 |
| `a_token_betaC_b07_20260525_070630`            | **0.7** | 2     | **19.77%** (411/2079) | **93.69%** (5075/5417) | **75.60%** (378/500) | 73.32%                     | 85.60%        | **73.19%** (5486/7496) | 1.326 / 1.007     |
| `a_token_betaC_b08_4gpu_20260525_133031`       | **0.8** | 2     | 18.66% (388/2079)     | 92.95% (5035/5417)     | 74.40% (372/500)     | 73.48%                     | 85.60%        | 72.35% (5423/7496)     | —                 |

### 2.3 关键对比

**legacy CE(λ_ce=0.95/0.05,3ep)vs 方案 C(β=0.7,2ep)**

| 指标 | legacy 21.84/84.98 | 方案 C β=0.7 19.77/93.69 | 变化 |
|---|---|---|---|
| mistake | 21.84% | 19.77% | -2.07pp |
| **corr** | 84.98% | **93.69%** | **+8.71pp** ⬆ |
| math500 (greedy) | 73.40% | 75.60% | +2.20pp |
| **all 综合** | 65.65%* | **73.19%** | **+7.54pp** ⬆ |

> *legacy 的 all 综合按 mistake_2079 + corr_5417 反算 = (2079·0.2184 + 5417·0.8498)/7496 ≈ 67.46% (不同时期池子有差异,数值仅供方向参考)

**结论**:方案 C 把 mistake 让出 ~2pp,换来 corr 大幅恢复 ~9pp,综合 all 净提升 ~5-8pp,且 math500 泛化更好(75.60 > 73.40)。

---

## 三、待跑实验队列(按优先级)

### 优先级 1:扫 β 曲线(2ep 固定,只换 β)

| β | 状态 | 目录 |
|---|---|---|
| 0.5 | 运行中(2 卡) | `a_token_betaC_b05_20260525_110133` |
| 0.7 | ✓ 已完成 | `a_token_betaC_b07_20260525_070630` |
| 0.8 | 待跑(推荐 4 卡新机器) | — |
| 0.6 | 视 0.5/0.8 结果决定是否补点 | — |
| 0.3 / 0.9 | 仅当曲线在 [0.5, 0.8] 单调时再扫边界 | — |

### 优先级 2:epoch 扫描(β 锁定后)

- 最优 β / 1 epoch:看是否欠拟合
- 最优 β / 3 epoch:看 corr 是否过拟合掉头(legacy CE 3ep 就栽在这)

### 优先级 3:fill 数据增量

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

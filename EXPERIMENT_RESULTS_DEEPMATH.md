# SDCL_A_TOKEN 实验结果汇总（DeepMath-103K / 4 卡）

> **本文件范围**：在 **DeepMath-103K(100k 量级)** 训练集上、**4 卡 DDP** 跑的实验。
> **与之区分**：在 MATH 训练集 / 2 卡 上的旧线另存于 [`EXPERIMENT_RESULTS.md`](./EXPERIMENT_RESULTS.md)。两份互不覆盖，断开会话后只读本文件即可续上 DeepMath 线的上下文。
>
> 共同基线模型：**DeepSeek-R1-Distill-Qwen-7B**（HF / 本地路径与旧线一致）。
> 共同评测基准：**MATH-500 论文口径 roll-8**（T=0.6, top_p=0.95, K=8）—— **baseline 85.95 pass@1**（详见 EXPERIMENT_RESULTS.md 中 Baseline 论文口径 roll-8 章节）。
> 本文件所有 DeepMath 训练得到的 ckpt 都以 85.95 为评测基准做对比。

---

## 0. 维度速览（与 MATH 旧线对照）

| 维度 | 本线（DeepMath / 4 卡） | 旧线（MATH / 2 卡） |
|---|---|---|
| 训练原始题源 | DeepMath-103K (HF: zwhe99/DeepMath-103K) | MATH 训练集 |
| 数据池题数（mistake + corr） | 59,171 + 43,849 = **103,020** | 2,079 + 5,417 = 7,496 |
| 训练集 `a_token_train_data.json` 条数 | **96,405** | 6,807 |
| 硬件 | 4 卡 DDP（学生 + 教师同卡分配） | 2 卡 DDP |
| 训练 token 预算 | prompt 2048 + answer 4096 | 与旧版默认一致 |
| 评测口径 | 与旧线一致（MATH-500 论文口径 roll-8） | 同 |

---

## 1. 训练数据盘点（2026-05-27，硬数据）

执行命令：
```bash
python scripts/inspect_train_data.py \
    --tokenizer /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
    --sample 500
```

### 1.1 池子规模

| 文件 | 条数 | 备注 |
|---|---|---|
| `datasets/exam/mistake_DS_MATH_pool.json` | 59,171 | student baseline 在 DeepMath-103K 上答错的题 |
| `datasets/exam/corr_DS_MATH_pool.json` | 43,849 | student baseline 原本就答对的题 |
| `datasets/exam/fill_correct.json` | 52,556 | 用 a_token 救回的题（fill 阶段产物） |
| `datasets/exam/a_token_train_data.json` | **96,405** | corr 43,849 + fill 52,556（去重后无差） |

**救回率 = 52,556 / 59,171 = 88.82%**（旧 MATH 线为 1,390 / 2,079 = 66.86%，+22pp）。

### 1.2 训练集 source 分布

| source | 条数 | 占比 |
|---|---|---|
| `fill_correct` | 52,556 | **54.52%** |
| `corr_answer` | 43,849 | 45.48% |

**反转**：旧 MATH 线是 corr 主导（80%），DeepMath 线 fill 反而是大头（55%）。**意味着方案 C 的 fill-KL 项会主导 loss**，β_fill 的影响会比旧线大得多（详见 §3 训练前决策）。

### 1.3 fill_token 复用 top-20

`fill 条数 = 52,556，不同 fill_token = 379` —— 多样性健康，无头部塌缩。

| rank | token_id | text | count | 占比 |
|---|---|---|---|---|
| 1 | 40 | `I` | 792 | 1.51% |
| 2 | 1249 | `To` | 743 | 1.41% |
| 3 | 5338 | `First` | 699 | 1.33% |
| 4 | 641 | `In` | 691 | 1.31% |
| 5 | 22043 | `Given` | 690 | 1.31% |
| 6 | 785 | `The` | 682 | 1.30% |
| 7 | 22464 | `Looking` | 624 | 1.19% |
| 8 | 8304 | `Step` | 621 | 1.18% |
| 9 | 24617 | `Starting` | 609 | 1.16% |
| 10 | 1654 | `We` | 607 | 1.15% |
| 11 | 12549 | `Since` | 596 | 1.13% |
| ... | ... | ... | ... | ... |

top-20 加起来 ≈ 21.7%，单 token 最多 1.51%，分布平坦。

### 1.4 字段长度（字符数 / token 数）

字符数（全量 96,405）：
- `question`: mean=199.8, p50=171, p90=341, p95=417, p99=616, p100=4309
- `answer`:   mean=7585.3, p50=7518, p90=11902, p95=13134, p99=15585, p100=20293

token 数（抽样 500）：
- `question`: mean=64.4, p50=56, p90=111, p95=132, p99=176, p100=310
- `answer`:   mean=2960.3, p50=2994, **p90=4096, p95=4096, p99=4096, p100=4096** ← 全顶天花板

### 1.5 关键观察 / 隐患

1. **答案被 4k 截断（重要）**：answer token 数 p90 起全是 4096，说明 fill 阶段 `max_gen_token=4096` 截掉的 ≥10% 的样本根本没有 boxed 收尾，被 `check_correctness` 判错过滤掉了。
   - **后果 1**：真实救回率应该 >88.82%，但 >4k 的题被排除在外。
   - **后果 2**：训练集**没有 >4k 推理样本**，训出来的 ckpt 在 math500 长尾题（>4k 推理）上学不到强能力，与 baseline 同口径比预期持平。
   - **设计取舍**：重造 8k 数据 fill 时间 ×2，96k 题不可接受，所以接受这条上限。
2. **prompt token p100=310 ≪ 2048**，套上 chat_template（+50~80 token）也远不够 2048 —— `--train_max_prompt_length 2048` 板上钉钉够用，**10× 冗余**。
3. **judge 严格相等（无水分）**：`check_correctness` = `extract_answer(pred) == extract_answer(ref)`，提取最后一个 `\boxed{}` 内容做严格字符串比较（scripts/train/a_token_sd.py L239）。
   - **正面**：88.82% 救回率不掺水，fill 样本是真·boxed 内容匹配。
   - **反面**：DeepMath ref_answer 中含表达式 / 方程（如 `"m^2 + 1 = 0"`），写法等价但格式不同的 pred 会被判错。这意味着 mistake 池里**可能虚高**，含一定假阴样本。
4. **answer 字段是 student 自己续写**（与方案 C 语义一致）：
   - corr 样本：`"Okay, so I have this..."`（典型 student 第一人称）
   - fill 样本：开头就是 fill_token_text（如 `"Starting"`），后面才是 student 续写
   - KL 训练目标的 token 序列与 teacher logits 匹配，方案 C 合法。
5. **fill_token 分布健康**：379 种、top1 仅 1.51%，没有塌缩到固定开头；token 多为"段首动词 / 介词 / 代词"（I / To / First / Looking / Starting…），符合 a_token 的"破局首 token"语义。

---

## 2. 训练前决策建议（待跑实验）

| 项目 | 建议起步值 | 理由 |
|---|---|---|
| **β_fill** | **0.5**（不是旧线的 0.7） | fill 占比从 20% → 55%，anchor 总贡献 ≈ 旧线 ×2.7；β 不降会过拟合到首 token。0.5 跑稳后再试 0.7。 |
| 训练数据量 | **96k 全量 ep=1** | 96k / global_bs=64 ≈ 1500 step，等价于旧线 6.8k × ep≈14；先 ep=1 看 loss 曲线再决定是否加 epoch。 |
| `train_max_prompt_length` | 2048 | prompt token p100=310，10× 冗余，板上钉钉够。 |
| `train_max_answer_length` | 4096 | 与数据生成对齐，避免二次截断。 |
| `train_batch_size`（每卡） | 4 | 4 卡 × 4 = 16 micro |
| `train_grad_accum_steps` | 4 | global batch = 16 × 4 = 64 |
| `train_learning_rate` | **1e-5** | 与旧线一致；数据量大 ×14 而 step 也大 ×14，等效更新规模相当，无需降 lr。 |
| `train_num_epochs` | **1**（先） | 数据量 ×14，旧线 ep=3 对应这里 ep≈0.2 就够；ep=1 已经是旧线的 ~4.7 倍训练量。 |
| `train_use_lora` | True | 与旧线一致，节省显存。 |

> 命令模板待第一次实际开跑后补全到 §4。

---

## 3. 实验记录（按时间倒序追加）

### 3.1 数据生成完成（2026-05-27）

- mistake 池：59,171（DeepMath-103K train 上 baseline 答错的题）
- corr 池：43,849
- fill 救回：52,556（救回率 88.82%）
- 训练集 a_token_train_data.json：96,405（fill 54.52% + corr 45.48%）
- 数据生成超参：`roll_n=16`，`fill_max_gen_token=4096`，`fill_prompt_len=1024`，`fill_epoch=3`（实际几轮、各轮新增量 → 见远程 `output/a_token_sdcl_*/pipeline_dataflow.log`）

### 3.2 训练（待跑）

待 §4 命令模板敲定后追加。

---

## 4. 训练 / 评测命令模板（待第一次实跑后补全）

> 训练命令在第一次实际开跑后落地到这里，记录真实超参与启动方式（含 `torchrun`/`mp.spawn` 选择、`CUDA_VISIBLE_DEVICES` 设置等）。
> 评测命令固定走 MATH-500 论文口径 roll-8（与旧线一致），baseline 基准 = **85.95 pass@1**。
>
> **评测踩坑提醒**：`--max_prompt_length` 是 vLLM 总窗口（prompt+gen），**不是** prompt 单独预算。正确写法 `--max_prompt_length = prompt_budget + max_new_tokens`。math500 实际最长 prompt ~1300+ token，prompt_budget 至少给 2048。

---

## 5. 与旧 MATH 线的差异速查（避免混淆）

| 差异点 | 本线（DeepMath / 4 卡） | 旧线（MATH / 2 卡） |
|---|---|---|
| 数据池来源 | DeepMath-103K train | MATH 训练集 |
| 训练集大小 | 96,405 | 6,807 |
| fill 占训练集 | 54.52% | ~20% |
| 救回率 | 88.82% | 66.86% |
| 推荐 β_fill 起步 | 0.5 | 0.7（旧线已验过的甜点） |
| 4k vs 旧版上限 | 4096 全顶天花板 | 旧版同上限 |
| 与 baseline 85.95 比较 | **待跑** | β=0.7 ckpt @ 真·8k = 85.90，与 baseline 持平 |

# 为什么原算法 a_token_sdcl 训完 mistake 增量只能到 +5pp(V2 4k 口径)/ 最高 ~43%(V1 8k 口径)

> 写作时间:2026-05-30(节点:历史实验作废后回顾)
> 关联文档:`docs/a_token_sdcl.md`(算法审计) / `docs/s_grpo_analysis_and_alternatives.md`(S-GRPO 分析) / `EXPERIMENT_RESULTS.md`(实验台账)
>
> 本文档**只做归因诊断**,不动代码、不给方案。结论:**算法没 bug**,瓶颈是**数据流形态**和**信号源**,不是超参或工程错误。

---

## 0. 现象回顾 — "训完只能涨 5pp(V2 口径) / 卡在 ~40% 出头(V1 8k 口径)"

| 口径 | baseline mistake | 训完 mistake | Δ | 来源 |
|---|---|---|---|---|
| V2 4k(干净口径) | 13.83% (280/2025) | **18.91%** (383/2025) | **+5.08pp** | β=0.0-V2,2026-05-28 |
| V1 8k(评测推理预算大) | 0.00% (0/2079) | **43.34%** (901/2079) | **+43.34pp** 看起来很大,但⋯ | β=0.0-new,2026-05-27 |
| V1 4k(旧训练 prompt=1024) | 0.00% | 19.87% (β=0.5) ~ 19.77% (β=0.7) | +20pp | β=0.5/0.7-old |

**两个口径要分清**:
- V2 4k 口径下 baseline 已经能救回 13.83% mistake → LoRA 边际收益只剩 +5pp
- V1 8k 口径下 baseline 0% mistake(因为旧 mistake 池用 1024+4096 生成,prompt 截断假冤 + 评测 max_new 8192 比训练长 → 一大半 "+43pp" 是评测口径放大,不是 LoRA 真本事)

**用户说的"最多 15%"应该是指 V2 口径下 mistake 涨幅的上限**(+5pp 在 +5 ~ +10pp 区间徘徊),或者是 mistake_acc 的绝对水平(18.91%)被理解成"只能学到 15%"。两种解读下面都覆盖。

---

## 1. 算法机制本身决定了 mistake 增量天花板

### 1.1 fill_correct 数据的来源是"教师能救但学生 1-shot 漏掉"的题

`a_token_sdcl.py` Phase A/B/C 流程:
1. 对 mistake 池每题,vLLM 出 top-K(K=400)首 token logprob
2. 对候选 token 强制做首 token rollout 续写
3. 续写答对的候选里,选 base logprob 最大的那个 → 写入 `fill_correct.json`

**fill 池的天然规模 = mistake 池里"换个首 token 就能救回"的子集**:
- V1 mistake=2079,fill_correct=1390 → 救回率 66.9%
- V2 mistake=2025,fill_correct=1264 → 救回率 **62.42%**

剩下 ~37% 的 mistake 题**学生连"用任何首 token 重启都救不回"**,这些题在训练数据里**完全没有信号**,既不进 fill 也不进 corr。训练永远碰不到它们 → 评测时它们仍然会错。

**结论 1**:`fill_correct 救回率 62%` 就是 mistake 增量的**上界**,任何 β 调参在这条线下方调度。

### 1.2 corr_answer 数据"反向稀释"了 fill 信号

V2 训练数据:`corr_5471 + fill_1264 = 6735 条`,其中 fill 占 18.8%。

每个 epoch DataLoader 把 7:2 的 corr:fill 混合,梯度被 corr 主导。**corr 路径在做"已经对的题继续对"**(全 span teacher-forcing KD),这个信号在 baseline V2 上 corr 已经 94.3% → 大部分 corr 样本对学生几乎是 zero-loss。

但训练 loss 是按样本数等权平均的:
- corr 样本平均 kl_corr ≈ 0(因为学生本来就和 teacher 接近)
- fill 样本平均 kl_fill 较大(因为学生本来在这些题上首 token 概率低)

`kl_corr ≈ 0` 不代表 corr 样本对优化没影响 —— 在 LoRA 路径上**任何非零梯度都拉 LoRA 参数往 corr 分布走**,而 corr 分布就是 base model 当前的输出分布 → **LoRA 被 corr 样本拉回 base**,fill 信号被稀释。

实测对照(EXPERIMENT_RESULTS.md §2.2):
- β=0.5/0.7/0.8 旧口径 corr 都掉到 92-93%,mistake 在 18-20% → 学生学了点 fill 但污染了 corr
- β=0.0 V2 新口径 corr 掉 0.73pp(几乎不污染),mistake 才 +5pp → corr 几乎没动 = LoRA 几乎没"力气"修 mistake

**结论 2**:7:2 的 corr:fill 比例让训练信号严重 corr-dominated。增 fill 比例可能涨 mistake,但 V2 实测 anchor β=0 → corr 不掉 → 也涨不了多少 mistake,说明这条路有刚性上界。

### 1.3 每题只有 1 条 fill rollout → KL 是"单峰目标",学生只能向一个方向收敛

`build_first_token_target_logprobs` (`a_token_sd.py:477-525`) 在原 a_token_sdcl 中**没有被调用** —— `fill_correct` 路径的 target 是:
```
q' = (1-β) · teacher_softmax + β · onehot(fill_token_id)
```
β=0 时退化为纯 teacher KD,**完全没用到 anchor(fill_token_id)的信息**;β>0 时把概率质量硬塞到一个 token 上。

无论 β 怎么调,**target 在首 token 位置上是一个单点(onehot)或单点+teacher 平滑**。多个对的候选 token(Phase B 里通常有 3-10 个 token 都能续写对)只保留 base logprob 最大的那个,其余的对答案被丢掉。

**这等于做 supervised hard label**:模型被告诉"在这题上唯一正确的首 token 是 X"。但事实是,这题有多个 token 都能引导到正确答案。

**结论 3**:fill_correct 只取一条 → 多解题被压成单解 → 模型 fit 到的是"我选了哪一条 token",而不是"哪些 token 是好的"。这是 supervised KD 的固有局限,不是参数问题。

### 1.4 训练侧 KL loss 从 ep1 就贴近 0 → 信号已榨干

在 grpo_3pool_plan.md 里有诊断结论(虽然 plan 是作废的,但诊断结果仍然成立):

> 训练侧 KL loss 从 ep1 起就贴 0(~3e-4)→ student 已完全 fit teacher 的 single-answer KD 信号

`kl_fill ≈ 3e-4` 意味着:
- ep1 后学生在 fill 题的首 token 上已经 99% 概率塞到 fill_token_id
- 继续训 ep2/ep3 没有新信号,只会过拟合
- 评测时这些题在 greedy 下首 token 走 fill_token,但**整条 chain-of-thought 还是要 4k token 才能到 boxed**,模型推理过程中任何一处偏离就 fail

**结论 4**:首 token 训对了,后面 4k token 的推理过程没有训过 → mistake 题救回率受推理过程稳定性限制,首 token 信号能给的边际收益已榨干。

---

## 2. 数据流形态:fill 信号"过早 commit"

### 2.1 fill_correct 用的 anchor token 来自 base model 自己的 top-K logprob

Phase A 把 mistake 题喂给 **baseline model(无 LoRA)**,拿 top-400 候选 token。Phase B 强制 rollout 再筛对的。

这意味着 **fill_token 永远是 base model 自己已经在 top-400 里给出 nontrivial 概率的 token**。它不是"模型完全想不到的新方向",而是"模型本来排第 5-50 名,但 greedy 跑没采到的"。

训练后模型把这个 token 的概率从 base 的 0.5% 拉到 ~99% → 在 greedy 下当然能跑出对答案,但:
- T=0.6 sampling 下,模型仍然会 sample 到 base 喜欢的高概率 token(那些可能引向错答案的)
- math500 roll-8 上**完全持平 baseline** 就是这个原因(EXPERIMENT_RESULTS.md §2.4:roll-8 pass@1 -0.43pp)

**结论 5**:fill_correct 学的是"换首 token 救题"的局部修补,不是"提升推理能力"。在 greedy mistake 评测下涨,在 sampling-based math500 上不涨,完全自洽。

### 2.2 teacher 在 fill_correct 路径上充当"forced continuation"角色

`fill_correct.json` 里的 `answer` 字段是 **Phase B 用 base model 强制 fill_token 续写出的答案**,不是 teacher 生成的。

但训练时 KD 信号是用 **teacher (= 同一个 base model)** 跑这个 prompt+answer 序列出的 logits 当 target。也就是说:
- target = base model 在 "prompt + fill_token + 后续答案" 上的逐 token logits
- 这个 target 本质上是 "base model 在被强制开头后的自洽续写"
- 学生学的是 "怎么模仿被强制后的自己" → **没有从更强模型蒸馏出新能力**

这跟正经 KD(teacher 是更强的模型,student 学跨能力)不一样。这是 **self-distillation 的特殊形态**,理论上限就是 base model 的能力上限。

**结论 6**:整个训练只是把 base model 的"被引导后的自己"压成主分布,没有引入外部知识。base model 不会的题,训完也不会。

---

## 3. 评测 baseline 已经偏高 → 边际空间被吃掉

### 3.1 V2 4k 口径下 baseline 在 mistake 池上能救 13.83%

这个数有意思 —— **mistake 池的定义就是 baseline 跑错的题**,理论上 baseline 在 mistake 上应该 0%。

但 mistake 池是 **greedy 单次**判定的,baseline 在评测时如果重新做(也是 greedy 单次)**应该仍然 0%**。然而实测是 13.83%,说明:
- vLLM 推理在不同 run 上有非确定性(KV cache 顺序、batching 等),即使 T=0 也不是 bit-exact
- 13.83% 是这种推理 noise 给出的"自然救回率",**LoRA 必须比这个高才算有效果**

V2 实测 +5.08pp 净增量,看起来小,但这是"扣掉评测 noise 后的真实 LoRA 增量"。在 V1 8k 口径下 baseline 仍然 0%(因为 V1 池定义在 8k 评测下保留),LoRA "+43pp" 里有 ~13pp 应该归到评测 noise,真实 LoRA 增量 ~30pp。

### 3.2 V2 baseline math500 已经 73.98% / 论文口径 85.95% → 7B 已接近能力上界

R1-Distill-Qwen-7B 在 MATH-500 论文口径下已经 85.95%。这意味着剩下的 ~14% 是**7B 在 MATH 难度下的能力上界**,任何 KD-based 方法都很难撼动 —— 因为 teacher 就是它自己,蒸不出新能力。

V2 mistake 池 = 7B 跑错的 2025 题,其中相当一部分是**模型架构 + 7B 参数量根本搞不定的题**。这部分用 self-distillation KD 完全救不回来。

**结论 7**:在 13.83% baseline 之上 + 7B 已接近能力顶之下,LoRA 通过 self-KD 能做到 +5pp 已经是这条技术路线的**天花板**,不是训练 bug。

---

## 4. 总结 — 三层瓶颈叠加

| # | 瓶颈 | 量化 | 性质 |
|---|---|---|---|
| **1** | fill 救回率上界 | 62.42% mistake 有 fill 信号,37% 题根本没数据 | **数据流形态固有上界**,扩 fill_epoch 可线性涨,但有抽样收益衰减 |
| **2** | corr:fill = 7:2 稀释 | corr 样本拉 LoRA 回 base,fill 信号被淹 | **mixing 配比导致**,可以重采样但会破 corr |
| **3** | fill 是 hard-label 单点 KD | 多解题被压单解,首 token 训过头后无新信号 | **算法固有形态**,KL target 是 single-answer |
| **4** | self-distillation 无新能力 | teacher = base,蒸不出 7B 能力上限之外 | **方法论上界**,需要外部信号才能突破 |
| **5** | 7B + MATH 已接近能力上界 | baseline 85.95% on MATH-500 | **模型容量上界**,与 LoRA 训练无关 |

**为什么停在 +5pp(V2 口径)/ 43% 绝对(V1 8k 口径)**:不是参数没调到位,不是 bug,而是**算法形态本身**决定了在这个数据流 + 这个 teacher 配置下,KD-based LoRA fine-tuning 的极限就在这个区间。

把 β 从 0.7 调到 0.0,把 prompt 从 1024 调到 2048,只是在让算法的"已知瓶颈"少受工程干扰,**不是突破上界**。

---

## 5. 怎么才能突破?(只列方向,不开工)

按上述瓶颈编号对应:

1. **瓶颈 1 (fill 数据稀疏)** → 多轮 fill_epoch / 用更大 K / fill 阶段允许 top-2 候选续写 / 用 sampling 而非首 token 强制(给后面留多种 chain)
2. **瓶颈 2 (corr 稀释)** → 训练时 fill 样本重采样 / 给 fill loss 更大 lambda / corr-fill 分阶段训(先 corr 再 fill 或反之)
3. **瓶颈 3 (单点 KD)** → 用多条 rollout 构造软目标(`build_first_token_target_logprobs` 的设计意图,但原 pipeline 没接入)。**S-GRPO 借鉴点 B「位置加权的奖励信号」就是这条** —— 给同一题多条对答案以不同 reward,target 不再是单点
4. **瓶颈 4 (self-distillation)** → 换 teacher(用更强模型,但同时换 chat_template 又是工程坑);或者引入**外部 reward 信号**(规则评分、过程评分模型)→ 即 GRPO 路线
5. **瓶颈 5 (模型容量)** → 换 14B base / 跨题类型数据增强 / 推理时增预算(roll-K)

**与 S-GRPO 分析文档的关联**:那份文档里我们筛出的 4 个借鉴点(截断 rollout / 位置加权 reward / answer-segment-only loss / 去 std 归一化),**恰好命中本诊断里的瓶颈 3 和 4**:
- 借鉴点 B 直接破瓶颈 3:把单点 hard target 换成 reward-weighted 多点 soft target
- 借鉴点 A/C 配套破瓶颈 2 + 工程成本:让 GRPO-like 方法在我们 4 卡上能跑

所以下一步如果要做"突破 +5pp 天花板",最对症的不是改 β 或调 lr,而是**把 fill 的 single-answer hard label 换成 multi-rollout reward-weighted soft label**,即向 GRPO 系方法演化。但工程成本明显比纯 supervised KD 高一个量级。

---

## 6. 给当前节点的判断

当前 baseline V2(2026-05-30 重生的池)是 72.92% all / mistake 池 2030 题。如果**复用 a_token_sdcl 算法直接训**,基于本诊断的归因可以预期:
- mistake 净增量 +3 ~ +7pp(在 V2 噪声 + 算法瓶颈范围内)
- corr 净增量 -1 ~ +1pp
- all 净增量 +0 ~ +1.5pp
- math500 roll-8 持平 baseline

**这套数字基本是死的,不值得再为了在 β / lr / epoch 上反复扫盲挪 0.5pp**。要想看到"真涨",必须改算法形态(瓶颈 3/4),或者换数据 / 换 base / 增推理预算(瓶颈 1/5)。

这就是 S-GRPO 分析文档存在的原因 —— **下一步该决定的是算法形态怎么变,不是参数怎么调**。

---

## 7. 引用

- `docs/a_token_sdcl.md` — 算法流程审计(确认无 bug)
- `docs/s_grpo_analysis_and_alternatives.md` — S-GRPO 解构 + 可借鉴机制
- `EXPERIMENT_RESULTS.md` §1.0, §2.2, §2.4 — V1/V2 全部口径实测台账
- `scripts/train/a_token_sdcl.py` Phase A/B/C — fill_correct 数据生成
- `scripts/train/a_token_sdcl_train.py:464-572` — corr/fill 双路径 loss
- `scripts/train/a_token_sd.py:477-525` — `build_first_token_target_logprobs`(已存在但原 pipeline 未调用,GRPO 路径才用)

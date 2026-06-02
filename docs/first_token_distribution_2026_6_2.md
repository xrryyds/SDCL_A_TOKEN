# 首 token 分布对比: Qwen3-8B vs DeepSeek-R1-Distill-Qwen-7B

**日期**: 2026-06-02
**结论**: 所有指令对齐过的 reasoning 模型, 首词分布都极度尖锐, 不是 R1-Distill 蒸馏特有现象

---

## 实验目的

验证假设: "首 token 在 reasoning 模型上的分布是否都被 post-train 锁死?"

R1-Distill-Qwen-7B 在 MATH-500 上 95% 都用 'Okay' 起头是已知现象。我们想看 Qwen3-8B
(post-train + RL, 非蒸馏) 是不是也有这种"开头一致"现象。

如果只有蒸馏模型有, 说明这是 R1-Distill 蒸馏弱点, V3 pool 训练应该能突破;
如果所有模型都有, 说明这是**指令对齐的共性**, 想让 LoRA 学到 fill_token 控制本质上
是在对抗一个非常厚的先验墙。

---

## 实验设置

| 项 | 值 |
|---|---|
| 数据 | MATH train (Math_All train=True), 7496 题 (剔除空 problem 后) |
| 解码 | greedy, T=0, top_p=1.0 |
| max_prompt_length | 2048 |
| max_model_len (vLLM 总窗口) | 6144 (= 2048 prompt + 4096 reserve) |
| chat_template | system_prompt + user prompt |
| system_prompt | "Please reason step by step, and put your final answer within \\boxed{}." |
| 硬件 | 4 卡 H800, vLLM tensor_parallel_size=4 |

实现: `scripts/qwen3_first_token_stats.py`

---

## 结果 1: Qwen3-8B + enable_thinking=True (默认)

Qwen3 chat_template 在 thinking on 时会在 generation 起始位插入 `<think>` 标签。

| 位置 | token | 占比 | 性质 |
|---|---|---|---|
| 第 1 个 | `<think>` (tid=151667) | **100%** | 模板强制 |
| 第 2 个 | `\n` | **100%** | 模板锁定 |
| **第 3 个** | **`Okay`** | **100%** (用户实测) | **真正的"思考首词"** |

- unique = 1
- entropy = 0
- 跟 R1-Distill 的 'Okay' 完全一致

数据文件: `output/first_token_qwen3_8b/`

---

## 结果 2: Qwen3-8B + enable_thinking=False

不开 thinking, chat_template 不加 `<think>` 包装, 第 1 个 token 就是真正的首词。

```json
{
  "n_total": 7496,
  "unique": 3,
  "tokens": [
    {"token_text": "We",   "count": 7364, "pct": 98.24},
    {"token_text": "To",   "count": 129,  "pct": 1.72},
    {"token_text": "Let",  "count": 3,    "pct": 0.04}
  ]
}
```

- **unique = 3**, top1 'We' = **98.24%**
- entropy ≈ 0.13 bits (极低)
- 比 thinking on 还窄

数据文件: `output/first_token_qwen3_8b_thinkoff/`

---

## 结果 3: DeepSeek-R1-Distill-Qwen-7B (历史数据, 旧实验)

数据来源: V3 LoRA 评测, MATH-500 (500 题, 量纲不同, 但量级可比)

| Rank | tid | token | count | pct |
|---|---|---|---|---|
| 1 | 32313 | `'Okay'` | 475 | **95.0%** |
| 2 | 71486 | `'Alright'` | 18 | 3.6% |
| 3 | 11212 | `'Graph'` | 1 | 0.2% |
| 4 | 19434 | `'Keep'` | 1 | 0.2% |
| 5 | 24617 | `'Starting'` | 1 | 0.2% |
| 6 | 1359 | `'By'` | 1 | 0.2% |
| 7 | 5338 | `'First'` | 1 | 0.2% |
| 8 | 1649 | `'Set'` | 1 | 0.2% |
| 9 | 36032 | `'Employ'` | 1 | 0.2% |

- unique = 9
- top1 占比 = 95%
- entropy ≈ 0.34 bits

数据文件: `output/eval_v3_20260531_164556/first_token/lora_counts.json`
脚本: `scripts/first_token_stats.py`

---

## 三个模型横向对比

| 模型 | 模式 | unique | top1 | entropy (bits) | 数据集 / n |
|---|---|---|---|---|---|
| **Qwen3-8B** | thinking on (第3 token) | **1** | 100% `Okay` | **0.00** | MATH train, 7496 |
| **Qwen3-8B** | thinking off (第1 token) | **3** | 98.24% `We` | **0.13** | MATH train, 7496 |
| **R1-Distill-7B** | (无 thinking 模式) | **9** | 95% `Okay` | **0.34** | MATH-500, 500 |

**注**: 模型/数据集/题数都不完全一样, 但量级/性质足以下结论。

---

## 结论

### 1. 所有指令对齐过的模型, 首词分布都极度尖锐

post-train (SFT/RLHF) + reasoning task 让首词坍缩到极少数选项 (1-9 个), 这是
**instruction-following 的标准副产物**, 不是蒸馏弱点。

R1-Distill 的 unique=9 比 Qwen3 的 unique=1/3 看似"分散一些", 但量级相同, 都是极尖锐分布。

### 2. thinking on/off 都尖锐, 但锁住的具体 token 不同

| 模式 | 锁定首词 |
|---|---|
| Qwen3 thinking on | `Okay` (跟 R1-Distill 撞 token) |
| Qwen3 thinking off | `We` |
| R1-Distill | `Okay` |

`Okay` 在 reasoning + thinking 数据上几乎是通用的"思考开始词";
`We` 是 instruction tuning 数据上的 explanation 套话起首。

### 3. 对 SDCL / V3 路线的启示

我们的训练目标 ("让 LoRA 学到 fill_token 控制, 不再无脑 'Okay'") 本质上是在**对抗一个
非常厚的先验墙**。三条路线的结果可以用这个先验墙来解释:

| 路线 | pool Δ | 解释 |
|---|---|---|
| **V3** | **+14.60%** ⭐ | 用强 supervised 信号 (one-hot CE) 直接覆盖先验, 在 pool 题上能突破墙 |
| **SDFT** | +0.40% | KL 信号被稀释到全 span, 信号强度不够突破墙 |
| **GRPO (修复后未跑)** | — | reward 间接信号, 配合 fill rollout 给信号注入, 待验证 |

**跨分布泛化失败 (MATH -3%)** 也能用先验墙解释: LoRA 学到"看到 pool 题特征就破墙",
但 MATH-500 / MATH test 的 prompt 没有 pool 特征, 触发不了破墙逻辑, **先验墙照常起作用**。

### 4. 设计层面的反思

如果想让 LoRA 真正泛化"在所有 reasoning 题上选择性使用 fill_token", 必须:
- 要么训练数据**强制显式打破先验墙** (在 pool/roll 之外的题上也做 fill rollout 训练)
- 要么改 loss 让首 token 信号**永远不被 token-level mask 稀释** (V3 已经这么做了, 所以 pool 涨)
- 要么放弃"在原 prompt 下学 fill_token"的目标, 改成"在 prompt 上加 hint 显式触发"
  (SDFT 已经这么做了, 但 student 看不到 hint, 所以失败)

---

## 数据复现

### Qwen3-8B 测试

```bash
# thinking on (默认)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/qwen3_first_token_stats.py \
  --model_path /workspace/SDCL_A_TOKEN/model/Qwen/Qwen3-8B
# 第 1 个全 <think>, 改 max_tokens 看后续位置

# thinking off
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/qwen3_first_token_stats.py \
  --model_path /workspace/SDCL_A_TOKEN/model/Qwen/Qwen3-8B \
  --enable_thinking false
```

### R1-Distill 历史数据查看

```bash
cat output/eval_v3_20260531_164556/first_token/lora_counts.json
```

---

## 引用

- V3 LoRA 评测结果: `memory/project-v3-eval-result.md`
- SDFT 评测结果: `memory/project-sdft-eval-result.md`
- GRPO 评测结果: `memory/project-grpo-eval-result.md`
- 首 token 分布脚本: `scripts/first_token_stats.py` (R1-Distill 用), `scripts/qwen3_first_token_stats.py` (Qwen3 用)

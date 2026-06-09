# Pipeline Runbook (4096 口径, v2)

完整流程: 池构造 → roll-8 → fill → ORPO 训练数据 → ORPO 训练 → 评测。
含 v1 chat_template bug 诊断和修复 (commit `8af6a7a`)。

口径锁定:
- prompt 上限: 2048 token
- gen/answer 上限: 4096 token
- 总长度: 6144 token (vLLM 总窗口)
- T=0/top_p=1/n=1 (greedy, 池构造) 或 T=0.6/top_p=0.95/n=8 (论文 roll-8)
- 模型: `model/DS/DeepSeek-R1-Distill-Qwen-7B`
- 4 卡 H800
- chat template: `apply_chat_template([sys, user], add_generation_prompt=True)` — **末尾自带 `<｜Assistant｜><think>\n`**
- SYSTEM_PROMPT = `"Please reason step by step and put your final answer within \\boxed{}."`

---

## 阶段 1: 重建 corr / mistake 池

**目的**: 用 take_exam 跑 Base on MATH train 全集 7500 题, teacher_mark_paper 判分拆 corr / mistake 两池。

**输入**:
- 数据源: MATH train 全集 (`Math_All(train=True, subset_name="all")`, ~7500 题)

**输出**:
- `datasets/exam/corr_DS_MATH_pool.json` (覆盖) — Base 做对题
- `datasets/exam/mistake_DS_MATH_pool.json` (覆盖) — Base 做错题
- 中间文件: `datasets/exam/mistake_collection_book_4096.json` / `corr_answer_4096.json`
- 旧池备份: `datasets/exam/*.bak.<TS>`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/rebuild_math_pool_4k.py
```

脚本内部 `student_take_exam_Math_sub(max_prompt_length=6144, max_new_tokens=4096)`,
然后 `TeacherCorrecter(max_new=4096).teacher_mark_paper_with_save()` 拆池。

**结果** (2026-06-07):
| 指标 | 值 |
|---|---|
| total | 7496 |
| corr | **5473** |
| mistake | **2023** |
| accuracy | 73.01% |

---

## 阶段 2: mistake 池 roll-8 评测 + 收集 unsolve_pool

**目的**: 用论文口径 T=0.6 多次 sample, 找出 8 次都做不对的硬骨头题进 unsolve_pool。

**输入**:
- `datasets/exam/mistake_DS_MATH_pool.json` (2023 题)

**输出**:
- `scripts/tmp/roll8_base_<TS>.jsonl` — 每题 8 sample raw answer + n_correct_of_8
- `datasets/exam/unsolve_pool.json` — 8 次全错的题 (字段: question_idx / question / ref_answer / ref_solution, 不带 samples)

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_roll8.py --collect_unsolved
```

口径: max_prompt=6144, max_new=4096, T=0.6, top_p=0.95, n=8。
跑 47 min。

**结果** (2026-06-07):
| 指标 | 值 |
|---|---|
| pass@1 (8 次平均) | **21.30%** |
| pass@8 (any@8) | **45.92%** |
| unsolve (8 次全错) → unsolve_pool | **1094 题 / 2023 = 54.08%** |

每题对错次数分布:
| 对 N/8 次 | 题数 |
|---|---|
| 0/8 | 1094 (54%) |
| 1/8 | 229 |
| 2/8 | 150 |
| 3/8 | 108 |
| 4/8 | 101 |
| 5/8 | 97 |
| 6/8 | 82 |
| 7/8 | 82 |
| 8/8 | 80 (4%) |

---

## 阶段 3: fill unsolve_pool

**目的**: 对 unsolve 池每题用 376 个 first token 池逐个强塞 + greedy 续写, 收集 boxed 命中的 candidate (chosen 数据来源 1)。

**算法细节** (`build_fill_pool_token.py:_worker_fill_pool`):
```
对每道 unsolve 题 q:
  prompt = chat_template([sys, user(q)], add_generation_prompt=True)
         末尾: ...<｜Assistant｜><think>\n
  prompt_ids = tokenize(prompt)
  for tid in 376 candidate tokens:
      input_tokens = prompt_ids + [tid]   ← tid 强塞 prompt 末尾
      gen_text = vLLM.greedy(input_tokens, max_new=4096, T=0)
      full_answer = token_text(tid) + gen_text
      if extract_boxed(full_answer) == ref_answer:
          candidate.answer = full_answer    ← 落盘第一个 token = tid
```

**输入**:
- unsolve 池: `datasets/exam/unsolve_pool.json` (1094 题)
- first token 池: `datasets/first_tokens_test.json` (376 tokens)

**输出**:
- `datasets/exam/fill_unsolve_pool.json` — 救回的题 (含 candidates 列表)
- `datasets/exam/fill_unsolve_unresolved.json` — 376 token 都救不回的硬骨头

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

笛卡尔积 1094 × 376 = 411,344 条 prompt, 4 卡 H800 跑 5.6h。

**结果** (2026-06-07):
| 指标 | 值 |
|---|---|
| 救回 | **850 / 1094 = 77.70%** |
| unresolved | 244 / 1094 = 22.30% |
| candidates/题 | min=1, max=314, avg=61.69, median=39 |

---

## 阶段 4-1: 构造 ORPO 训练数据

**目的**: 从 mistake 池构造 (prompt, chosen, rejected) 偏好对。

**配对规则**:
- **prompt** = mistake 池里的 question
- **rejected** = mistake 池里的 base greedy 错答案 (`it["answer"]`)
- **chosen** 分两源:
  - 题在 unsolve_pool (roll-8 全错) → fill_unsolve_pool 里救回的随机选 1 个 candidate.answer (`chosen_source="fill"`)
  - 题不在 unsolve_pool (roll-8 至少做对一次) → roll8 8 sample 里做对的随机 1 个 (`chosen_source="roll8"`)
- 跳过: unsolve 池里 fill 也救不回的硬骨头 (244 题)

**输入**:
- `datasets/exam/mistake_DS_MATH_pool.json` (2023 题, rejected 来源)
- `datasets/exam/unsolve_pool.json` (1094 题)
- `datasets/exam/fill_unsolve_pool.json` (850 题, chosen 来源 1)
- `scripts/tmp/roll8_base_20260606_173234.jsonl` (chosen 来源 2)

**输出**: `datasets/train/train_data_orpo.json`

每条样本 schema:
```json
{
  "question_idx": int,
  "question": str,
  "ref_answer": str,
  "prompt": str,           // == question
  "chosen": str,           // candidate.answer 或 roll8 做对 sample
  "rejected": str,         // mistake 池 base greedy 错答案
  "chosen_source": str     // "fill" / "roll8"
}
```

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
python scripts/build_train_data_orpo.py \
  --roll8_jsonl scripts/tmp/roll8_base_20260606_173234.jsonl \
  --out_path datasets/train/train_data_orpo.json \
  --seed 42
```

**结果** (2026-06-07):
| 指标 | 值 |
|---|---|
| 总样本 | **1779** |
| chosen from fill | 850 |
| chosen from roll8 | 929 |
| 跳过 (fill 不resolved) | 244 |

---

## 阶段 4-2: ORPO 训练 (含 chat_template bug 诊断)

### v1 (❌ 作废) — chat_template bug 全程学错位置

**用了官方 ORPO main.py 的数据格式**:
```python
chosen_str = apply_chat_template([sys, user, assistant=chosen], tokenize=False)  # ← bug 在此
```

**训练 log**:
- step 0: L_SFT=2.0, first_ce ≈ 0.5
- step 40: L_SFT=0.9, train_loss=1.69
- 看起来 loss 在降, 但 first_ce 量级太小 (fill_token P_base=0.001 应该 -log≈6.9, 不是 0.5)

**评测结果** (`output/eval_v3_20260607_131236/`, ckpt: `output/orpo_4card_20260607_100554/`):

[Greedy]:
| Dataset | Base | LoRA | Δ |
|---|---|---|---|
| corr (5473) | 94.88% | 93.84% | -1.04% |
| mistake (2023) | 18.29% | 20.81% | **+2.52%** |
| unsolve (1094) | 1.46% | 2.01% | +0.55% |
| math500 | 76.00% | 74.60% | -1.40% |
| math_test | 74.84% | 75.64% | +0.80% |

[Roll-8]:
| Dataset / Metric | Base | LoRA | Δ |
|---|---|---|---|
| math500 / pass@1 | 73.85% | 74.10% | +0.25% |
| math500 / any@8 | 86.20% | 87.00% | +0.80% |
| math_test / pass@1 | 74.99% | 75.94% | +0.95% |
| math_test / any@8 | 86.62% | 87.50% | +0.88% |

**首 token 分布** (致命证据):
| 数据集 | Base "Okay" | LoRA "Okay" |
|---|---|---|
| mistake greedy | 96.69% | **96.59%** |
| unsolve greedy | 96.98% | **96.89%** |
| math500 roll-8 | 61.95% | **62.20%** |

→ LoRA 几乎没改 "Okay" 概率, fill_token 池里没一个 token 出现在 LoRA top-K。
→ ORPO 训练根本没学到 fill_token 信号, 27 min 训练白做。

### Bug 调查过程

**诊断脚本 1**: `scripts/tmp/diag_orpo_chat_template.py`
- 静态打 prompt_str 末尾 + chosen_str 前 300 字符 + chosen_ids[:len(prompt_ids)] == prompt_ids?
- **结果**: prompt_ids 末尾是 `<｜Assistant｜>` `<think>` `\n`, 但 chosen_ids 同位置不是 `<think>\n`, 而是 chosen 文本内容
- 而且 chosen_str 里出现的"First, we start with..."居然不是 fill_unsolve_pool.json 里的 candidate.answer 内容 ("Appreciate the problem...")

**诊断脚本 2**: `scripts/tmp/diag_chat_template_assistant.py`
- 测 3 种 assistant 输入下 chat_template 的行为
- **Test 1**: assistant content = "Appreciate the problem" → 渲染正常 (前后包 `<｜Assistant｜>...<｜end▁of▁sentence｜>`)
- **Test 2**: assistant content = `"<think>let me think</think>Answer is 2"` → **`<think>...</think>` 整段被吃掉**, 只剩 "Answer is 2"
- **Test 3**: 只 sys+user + add_generation_prompt=True → 末尾 `<｜Assistant｜><think>\n`

**根因 (双 bug)**:

1. **chat_template 吃 `<think>...</think>` 段** (Test 2)
   - 我们的 fill candidate.answer 都含 `</think>` (推理 + final answer)
   - 但 candidate.answer **不含开头的 `<think>` 标签** (因为模型推理时 `<think>` 是 prompt 给的)
   - 实际我们传 chat_template 的 chosen 字符串是 candidate.answer **不含 `<think>` 但含 `</think>`**, jinja template 看不到匹配的开标签可能行为不可控

2. **prompt 不是 chosen 严格前缀** (Test 1 vs Test 3)
   ```
   prompt_str (Test 3):  "...<｜Assistant｜><think>\n"
   chosen_str (Test 1):  "...<｜Assistant｜>Appreciate the problem<｜end▁of▁sentence｜>"
                                       ^ 没有 <think>\n!
   ```
   两边在 `<｜Assistant｜>` 之后立刻分叉。我们 `compute_loss` 里:
   ```python
   response_mask = chosen_attention_mask - prompt_attention_mask
   first_idx = argmax(response_mask)  # 找第一个 response token 位置
   first_ce = ce_per_token[first_idx]   # 训练它
   ```
   prompt 不是前缀 → response_mask 减法位置错位 → first_idx 找的根本不是 fill_token 位置。

### v2 (✅ 修复后) — commit `8af6a7a`

**修复**: 不用 chat_template 处理 assistant 消息, 改字符串拼接:
```python
prompt_str = apply_chat_template([sys, user], add_generation_prompt=True)
# prompt_str 末尾自带 ...<｜Assistant｜><think>\n
eos = tokenizer.eos_token  # = '<｜end▁of▁sentence｜>'
chosen_str   = prompt_str + chosen   + eos
rejected_str = prompt_str + rejected + eos
```

保证:
1. ✅ prompt 是 chosen/rejected 严格前缀
2. ✅ chosen 第一个 response token = fill_token (训练目标对)
3. ✅ chat_template 不会处理 chosen (直接字符串拼接)

### Loss 公式 (`FirstTokenSplitORPOTrainer.compute_loss`)

```
L_SFT = -log P(y_w[0] | prompt)                       ← 首 token 独立, 不除分母
      + 1/(n_r-1) · Σ_{t=2..n_r} -log P(y_w[t] | ...)   ← 续写 mean
       (prompt 部分跳过)

pos_logp = Σ_t log P(y_w[t] | ...)   ← sum 不 mean (DPO 通用做法防 log_odds 数值奇点)
neg_logp = Σ_t log P(y_l[t] | ...)
log_odds = (pos_logp - neg_logp) - (log(1-exp(pos_logp)) - log(1-exp(neg_logp)))
ratio = F.logsigmoid(log_odds)

L = L_SFT - α · ratio
```

α=0.2 (Llama-2 论文默认, 激进)
fp32 计算 logp + clamp(max=-1e-6) 防 nan。

### 训练超参

| 项 | 值 |
|---|---|
| 卡 | 4 卡 H800 |
| Base | DeepSeek-R1-Distill-Qwen-7B |
| LoRA | r=32 / α=64 / dropout=0 |
| max_prompt_length | 2048 |
| response_max_length | 4096 |
| lr | 2e-5 |
| epoch | 3 |
| batch_size | 4 |
| gradient_accumulation_steps | 8 (有效 batch = 4×4×8 = 128) |
| warmup_steps | 10 |
| alpha | 0.2 |
| optim | paged_adamw_32bit |
| lr_scheduler | cosine |

总 step ≈ 1779 / 128 × 3 ≈ 42。

### v2 训练运行指令

```bash
cd /workspace/SDCL_A_TOKEN
git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
TS=$(date +%Y%m%d_%H%M%S)
python scripts/train/run_orpo_train.py \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --data_path datasets/train/train_data_orpo.json \
  --output_dir output/orpo_4card_${TS} \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --alpha 0.2 \
  --warmup_steps 10 \
  --prompt_max_length 2048 \
  --response_max_length 4096 \
  --lora_r 32 --lora_alpha 64
```

### v2 训练结果 (2026-06-08, ckpt: `output/orpo_4card_20260608_072007/checkpoint_final`)

- 训练耗时 1641s ≈ 27 min
- step 40: L_SFT 在 5-15 之间震荡 (vs v1 的 1-2), train_loss=25.35
- pos_logp ≈ -300~-700, neg_logp ≈ -450~-900, log_odds 双向震荡 (+几百 / -几百)
- **L_SFT 量级对 (fill_token 真实初始 CE 是 6-8) 但训练曲线没明显下降**

### 训练中遇到的多个 bug (按时间顺序)

1. **import 失败** `ModuleNotFoundError: No module named 'src'`
   - 原因: torchrun spawn 子进程后 `from src.orpo_trainer import` 找不到 (orpo/src/ 没 `__init__.py`)
   - 修: `importlib.util.spec_from_file_location` 直接加载文件 (commit `f6080cf` → `de26d82`)
2. **`use_worker` import 失败**
   - 同款 sys.path 问题, 不影响训练结果, 修了 launcher / eval_v3 (commit `60c0107`)
3. **compute_loss 签名兼容**
   - 新版 transformers 给 `compute_loss` 传 `num_items_in_batch=N` kwarg, 我们没接收 → TypeError
   - 修: 加 kwarg 忽略 (commit `d959b9e`)
4. **prompt 长度对齐**
   - 之前 prompt pad 到 prompt_max=2048, chosen 到 response_max=4096, mask 减法维度不一致
   - 修: prompt 也 pad 到 response_max (commit `ce17b4d`)
5. **L_OR nan 爆炸**
   - 官方 mean compute_logps 让 logp 量级 [-1, 0], log(1-exp(logp)) 在 logp→0 时 = log(0) = -inf → nan
   - 修: clamp + fp32 (commit `3952e6b`), 还不够稳
   - 终极修: sum 替代 mean (DPO 通用做法), logp 量级 -100~-1500, 远离奇点 (commit `84432a3`)
6. **chat_template 致命 bug** (上面详述, commit `8af6a7a`)

---

## 阶段 5: 评测 (含首 token 分布统计)

**目的**: 全面对比 Base vs LoRA, 含首 token 分布统计验证 fill_token 是否被推上来。

### 评测脚本改动 (commit `c55b2e2`)

`eval_v3.py` 加:
- `--mistake_path` / `--unsolve_path` 参数
- `--skip_pool` / `--skip_corr` / `--skip_mistake` / `--skip_unsolve` flag
- 评测时统计每条 sample 首 token, 落盘 + stdout top-20

### 输入

| 数据集 | 路径 | 题数 |
|---|---|---|
| corr | `datasets/exam/corr_DS_MATH_pool.json` | 5473 |
| mistake | `datasets/exam/mistake_DS_MATH_pool.json` | 2023 |
| unsolve | `datasets/exam/unsolve_pool.json` | 1094 |
| math500 | `Math_500()` | 500 |
| math_test | `Math_All(train=False, "all")` | ~5000 |

### 输出

- `output/eval_v3_<TS>/eval.log` — 完整 log
- `output/eval_v3_<TS>/<dataset>_<base|lora>_<greedy|roll8>.jsonl` — raw answer
- `output/eval_v3_<TS>/first_token_dist_<dataset>_<base|lora>_<greedy|roll8>.json` — 首 token 分布 (top20 + full_dist)
- `output/eval_v3_<TS>/summary.json` — 总表 JSON

### 评测内容

- **greedy**: 5 数据集 × Base + LoRA = 10 次
- **roll-8**: math500 / math_test × Base + LoRA = 4 次
- 跳过 roll 池 / pool 池 (新流程没用)

### 运行指令

```bash
cd /workspace/SDCL_A_TOKEN
git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/eval_v3.py \
  --lora_path output/orpo_4card_20260608_072007/checkpoint_final \
  --skip_roll \
  --skip_pool
```

预计耗时 3-5h (greedy ~1-2h, roll-8 ~2-3h)。

### 查首 token 分布

```bash
cd /workspace/SDCL_A_TOKEN
python scripts/tmp/show_first_token_dist.py output/eval_v3_<TS> \
  --tags mistake_base_greedy mistake_lora_greedy \
         unsolve_base_greedy unsolve_lora_greedy \
         math500_base_roll8 math500_lora_roll8
```

### v1 评测结果 (作废, 仅作对照)

见上方 "阶段 4-2 v1 (❌ 作废)" 块。结论: LoRA 首 token 分布几乎没变, fill_token 没学到。

### v2 评测结果

待跑。

---

## 关键文件速查

| 文件 | 作用 |
|---|---|
| `scripts/rebuild_math_pool_4k.py` | 阶段 1 池构造 |
| `scripts/tmp/diag_mistake_roll8.py` | 阶段 2 roll-8 评测 + 收集 unsolve_pool |
| `scripts/build_fill_pool_token.py` | 阶段 3 fill 笛卡尔积救题 |
| `scripts/build_train_data_orpo.py` | 阶段 4-1 构造 ORPO 偏好对 |
| `scripts/train/orpo_train.py` | 阶段 4-2 ORPO 训练 (含修复后的 ORPODataset + FirstTokenSplitORPOTrainer) |
| `scripts/train/run_orpo_train.py` | DDP launcher (master_port=29503) |
| `scripts/eval_v3.py` | 阶段 5 评测 (修改版含 mistake/unsolve + 首 token 分布) |
| `scripts/tmp/show_first_token_dist.py` | 查首 token 分布 top20 |
| `scripts/tmp/diag_chat_template_assistant.py` | bug 诊断脚本 |
| `scripts/tmp/diag_orpo_chat_template.py` | bug 诊断脚本 |

## 关键 commit

| commit | 内容 |
|---|---|
| `94117f9` | 切口径 8192→4096, 改 rebuild_math_pool / eval_v3 默认 |
| `be7a424` | rebuild_math_pool_8k.py → rebuild_math_pool_4k.py 重命名 |
| `bc76abe` | 阶段 2 准备: roll-8 + collect_unsolved |
| `a82d363` | 阶段 3 准备: fill 切 4096 + 输入 unsolve_pool |
| `f6080cf` | 阶段 4 ORPO 数据 + 训练脚本 |
| `de26d82` | 修 import bug: importlib 直接加载 orpo_trainer.py |
| `d959b9e` | 修 compute_loss 签名 (num_items_in_batch kwarg) |
| `ce17b4d` | 修 prompt 长度对齐 chosen/rejected |
| `3952e6b` | 修 log_odds 数值不稳 (clamp + fp32) — 部分修 |
| `84432a3` | L_OR 改 sum-of-logp (DPO 通用做法) — 完整修 |
| `c55b2e2` | eval_v3 加 mistake/unsolve + 首 token 分布 |
| **`8af6a7a`** | **🔥 修 chat_template 致命 bug (字符串拼接代替)** |

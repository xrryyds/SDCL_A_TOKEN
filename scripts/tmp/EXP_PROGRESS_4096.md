# 实验日志 (2048+4096 口径)

口径锁定 (所有阶段一致):
- prompt 上限: 2048 token
- gen/answer 上限: 4096 token
- 总长度: 2048 + 4096 = 6144 token (vLLM 总窗口)
- T=0.0 / top_p=1.0 / sample_n=1 (greedy, 池构造)
- 模型: `model/DS/DeepSeek-R1-Distill-Qwen-7B` (Base, 无 LoRA)
- 4 卡 H800
- chat template: `apply_chat_template` + SYSTEM_PROMPT = `"Please reason step by step and put your final answer within \\boxed{}."`

旧 8192 口径流程归档在 `EXP_PROGRESS_8192.md`。

---

## 阶段 1: 重建 corr / mistake 池 ✅

策略: 复用 `scripts/rebuild_math_pool_4k.py` (从 _8k.py 重命名 + 改默认), max_prompt=6144 max_new=4096。

**输入**: MATH train 全集 (~7500 题, `Math_All(train=True, subset_name="all")`)

**输出**:
- `datasets/exam/corr_DS_MATH_pool.json` (覆盖) — **5473 题**
- `datasets/exam/mistake_DS_MATH_pool.json` (覆盖) — **2023 题**
- 中间文件: `datasets/exam/mistake_collection_book_4096.json` / `corr_answer_4096.json`
- 旧池备份: `datasets/exam/*.bak.<TS>`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/rebuild_math_pool_4k.py
```

**结果** (2026-06-07):

| 指标 | 值 |
|---|---|
| total | 7496 |
| correct (corr 池) | **5473** |
| incorrect (mistake 池) | **2023** |
| accuracy | **73.01%** |

对比 8192 口径 (作废历史): corr=6117 / mistake=1379 / acc=81.60%
→ 4096 口径 mistake 多 ~640 题 (4096 截断让一些题没写完 boxed)

---

## 阶段 2: mistake 池 roll-8, 收集 unsolve_pool ✅

策略: 用论文口径 T=0.6 top_p=0.95 sample_n=8 跑 Base on mistake 池 (2023 题), 8 次全错的题进 unsolve_pool (硬骨头, 后续作训练数据用)。

**口径**: max_prompt=6144, max_new=4096 (与池构造对齐)

**输入**: `datasets/exam/mistake_DS_MATH_pool.json` (2023 题)

**输出**:
- `scripts/tmp/roll8_base_20260606_173234.jsonl` (8 次 raw answer, 含 n_correct_of_8)
- `datasets/exam/unsolve_pool.json` (8 次全错题, 1094 题, 字段: question_idx / question / ref_answer / ref_solution)

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_roll8.py --collect_unsolved
```

**结果** (耗时 2812s ≈ 47 min, 4 卡 H800):

| 指标 | 值 |
|---|---|
| pass@1 (8 次平均, 论文口径) | **21.30%** |
| pass@8 (any@8) | **45.92%** |
| unsolved (0/8 对) → unsolve_pool | **1094 题 (54.08%)** |

每题对错次数分布:
| 对 N/8 次 | 题数 | 占比 |
|---|---|---|
| 0/8 | 1094 | 54.08% |
| 1/8 | 229 | 11.32% |
| 2/8 | 150 | 7.41% |
| 3/8 | 108 | 5.34% |
| 4/8 | 101 | 4.99% |
| 5/8 | 97 | 4.79% |
| 6/8 | 82 | 4.05% |
| 7/8 | 82 | 4.05% |
| 8/8 | 80 | 3.95% |

解读: mistake 池 2023 题里:
- 54% (1094 题) → 硬骨头, 8 次都做不对, 进 unsolve_pool 待 fill
- 4% (80 题) → 8 次都对 (mistake 标签是 greedy 一次错, 但 T=0.6 能稳做对)
- 中间各档分布相对均匀, 说明 mistake 池里大量是"不稳定"题

---

## 阶段 3: fill unsolve_pool ✅

策略: 对 unsolve 池 (1094 题, roll-8 全错的硬骨头) 做 376 token × 题 笛卡尔积, T=0 greedy 续写, boxed 命中即收。

**Fill 算法细节** (`build_fill_pool_token.py:_worker_fill_pool`):

每张卡 worker 流程:
1. 把题 chat template 后 token 化, 左截断到 `max_prompt_length=6144`
2. 笛卡尔积: 每题 × 每个 first token tid → `TokensPrompt(prompt_ids + [tid])`
   - tid 是**强制塞进 prompt 末尾的最后一个 token** (不是模型 sample 的)
   - 模型实际看到 "system+user+<assistant>+tid", 从 tid 之后开始续写
3. vLLM `SamplingParams(n=1, temperature=0.0, max_tokens=4096, stop_token_ids=[eos, 151643, 151645])` greedy 续写
4. 拼回: `full_answer = token_text + gen_text`, boxed 命中即收 candidate

→ 学习目标: prompt → 模型自己生成第一个 token == fill_token (这正是 ORPO chosen 要教的)

**口径**: max_prompt=6144, max_new=4096 (与池构造对齐)

**输入**:
- unsolve 池: `datasets/exam/unsolve_pool.json` (**1094 题**)
- 首 token 池: `datasets/first_tokens_test.json` (**376 tokens**)

**输出**:
- `datasets/exam/fill_unsolve_pool.json` ← 救回的题
- `datasets/exam/fill_unsolve_unresolved.json` ← 376 token 都救不回的硬骨头

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

**结果** (耗时 20176s ≈ 336 min / 5.6h, 4 卡 H800):

| 指标 | 值 |
|---|---|
| 救回 | **850 / 1094 = 77.70%** |
| unresolved (硬骨头) | 244 / 1094 = 22.30% |
| candidates/题 | min=1, max=314, **avg=61.69, median=39** |

对比 8192 流程 (fill_multi_pool 86.05% 救回) → 4096 救回率略低 (77.70%), 因为 max_new 减半,
部分长链思考题在 4096 内还没写到 boxed{}。

---

## 阶段 4-1: 构造 ORPO 训练数据 ✅

**输入**:
- mistake: `datasets/exam/mistake_DS_MATH_pool.json` (2023 题, rejected 来源)
- unsolve: `datasets/exam/unsolve_pool.json` (1094 题)
- fill_unsolve_pool: `datasets/exam/fill_unsolve_pool.json` (850 题救回, chosen 来源 1)
- roll8: `scripts/tmp/roll8_base_20260606_173234.jsonl` (2023 题 × 8 sample, chosen 来源 2)

**输出**: `datasets/train/train_data_orpo.json`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
python scripts/build_train_data_orpo.py \
  --roll8_jsonl scripts/tmp/roll8_base_20260606_173234.jsonl \
  --out_path datasets/train/train_data_orpo.json \
  --seed 42
```

**结果**:

| 指标 | 值 |
|---|---|
| 总样本数 | **1779** |
| chosen 来自 fill (unsolve→fill 救回) | 850 |
| chosen 来自 roll8 (非 unsolve, roll-8 做对) | 929 |
| 跳过 (fill 也救不回的硬骨头) | 244 |

抽样: qi=1 ref='25 + 2\sqrt{159}'
- chosen_source=fill, chosen 前80: "Appreciate the problem, let's dive into solving it step by step..."
- rejected 前80: "Okay, so I have this equation to solve..."

---

## 阶段 4-2: ORPO 训练 🔄

**首次运行报错**: `ModuleNotFoundError: No module named 'src'`
- 原因: torchrun spawn 子进程后 `from src.orpo_trainer import ORPOTrainer` 找不到 (orpo/src/ 没 `__init__.py`)
- 修复: 改用 `importlib.util.spec_from_file_location` 直接加载 orpo/src/orpo_trainer.py 文件
- monkey-patch wandb.log / wandb.init 为 noop (官方 trainer 顶部依赖)

**超参** (B 方案: 增加 epoch + 减 warmup):

| 项 | 值 |
|---|---|
| 卡 | 4 卡 H800 |
| Base | DeepSeek-R1-Distill-Qwen-7B |
| LoRA | r=32 / α=64 / dropout=0 |
| max_prompt_length | 2048 |
| response_max_length | 4096 |
| lr | 2e-5 (官方仓库默认) |
| **epoch** | **3** (B 方案, 原 1 太少 14 step 不够学) |
| batch_size | 4 |
| grad_accum | 8 (有效 batch = 4×4×8 = 128) |
| **warmup_steps** | **10** (原 25 > 总 step) |
| alpha (λ) | 0.2 (Llama-2 默认, 激进) |
| optim | paged_adamw_32bit |
| lr_scheduler | cosine |

**总训练步数**: 1779 / 128 × 3 ≈ **42 step**

**Loss 公式** (覆盖官方):
```
L_SFT = -log P(y_w[0] | prompt)                    ← 首 token, 不除
      + 1/(n_r-1) · Σ_{t=2..n_r} -log P(y_w[t]|...)  ← 续写 mean
       (prompt 部分跳过)

L_OR  = -log σ(log odds(y_w)/odds(y_l))            ← 沿用官方

L = L_SFT + 0.2 · L_OR
```

**运行指令**:
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

**结果**: 待跑

---

## 状态 (v2)
- 阶段 1 ✅ corr=5473 / mistake=2023 / acc=73.01%
- 阶段 2 ✅ roll-8 Base pass@1=21.30% / pass@8=45.92%, unsolve_pool=1094 题
- 阶段 3 ✅ fill_unsolve_pool 850/1094=77.70% 救回 (耗时 5.6h)
- 阶段 4-1 ✅ ORPO 训练数据 1779 条 (chosen: 850 fill + 929 roll8)
- 阶段 4-2 🔄 ORPO 训练 (epoch=3, warmup=10, ~42 step)
- 阶段 5 待定 (评测 ORPO LoRA on mistake/unsolve roll-8)

---

## 阶段 4-2: ORPO 训练 (v1, ❌ 数据 bug 作废)

**结果** (2026-06-07):
- 训练耗时 1611s ≈ 27 min, ckpt: `output/orpo_4card_20260607_100554/checkpoint_final`
- L_SFT 1.6 → 0.9 (train_loss 1.69)
- log_odds 大正数 (chosen 概率压过 rejected 几百倍)

**评测结果** (`output/eval_v3_20260607_131236/`):

[Greedy 准确率] (n=1, T=0):
| Dataset | Base | LoRA | Δ |
|---|---|---|---|
| corr (5473) | 94.88% | 93.84% | -1.04% |
| mistake (2023) | 18.29% | 20.81% | **+2.52%** |
| unsolve (1094) | 1.46% | 2.01% | +0.55% |
| math500 | 76.00% | 74.60% | -1.40% |
| math_test (5000) | 74.84% | 75.64% | +0.80% |

[Roll-8] (n=8, T=0.6, top_p=0.95):
| Dataset / Metric | Base | LoRA | Δ |
|---|---|---|---|
| math500 / pass@1 | 73.85% | 74.10% | +0.25% |
| math500 / any@8 | 86.20% | 87.00% | +0.80% |
| math_test / pass@1 | 74.99% | 75.94% | +0.95% |
| math_test / any@8 | 86.62% | 87.50% | +0.88% |

**首 token 分布**:
| 数据集 mode | Base "Okay" | LoRA "Okay" |
|---|---|---|
| mistake greedy | 96.69% | **96.59%** (几乎没变) |
| unsolve greedy | 96.98% | **96.89%** |
| math500 roll-8 | 61.95% | **62.20%** |

**🔴 致命 bug 诊断**: ORPO LoRA 首 token 分布**完全没变**, fill_token 池里没一个 token 出现在 LoRA top-K。
→ ORPO 训练根本没看到 fill_token 信号, 27min 训练白做。

**根因** (诊断脚本 `diag_chat_template_assistant.py` 验证):
R1-Distill 的 `apply_chat_template` 在 assistant 消息上有 2 个坑:
1. **吃 chosen 里 `<think>...</think>` 段** (Test 2 验证: assistant content 含 `<think>` 时整段消失)
2. **prompt 不是 chosen 严格前缀** (add_generation_prompt 加 `<think>\n`, 但完整对话渲染同位置直接是 chosen, 在 `<｜Assistant｜>` 之后立刻分叉)

→ ORPODataset 用 chat_template 处理 chosen 后, response_mask = chosen_mask - prompt_mask 减法位置错位,
   first_idx 找到的不是 fill_token 位置。

**修复** (commit `8af6a7a`): 不用 chat_template 处理 assistant 消息, 改字符串拼接:
```python
prompt_str = apply_chat_template([sys, user], add_generation_prompt=True)
# prompt_str 末尾自带: ...<｜Assistant｜><think>\n
chosen_str   = prompt_str + chosen   + eos    # chosen 起手就是 fill_token
rejected_str = prompt_str + rejected + eos
```

---

## 阶段 4-2 v2: ORPO 重训 (chat template bug 修复后) 🔄

**改动**: ORPODataset 数据格式修复 (commit `8af6a7a`)
**数据**: 不变 (`datasets/train/train_data_orpo.json` 1779 条)
**超参**: 不变 (lr=2e-5, epoch=3, warmup=10, alpha=0.2, bs=4 grad_accum=8, LoRA r=32/α=64)

**运行指令**:
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

**预期**:
- step 0 first_ce 量级 5-8 (fill_token P_base ≈ 0.001 → -log ≈ 6.9), 不再是 v1 的 0.5
- 训完评测时 LoRA 首 token 分布出现 fill_token 池 token (App / If / The 等)
- mistake 池 acc 比 v1 +2.52% 涨更多

**结果**: 待跑

---

## 状态 (v3, 2026-06-08)
- 阶段 1 ✅ corr=5473 / mistake=2023 / acc=73.01%
- 阶段 2 ✅ roll-8 Base pass@1=21.30% / pass@8=45.92%, unsolve_pool=1094 题
- 阶段 3 ✅ fill_unsolve_pool 850/1094=77.70% 救回 (耗时 5.6h)
- 阶段 4-1 ✅ ORPO 训练数据 1779 条 (chosen: 850 fill + 929 roll8)
- 阶段 4-2 v1 ❌ 训完但 chat_template bug 数据错位, 27min 白训, 评测验证 fill_token 没学到
- 阶段 5 v1 ✅ 评测验证了 v1 失败 (首 token 分布几乎不变)
- 阶段 4-2 v2 🔄 重训 (chat_template bug 修复后, commit `8af6a7a`)
- 阶段 5 v2 待定 (重训完后再评测)

---

## 阶段 4: ORPO 训练数据 + 训练脚本 📝

### 4-1. 训练数据构造

策略: 每条样本 (prompt, chosen, rejected)
- **prompt** = mistake 池里的 question
- **rejected** = mistake 池里的 base greedy 错答案 (`it["answer"]`)
- **chosen** 分两源:
  - 题在 unsolve_pool (roll-8 全错) → 在 fill_unsolve_pool 里救回的 → 随机选 1 个 candidate.answer (`chosen_source="fill"`)
  - 题不在 unsolve_pool (roll-8 至少做对一次) → 从 roll8 8 sample 里做对的随机选 1 个 (`chosen_source="roll8"`)
- 跳过: unsolve 池里 fill 也救不回的硬骨头 (在 fill_unsolve_unresolved 里)

**输出**: `datasets/train/train_data_orpo.json`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
python scripts/build_train_data_orpo.py \
  --roll8_jsonl scripts/tmp/roll8_base_20260606_173234.jsonl \
  --out_path datasets/train/train_data_orpo.json \
  --seed 42
```

### 4-2. ORPO 训练

策略: 仿 `a_token_sdcl_train.py` 结构, 调官方 `orpo/src/orpo_trainer.py` 的 ORPOTrainer class。

**Loss**:
```
L_ORPO = L_SFT + λ · L_OR
L_SFT  = NLL on chosen
L_OR   = -log σ(log odds_θ(y_w|x) / odds_θ(y_l|x))
```

Reference-free, 不需 frozen ref model。

**超参 (论文激进版)**:

| 项 | 值 |
|---|---|
| 卡 | 4 卡 H800 |
| Base | DeepSeek-R1-Distill-Qwen-7B |
| LoRA | r=32 / α=64 / dropout=0 |
| max_prompt_length | 2048 |
| response_max_length | 4096 (chosen/rejected 完整序列上限, 总 6144) |
| **lr** | **2e-5** (官方仓库默认) |
| **epoch** | **1** |
| **batch_size** | 4 |
| **gradient_accumulation_steps** | 8 (有效 batch = 4 × 4 × 8 = 128) |
| **alpha (λ)** | **0.2** (激进, Llama-2 默认) |
| warmup_steps | 25 (数据小, 论文 5000 不适用) |
| optim | paged_adamw_32bit |
| lr_scheduler | cosine |

**输出**: `output/orpo_4card_<TS>/checkpoint_final/`

**运行指令** (待 fill 跑完后):
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
TS=$(date +%Y%m%d_%H%M%S)
python scripts/train/run_orpo_train.py \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --data_path datasets/train/train_data_orpo.json \
  --output_dir output/orpo_4card_${TS} \
  --num_epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --alpha 0.2 \
  --prompt_max_length 2048 \
  --response_max_length 4096 \
  --lora_r 32 --lora_alpha 64
```

**结果**: 待跑

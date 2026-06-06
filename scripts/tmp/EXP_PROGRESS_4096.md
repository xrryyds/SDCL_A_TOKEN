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

## 阶段 3: fill unsolve_pool 🔄

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

**输出 (新文件名, 不覆盖 8192 流程的 fill_multi_pool)**:
- `datasets/exam/fill_unsolve_pool.json` ← 救回的题 (含 candidates 列表)
- `datasets/exam/fill_unsolve_unresolved.json` ← 376 token 都救不回的硬骨头

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

**改动 (build_fill_pool_token.py)**:
1. `DEFAULT_UNRESOLVED_PATH` → `unsolve_pool.json`
2. `DEFAULT_OUT_PATH` → `fill_unsolve_pool.json` (不覆盖 fill_multi_pool)
3. `DEFAULT_OUT_UNRESOLVED_PATH` → `fill_unsolve_unresolved.json`
4. `--max_prompt_length` 默认 10240 → 6144
5. `--max_new_tokens` 默认 8192 → 4096

**结果**: 待跑 (在跑, 4 卡每卡 ~273 题 × 376 = ~10万 prompt; 总 411k prompt, 估计 ~6h)

**进度** (2026-06-06 18:54 启动后约 1h20min):
- 4 卡 vLLM 启动正常, KV cache 98.78 GiB
- batch_size=180.63x concurrency
- 每卡 ~25% 进度 @ 13.49 it/s, 估算总耗时 ≈ 5-6h

---

## 状态
- 阶段 1 ✅ corr=5473 / mistake=2023 / acc=73.01%
- 阶段 2 ✅ roll-8 Base pass@1=21.30% / pass@8=45.92%, unsolve_pool=1094 题
- 阶段 3 🔄 fill unsolve_pool (1094 × 376 = 411,344 条 prompt, 预计 ~10h)
- 阶段 4 📝 ORPO 数据 + 训练脚本就位, 待 fill 跑完后启动

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

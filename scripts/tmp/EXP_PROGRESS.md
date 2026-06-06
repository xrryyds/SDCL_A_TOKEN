# 实验日志 (新口径 max_prompt=10240 / max_new=8192)

口径锁定 (所有阶段一致):
- prompt 上限: 2048 token (实测 MATH 题远小于 2048, 截断不触发)
- gen/answer 上限: 8192 token
- 总长度: 2048 + 8192 = 10240 token
- T=0.0 / top_p=1.0 / sample_n=1 (greedy)
- 模型: `model/DS/DeepSeek-R1-Distill-Qwen-7B` (Base, 无 LoRA)
- 4 卡 H800
- chat template: `apply_chat_template` + SYSTEM_PROMPT = `"Please reason step by step and put your final answer within \\boxed{}."`

---

## 当前数据集清单

| 数据 | 路径 | 数量 | 用途 |
|---|---|---|---|
| corr | `datasets/exam/corr_DS_MATH_pool.json` | 6117 | Base 做对题 |
| mistake | `datasets/exam/mistake_DS_MATH_pool.json` | 1379 | Base 做错题 |
| fill_pool | `datasets/exam/fill_multi_pool.json` | 1181 | fill 救回 (含 candidates) |
| unresolved_pool | `datasets/exam/fill_multi_unresolved.json` | 198 | 376 token 都救不回的硬骨头 |
| **fillonly 训练数据** | `datasets/train/train_data_fillonly.json` | **3458** | 阶段 4 训练源 |
| MATH-500 | `datasets/data/...` (待确认) | 500 | 评测 |
| MATH test | `Math_All(train=False)` 加载 | - | 评测 |

---

## 阶段 1: 获取 corr / mistake 池 ✅

**输入**: MATH train 全集 (~7500 题, `Math_All(train=True, subset_name="all")`)
**输出**:
- `datasets/exam/corr_DS_MATH_pool.json` ← **6117 题** (Base 做对)
- `datasets/exam/mistake_DS_MATH_pool.json` ← **1379 题** (Base 做错)
- 旧池备份: `datasets/exam/*.bak.<TS>`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/rebuild_math_pool_8k.py
```

**结果**:
| 指标 | 值 |
|---|---|
| total | 7496 |
| correct | 6117 |
| incorrect | 1379 |
| accuracy | **81.60%** |
| toolong (截断) | 0 |

**附加验证: 新 mistake 池两次 greedy 抖动** (`scripts/tmp/diag_mistake_pool_jitter.py`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_pool_jitter.py
```

| 指标 | 值 |
|---|---|
| 两次文本完全相同 | 1034/1379 = **74.98%** |
| 判对翻转率 | 71/1379 = **5.15%** (错→对 36 / 对→错 35) |
| 两次都对 | 308 = 22.34% |
| 两次都错 | 1000 = 72.52% |
| 没 boxed (截断) | 0% |

结论: vLLM greedy 非严格确定, 长生成有 ~5% 浮点漂移翻转, 但截断已根治。
历史 27% 大头是 vLLM 抖动, 不是池脏。

---

## 阶段 2: 收集 mistake fill 数据 ✅

策略: 跳过 fill_multi, 全部 mistake 题做 **376 token × 题 笛卡尔积** (518,604 条), 
每条 prompt = `chat_template(question) + [候选首 token]`, T=0 greedy 续写, boxed 命中即收。

**输入**:
- mistake 池: `datasets/exam/mistake_DS_MATH_pool.json` (1379 题)
- 首 token 池: `datasets/first_tokens_test.json` (376 tokens)

**输出**:
- `datasets/exam/fill_multi_pool.json` ← 救回 (含 candidates)
- `datasets/exam/fill_multi_unresolved.json` ← 硬骨头

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

**结果** (耗时 50122s ≈ 13.9h):
| 指标 | 值 |
|---|---|
| 总 rollout | 518,604 |
| 救回 | **1181 / 1379 = 85.64%** |
| 仍 unresolved | 198 / 1379 = 14.36% |
| candidates/题 | min=1 max=375 avg=**136.41** median=121 |

平均 136/376 ≈ 36% 的 token 能救活, 分布很宽 (不是单一 magic token)。

---

## 阶段 3: 生成 fillonly 训练数据 ✅

策略: 复用 `build_train_data_poolonly.py`, 每题随机抽 3 个 candidate, 不足全取。source 保留 `"pool"` 兼容 trainer。

**输入**: `datasets/exam/fill_multi_pool.json` (1181 题)
**输出**: `datasets/train/train_data_fillonly.json`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
python scripts/build_train_data_poolonly.py \
  --pool_path datasets/exam/fill_multi_pool.json \
  --out_path datasets/train/train_data_fillonly.json \
  --n_per_q 3 --seed 42
```

**结果**:
| 指标 | 值 |
|---|---|
| pool 题数 | 1181 |
| candidates<3 全取的题 | 59 |
| 展开总样本 | **3458** (= (1181-59)×3 + ≤59×<3) |

**Schema 抽样**:
```json
{
  "source": "pool",
  "question": "Find the positive real number(s) x such that ...",
  "answer": "Casey, so I have this equation to solve: ...",
  "fill_token_id": 4207,
  "fill_token_text": "Case",
  "question_idx": ...,
  "ref_answer": "..."
}
```

---

## 阶段 4: fillonly LoRA 训练 🔄

策略: 复用 V3 主线 trainer `a_token_sdcl_train.py` (能吃 source="pool" 样本)。
fillonly = 只 pool 池、无 corr/roll。loss 口径 (V3 pool 设计):
- pool 样本: **首 token CE on fill_token_id + 后续反向 KL**
- 默认 `--use_ema`, `--lambda_kl=1.0`, `--lambda_ce=0.0`
- ⚠ 默认 `beta_fill>=0` 时整段 loss 都是 KL, CE 项被忽略 (见 trainer L1612-1618)

**口径核对** (与前 3 阶段一致):

| 项 | 阶段 1 | 阶段 2 | 阶段 4 |
|---|---|---|---|
| prompt 上限 | 无强制 (vLLM 窗口内) | 10240 (左截断, MATH 不触发) | 2048 (左截断, MATH 不触发) |
| gen/answer 上限 | 8192 | 8192 | 8192 |
| 总长度 | 10240 | 18432 | 2048+8192=10240 |
| SYSTEM_PROMPT | 同 | 同 | 同 ✅ |
| apply_chat_template | 同 | 同 | 同 ✅ |

→ MATH 题 prompt 实际都远小于 2048, 三阶段实质效果一致。

**输入**: `datasets/train/train_data_fillonly.json` (3458 条)
**输出**: `output/fillonly_4card_<TS>/`

**训练超参** (4 卡 H800):
- bs=4, grad_accum=8 → 有效 batch = 4 × 4 × 8 = **128**
- lr=1e-5, num_epochs=2
- LoRA r=32, α=64
- max_prompt_length=2048, max_answer_length=8192 (总 10240 与池一致)
- gradient_checkpointing 默认开

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
TS=$(date +%Y%m%d_%H%M%S)
python scripts/train/run_a_token_sdcl_train.py \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --data_path datasets/train/train_data_fillonly.json \
  --output_dir output/fillonly_4card_${TS} \
  --num_epochs 2 \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --max_prompt_length 2048 \
  --max_answer_length 8192 \
  --lora_r 32 --lora_alpha 64
```

**结果**:
- ckpt: `output/fillonly_4card_20260606_073155/checkpoint_epoch_2`
- epoch 2/2: avg_loss=**55.671** n_pool=865 (rank0)
- ⚠ launcher 末尾报 `ModuleNotFoundError: No module named 'main'` (use_worker 保活 bug, 训练产物无影响)

---

## 阶段 5: 评测 fillonly LoRA 🔄

复用 `scripts/eval_v3.py`, 改动:
- 默认 `max_prompt_length 6144→10240`, `max_new_tokens 4096→8192` (与训练 + 池构造对齐)
- 新增 `--skip_roll`: 跳过 roll 池整段评测 (本次不跑 roll)

口径: max_prompt=2048+8192=10240, max_new=8192, T=0/top_p=1 greedy, 4 卡 H800

数据集 (跳 roll, 跳 roll-8):
- corr (6117) — Base 做对题, 看是否灾难性遗忘
- pool/fill_multi_pool (1181) — 训练数据本身, in-domain 上界
- math500 (500) — 跨分布
- math_test (~5000) — 跨分布大集

**输出**: `output/eval_v3_<TS>/eval.log`

**运行指令** (跑 Base + LoRA):
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/eval_v3.py \
  --lora_path output/fillonly_4card_20260606_073155/checkpoint_epoch_2 \
  --skip_roll \
  --skip_roll8
```

**结果**: 待跑

---

## 状态
- 阶段 1 ✅ corr/mistake 池
- 阶段 2 ✅ fill 收集
- 阶段 3 ✅ fillonly 训练数据 (3458 条)
- 阶段 4 ✅ 训练 (loss=55.67, ckpt epoch_2)
- 阶段 5 🔄 评测 (Base + LoRA, 跳 roll)

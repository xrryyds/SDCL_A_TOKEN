# 实验日志 (新口径 max_prompt=10240 / max_new=8192)

口径锁定 (所有阶段一致):
- `max_prompt_length=10240` (vLLM 总窗口 = 2048 prompt + 8192 gen)
- `max_new_tokens=8192`
- T=0.0 / top_p=1.0 / sample_n=1 (greedy)
- 模型: `model/DS/DeepSeek-R1-Distill-Qwen-7B` (Base, 无 LoRA)
- 4 卡 H800

---

## 当前数据集清单

| 数据 | 路径 | 数量 | 用途 |
|---|---|---|---|
| corr | `datasets/exam/corr_DS_MATH_pool.json` | 6117 | Base 做对题 |
| mistake | `datasets/exam/mistake_DS_MATH_pool.json` | 1379 | Base 做错题 |
| fill_pool | `datasets/exam/fill_multi_pool.json` | 1181 | fill 救回 (含 candidates) |
| unresolved_pool | `datasets/exam/fill_multi_unresolved.json` | 198 | 376 token 都救不回的硬骨头 |
| MATH-500 | `datasets/data/...` (待确认) | 500 | 评测 |
| MATH test | `Math_All(train=False)` 加载 | - | 评测 |

---

## 阶段 1: 获取 corr / mistake 池 ✅

**输入**:
- 数据源: MATH train 全集 (~7500 题, `Math_All(train=True, subset_name="all")`)

**输出**:
- `datasets/exam/corr_DS_MATH_pool.json` ← **6117 题** (Base 做对)
- `datasets/exam/mistake_DS_MATH_pool.json` ← **1379 题** (Base 做错)
- 中间文件: `datasets/exam/mistake_collection_book_8192.json` / `corr_answer_8192.json`
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

策略: 跳过 fill_multi 阶段, 直接对全部 mistake 题做 **376 token × 题 笛卡尔积** (518,604 条 prompt),
每条 prompt = `chat_template(question) + [候选首 token]`, T=0 greedy 续写, boxed 命中即收。

**输入**:
- mistake 池: `datasets/exam/mistake_DS_MATH_pool.json` (**1379 题**)
- 首 token 池: `datasets/first_tokens_test.json` (**376 tokens**)

**输出 (覆盖)**:
- `datasets/exam/fill_multi_pool.json` ← 救回的题 (含 candidates 列表)
- `datasets/exam/fill_multi_unresolved.json` ← 376 token 都救不回的硬骨头

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

**结果** (耗时 50122s ≈ 13.9h, 4 卡 H800):
| 指标 | 值 |
|---|---|
| 总 rollout | 1379 × 376 = 518,604 |
| 救回 | **1181 / 1379 = 85.64%** |
| 仍 unresolved (硬骨头) | **198 / 1379 = 14.36%** |
| candidates/题 min | 1 |
| candidates/题 max | 375 |
| candidates/题 avg | **136.41** |
| candidates/题 median | 121 |

解读: 几乎每道 mistake 题都至少有一个 first token 能引导 Base 续写出正确答案;
平均 136 / 376 ≈ 36% 的 token 能救活, 说明分布相当宽 (不是单一 magic token)。

---

## 阶段 3: 生成 fillonly 训练数据 🔄

策略: 只用 fill 救回的数据, 每题随机抽 3 个 candidate, 不用 corr / mistake / unresolved。
复用 `build_train_data_poolonly.py` (功能完全等同), source 字段保留 `"pool"` 兼容现有 trainer。

**输入**:
- `datasets/exam/fill_multi_pool.json` (**1181 题**)

**输出**:
- `datasets/train/train_data_fillonly.json` (预期 ~3543 条, 1181 × 3, 少数 candidates<3 题会少)

**输出 schema** (每条样本):
```json
{
  "source": "pool",
  "question": "...",
  "answer": "Okay, so I have...\\boxed{262}",
  "fill_token_id": 8420,
  "fill_token_text": "Okay",
  "question_idx": 1,
  "ref_answer": "262"
}
```

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
python scripts/build_train_data_poolonly.py \
  --pool_path datasets/exam/fill_multi_pool.json \
  --out_path datasets/train/train_data_fillonly.json \
  --n_per_q 3 --seed 42
```

**结果**: 待跑

---

## 状态
- 阶段 1 ✅
- 阶段 2 ✅
- 阶段 3 🔄
- 阶段 4 (训练 / 评测): 待定

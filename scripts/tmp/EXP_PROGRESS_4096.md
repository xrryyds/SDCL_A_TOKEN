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

## 阶段 2: mistake 池 roll-8, 收集 unsolve_pool 🔄

策略: 用论文口径 T=0.6 top_p=0.95 sample_n=8 跑 Base on mistake 池 (2023 题), 8 次全错的题进 unsolve_pool (硬题, 后续作训练数据用)。

**口径**: max_prompt=6144, max_new=4096 (与池构造对齐)

**输入**: `datasets/exam/mistake_DS_MATH_pool.json` (2023 题)

**输出**:
- `scripts/tmp/roll8_base_<TS>.jsonl` (8 次 raw answer, 含 n_correct_of_8)
- `datasets/exam/unsolve_pool.json` (8 次全错题, 字段: question_idx / question / ref_answer / ref_solution)

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/tmp/diag_mistake_roll8.py --collect_unsolved
```

**结果**: 待跑

| 指标 | 值 |
|---|---|
| pass@1 (8 次平均, 论文口径) | ? |
| pass@8 (any@8) | ? |
| unsolved (0/8 对) → unsolve_pool | ? 题 |

---

## 状态
- 阶段 1 ✅ corr=5473 / mistake=2023 / acc=73.01%
- 阶段 2 🔄 roll-8 收集 unsolve_pool
- 阶段 3 待定

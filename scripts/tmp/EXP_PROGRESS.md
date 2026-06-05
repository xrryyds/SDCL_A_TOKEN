# 实验进度

## 当前节点 (2026-06-06)

### 背景
诊断 mistake 池 Base 重评 27% 异常。新 8192 口径重建池后重新查抖动。

### 已完成 1: 重建 corr / mistake 池 (新口径 2048+8192) ✅
- total: **7496 题**, accuracy: **81.60%**
- corr:    **6117 题** → `datasets/exam/corr_DS_MATH_pool.json`
- mistake: **1379 题** → `datasets/exam/mistake_DS_MATH_pool.json`
- toolong: 0
- 旧池备份到 `*.bak.<ts>`

### 已完成 2: 新 mistake 池两次 greedy 抖动 ✅

口径: max_prompt=10240, max_new=8192, T=0/top_p=1/sample_n=1, 4 卡 H800

结果 (n=1379):
| 指标 | 值 |
|---|---|
| 两次文本完全相同 | 1034/1379 = **74.98%** |
| 判对翻转率 | 71/1379 = **5.15%** |
| 错→对 | 36 |
| 对→错 | 35 |
| 两次都对 | 308 = 22.34% |
| 两次都错 | 1000 = 72.52% |
| 没boxed(截断) | 第1次/第2次都 = **0%** |

**关键结论**:
- vLLM greedy **非严格确定**: 25% 题两次生成文本不同, 5.15% 判分翻转
- 截断已被根治 (toolong=0, 没 boxed=0%)
- 翻转双向均衡 (错→对 36 vs 对→错 35) → 浮点漂移导致的随机翻转
- mistake 池 Base 抖动重评率 ≈ 22.34% (两次都对) → 历史 27% 大头是 vLLM 长生成的浮点抖动, 不是池脏

### 当前在做: mistake fill 收集 (build_fill_pool_token.py)

**口径**: max_prompt=10240, max_new=8192, T=0 greedy, sample_n=1, 4 卡 H800
**首 token 池**: `datasets/first_tokens_test.json` (376 tokens)
**输入**: 全部 1379 题 mistake 池 (跳过 fill_multi 阶段, 直接笛卡尔积)
**总 rollout**: 1379 × 376 = 518,604 条 prompt
**输出 (覆盖)**:
  - `datasets/exam/fill_multi_pool.json` ← 救回的题
  - `datasets/exam/fill_multi_unresolved.json` ← 376 个 token 都救不回的硬骨头

**改动** (build_fill_pool_token.py):
1. `DEFAULT_UNRESOLVED_PATH` → `mistake_DS_MATH_pool.json`
2. `--max_prompt_length` 默认 2048 → 10240
3. `--max_new_tokens` 默认 4096 → 8192
4. docstring 改写为 "对 mistake 池"

### 状态
🔄 待跑

```bash
cd /workspace/SDCL_A_TOKEN
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_pool_token.py
```

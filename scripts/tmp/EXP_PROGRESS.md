# 实验进度

## 当前节点 (2026-06-05)

### 背景
诊断 mistake 池 Base 重评 27% 异常,定位到:
- vLLM greedy 零抖动零截断 (`diag_greedy_jitter`)
- 同 prompt 下,池里存的 vs 现在生成 几乎 100% 不同 (`diag_dump_compare`)
- 池里大量题没 boxed (截断),现在重新生成反而能收尾做对
- 结论: 池构造时和当前 take_exam 的生成参数不一致 (具体哪个未定位)

### 已完成: 重建 corr / mistake 池 (新口径 2048+8192) ✅

参数:
- `max_prompt_length=10240` (vLLM 总窗口 = 2048 prompt + 8192 gen)
- `max_new_tokens=8192`

结果 (Base on MATH train, 4 卡 H800):
- total: **7496 题**, accuracy: **81.60%**
- corr:    **6117 题** → `datasets/exam/corr_DS_MATH_pool.json`
- mistake: **1379 题** → `datasets/exam/mistake_DS_MATH_pool.json`
- toolong: 0
- 旧池备份到 `*.bak.<ts>`
- 中间产物保留: `mistake_collection_book_8192.json` / `corr_answer_8192.json`

### 当前在做
**用新池查 mistake 池抖动**: 拿新 mistake 池题, 用同 take_exam 跑两次 (greedy / 同参),
看判对结果翻转率, 验证 mistake 池现在是否稳定。

脚本: `scripts/tmp/diag_mistake_pool_jitter.py`

### 状态
🔄 写脚本

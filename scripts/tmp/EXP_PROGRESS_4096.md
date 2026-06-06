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

## 阶段 1: 重建 corr / mistake 池 🔄

策略: 复用 `scripts/rebuild_math_pool_8k.py`, 临时改默认参数: max_prompt 10240→6144, max_new 8192→4096。

**输入**: MATH train 全集 (~7500 题, `Math_All(train=True, subset_name="all")`)

**输出**:
- `datasets/exam/corr_DS_MATH_pool.json` (覆盖)
- `datasets/exam/mistake_DS_MATH_pool.json` (覆盖)
- 中间文件: `datasets/exam/mistake_collection_book_4096.json` / `corr_answer_4096.json`
- 旧池备份: `datasets/exam/*.bak.<TS>`

**运行指令**:
```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/rebuild_math_pool_8k.py
```

**结果**: 待跑

| 指标 | 值 |
|---|---|
| total | ? |
| correct (corr 池) | ? |
| incorrect (mistake 池) | ? |
| accuracy | ? |
| toolong (截断) | ? |

---

## 状态
- 阶段 1 🔄 重建 corr/mistake 池 (2048+4096)
- 阶段 2 待定
- 阶段 3 待定

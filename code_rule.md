# code_rule — 项目代码规则

简洁记录踩过的坑和必须遵守的口径,新会话/新命令前先核对。

---

## 1. `max_prompt_length` 在两个脚本里语义不同(最易踩的坑)

同名参数 `max_prompt_length`,在训练脚本和评测脚本里**含义完全不同**,填错会导致截断、评分严重低估。

| 脚本 | `max_prompt_length` 真实含义 | 配套参数 |
|---|---|---|
| **训练** `scripts/train/a_token_sdcl_train.py` | **prompt 单独预算**(只截 prompt:`prompt_ids[-max_prompt_length:]`) | `--max_answer_length`(answer 单独预算) |
| **评测** `scripts/eval_v3.py` / `scripts/inference/take_exam.py` | **vLLM 总窗口** `max_model_len`(prompt + gen 合计) | `--max_new_tokens`(生成预算) |

**口径换算(以 prompt 2048 + gen 8192 为例):**

- 训练:`--max_prompt_length 2048 --max_answer_length 8192`(两个都是真值)
- 评测:`--max_prompt_length 10240 --max_new_tokens 8192`(10240 = 2048 + 8192 总窗口)

⚠ **bug 历史**:在 eval 里把 `max_prompt_length` 当 prompt 真值填 2048 → 总窗口仅 2048 < 生成所需 → R1-Distill 长思考链被截断,抽不到 `\boxed{}` → corr/pool 评分掉到接近 0。务必填总窗口。

---

## 2. 评测生成预算 ≥ 数据构造预算

评测 `max_new_tokens` 必须 ≥ 数据(fill/corr)构造时的 gen 预算,否则需要长生成才能写完 boxed 的题被截断 → 假阴性,救回率被低估。

- fill 数据用 8192 gen 救回 → 评测也用 8192,不能用 4096。

---

## 3. vLLM 僵尸进程

任何 vLLM 报错(含 OOM、训练崩)后,重跑前先清:

```bash
pkill -9 -f vllm
pkill -9 -f a_token_sdcl   # 或对应训练/脚本名
nvidia-smi                 # 确认显存清干净再重跑
```

---

## 4. 显存空闲 ≠ 训练慢

GPU-Util 高(80%+)= 算力在跑,留的空闲显存只是 OOM 缓冲,不拖慢训练。要提速只能加 batch_size(减少 forward 次数),但需 kill 重跑且有 OOM 风险。OOM 时反向操作:降 batch_size、升 grad_accum(有效 bs 不变),并设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## 5. launcher use_worker 报错可忽略

`run_a_token_sdcl_train.py` 训练结束/失败后进 use_worker 保活,报 `ModuleNotFoundError: No module named 'main'` 是已知旧 bug,**不影响训练结果**(训练状态 ok 即成功)。

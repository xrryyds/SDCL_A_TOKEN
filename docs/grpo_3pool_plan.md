# GRPO 三池实验设计方案

> 起草于 2026-05-29，目标硬件 4×H200 141G+ / 单机 / 主仓库 hand-rolled 训练框架
> 同步备忘也存在 `~/.claude/projects/.../memory/project-grpo-3pool-design.md`

## 1. 背景

| 实验线 | mistake pool 上方案 C 增量 | 训练数据规模 |
|---|---|---|
| MATH / 2 卡 / V2 4k | β=0.0 +5.08pp / β=0.7 +2.19pp | 7.5k |
| DeepMath / 4 卡 / V2 4k* | β=0.5 +27.31pp（*ckpt 协议未完全对齐，需要重测） | 100k |

数据规模差 14×、增量差 5×。诊断推断：MATH 线 mistake_pool 里相当一部分题，
学生其实"能 rollout 出对的答案，只是 1-shot 概率低"；这部分被 supervised 单条对答案训练，
等价把概率拉到 ~1 而失去多解探索；用 on-policy rollout-N + GRPO 应该能把"对的方向"那族 token
整体上调，鲁棒性更好。

本设计就是要验证这个假设。

## 2. 池构造（数据准备阶段，2048+4096 协议）

```
Step 1: baseline 单次 greedy take_exam (max_prompt_length=6144 / max_new_tokens=4096)
        → exam.json
        → TeacherCorrecter.teacher_mark_paper_with_save() 拆分:
            mistake_collection_book_4096.json (mistake 候选)
            corr_answer_4096.json             (corr_pool)

Step 2: rolling-8 on mistake 候选 (k=8 / T=0.6 / top_p=0.95 / max_prompt=6144 / max_new=4096)
        - 任一对 (≥1/8 答对)：
            * 题目从 mistake 候选移除
            * 写入 grpo_pool，缓存 anchor_answer = 那条对的答案
        - 8 全错：留 mistake_pool

Step 3: 三池落盘:
        corr_pool       → datasets/exam/corr_DS_MATH_pool.json (V2 协议)
        mistake_pool    → datasets/exam/mistake_DS_MATH_pool.json
        grpo_pool       → datasets/exam/grpo_DS_MATH_pool.json (新增)
```

**预计规模**（按 V2-short 数据外推到 2048+4096）：

- corr_pool: ~5400
- mistake_pool: ~1500（rolling-8 全错的硬骨头）
- grpo_pool: ~700-800（rolling-8 救回）

## 3. 训练阶段（三池混合）

### 3.1 Source 分支与 loss

| Source | Loss 路径 | 现状 |
|---|---|---|
| `corr_answer` | KL(student ‖ teacher) on answer span | ✅ `a_token_sdcl_train.py:118` |
| `fill_correct` (mistake_pool 走这条) | 首 token KL(student ‖ q'), q'=(1-β)·teacher+β·onehot(fill_token_id) | ✅ 现成 |
| **`grpo`** (新增) | on-policy GRPO + 全错退化 | ❌ 新写 |

### 3.2 GRPO loss（首 token，复用现有实现）

GRPO 路径每 K=4 step 触发一次，本 batch 内 source==`grpo` 的样本：

1. 用常驻 vLLM 实例 rollout n=8（当前学生 LoRA + temperature=0.6）
2. 二值 reward：每条 rollout 抽答案 → check_correctness → 1 or 0
3. **全错退化**：若 8 条全 reward=0 → 用 sample 缓存的 anchor_answer 当成"虚拟第 9 条 reward=1"加入组
4. 取每条 rollout 的首 token id，调用 `scripts/train/a_token_sd.py:477 build_first_token_target_logprobs(student_first_logits, first_token_ids, rewards, alpha, delta)` 构造 KL target
5. KL(student ‖ target) 累入 `kl_grpo_sum`

**注意**：本设计 GRPO 只在首 token 上算（与 `a_token_sd.py` 现有实现一致），不是全序列 GRPO。

### 3.3 退化机制的算法解读

GRPO 经典痛点：`all-zero advantage`（组内 reward 全为 0 → advantage = 0 → 信号 collapse）。
本设计：rollout 全错时把 anchor_answer 当虚拟正样本加入组，等价 GRPO + expert demonstration anchor，
文献里类似的有 RLOO+demo / DAgger-style RL。

- 好处：GRPO 信号永远不会 collapse；学生在 grpo_pool 上一直有学习信号
- 风险：学生学到 anchor 那条特定路径的 bias（对冲：anchor 只在全错时才用，学生稍有起色就退出退化）

## 4. 工程方案（基于 4×140G+）

### 4.1 显存预算（单卡）

| 组件 | 显存 |
|---|---|
| Trainer student（7B + LoRA-r32 + grad + optim + activations，bs=6 / gas=3 / grad_ckpt） | ~50G |
| Teacher（frozen bf16） | ~15G |
| vLLM rollout 常驻（7B bf16 + 8 rollout × 6k token KV cache） | ~30G |
| 余量 | ~10G |
| **合计** | **~105G** |

H200 141G 单卡塞得下完整三件套 ✅

### 4.2 关键工程点

1. **vLLM 常驻 + 每 K=4 step LoRA hot-reload**：
   - trainer 侧 `peft_model.save_pretrained(tmp_dir)`
   - vLLM 侧 `LLMEngine.add_lora(LoRARequest(name='trainee', lora_path=tmp_dir))`
   - 一次 ~5s，K=4 时摊到每 step ~1.25s
2. **rollout 在 trainer 同卡 colocate**：训练每 step 前 trainer 把当前 batch 里 grpo 样本发给 vLLM；vLLM 占用同一份 GPU，不需要跨进程通信
3. **退化分支**：rollout 完判分，若 reward.sum() == 0 → 把 anchor_answer 拼到 rollouts 列表 + reward=1，再走 GRPO loss

### 4.3 时长估算（MATH 7.5k 题，ep=2，bs=6 / gas=3，4 卡 DDP）

- 总 step：~200
- 每 step 中属于 grpo_pool 的样本：~8 题
- 纯训练 step：~30s
- GRPO 那部分：rollout ~30-60s + LoRA sync 摊 1.25s
- **每 step 总：~70-100s**
- **总训练时长：~4-6h**（vs 纯 supervised ~1.5h）

## 5. 实施前置（必跑）

- [ ] V2 lr=3e-5 实验（确认 MATH 线瓶颈是 lr 还是数据/算法/样本利用）
- [ ] DeepMath β=0.5 ckpt @ V2 4k 重跑（确认 +27pp 在同协议下还剩多少）
- [ ] 上述都跑完后再启动 GRPO 三池

## 6. 实施 v1 工程量分解

| 步骤 | 工程量 |
|---|---|
| 数据 loader 加 source==`grpo` 分支 + anchor_answer 字段 | 0.5 天 |
| vLLM 常驻 + 每 K=4 step LoRA hot-reload | 2-3 天 |
| GRPO loss 接到训练 step（复用 `a_token_sd.py:477`） | 1 天 |
| "全错退化"分支 | 0.5 天 |
| 集成调试 + 跑实验 | 2-3 天 |
| **合计** | **~1 周** |

## 7. 评测口径

跟 V2 baseline 完全对齐：
- mistake/corr 池：`mistake_DS_MATH_pool.json` / `corr_DS_MATH_pool.json`（V2 4k 池）
- 评测协议：`max_prompt_length=6144` / `max_new_tokens=4096`
- math500 roll-8: T=0.6 / top_p=0.95
- 关键 KPI：grpo_pool 上的准确率（vs baseline 在该子集上 0%，因为该集合是 baseline rolling-8 救回的）

## 8. 跟其他实验线的关系

- 如 lr3e5 +15pp → 瓶颈是 lr，GRPO 可能不再必要
- 如 DeepMath @ V2 4k 缩到 +15pp → 验证 V2 4k 协议 + 数据规模假设
- 如上两个都跑完仍说明信号利用率有空间 → GRPO 三池是主攻方向

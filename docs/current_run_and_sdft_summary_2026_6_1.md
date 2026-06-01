# 2026-06-01 当前实验运行手册 + SDFT 论文总结

> 节点快照: 4 卡 H800 在跑 GRPO 全集, 2 卡机器在跑 SDFT v2 (lr=1e-4, epoch=3)
> max_prompt_length=2048 bug 已修 (commit 16860cb), 历史评测数字作废见 [[project-v3-eval-result]]
> 评测口径统一: `eval_v3.py` 默认 max_prompt_length=**6144** (vLLM 总窗口=2048 prompt + 4096 gen)

---

## 1. 当前两台机器正在跑的命令

### 1.1 4 卡 H800: GRPO + first-token-fill 全集训练

**入口**: `scripts/train/run_grpo_a_token_train.py` → 调 `scripts/train/grpo_a_token_train.py`
**实现**: TRL 0.27.2 `GRPOTrainer` + 自定义 `rollout_func` hack
**输出**: `output/grpo_v1_<ts>/`

```bash
cd /workspace/SDCL_A_TOKEN && git pull
pkill -9 -f vllm; pkill -9 -f python; sleep 2

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train/run_grpo_a_token_train.py \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --output_dir output/grpo_v1_$(date +%Y%m%d_%H%M%S) \
  --num_epochs 1 \
  --log_interval 10 \
  --save_steps 200
```

**核心算法流程** (`FillRolloutFunc.__call__`):

```
for each prompt in batch:
    coin = rng.uniform(0, 1)
    if coin < num_fill / num_gen:       # 默认 p = 4/8 = 0.5
        tid = rng.choice(pool_tids)     # 376 候选首 token 随机抽 1
        vllm_input = prompt_ids + [tid] # prepend fill token
        fill_log = log(1/376)           # uniform placeholder for PPO ratio
    else:
        vllm_input = prompt_ids         # 标准 self rollout

reward = (extract_boxed(completion) == ref_answer) ? 1 : 0  # binary
advantage = (r - mean(r)) / (std(r) + 1e-8)                 # GRPO 标准口径, 同 prompt 的 8 个 rollout 一组
loss = PPO_clip(ε=0.2) + β * KL(policy || ref), β=0.001
```

**关键设计点**:
- 每条 prompt 独立掷骰子, **不是**严格 4 self + 4 fill (TRL colocate RepeatSampler 把同题 num_gen 副本撒到不同 rank, 单次 rollout_func 看不到完整 num_gen 副本)
- ref policy = LoRA disable adapter 的 base (省 ref_model 加载, 绕开 TRL 0.27.2 DDP device_map='auto' bug)
- vLLM colocate 4 卡共享显存, gpu_memory_utilization=0.3

**超参** (锁定):
- bs=4, grad_accum=4, world=4, 有效 bs=64
- lr=1e-5, max_prompt=2048, max_new=4096, seed=42
- num_generations=8 (4 self + 4 fill 期望)
- LoRA r=32, α=64, dropout=0.0, target=q/k/v/o_proj (对齐 V3)
- num_epochs=1, save_steps=200, log_interval=10
- 数据: `Math_All(train=True)` MATH 训练集 ~7500 题

**预估**: 28.5s/step × ~1875 steps ≈ **15 小时**

---

### 1.2 2 卡机器: SDFT v2 训练 (lr=1e-4, epoch=3, bs=8)

**入口**: `scripts/train/run_a_token_sdft_train.py` → 调 `scripts/train/a_token_sdft_train.py`
**输出**: `output/sdft_v2_lr1e4_e3_bs8_<ts>/`

```bash
cd /workspace/SDCL_A_TOKEN && git pull
pkill -9 -f vllm; pkill -9 -f python; sleep 2

CUDA_VISIBLE_DEVICES=0,1 python scripts/train/run_a_token_sdft_train.py \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --data_path datasets/train/train_data_v3.json \
  --output_dir output/sdft_v2_lr1e4_e3_bs8_$(date +%Y%m%d_%H%M%S) \
  --num_epochs 3 \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_prompt_length 2048 \
  --max_answer_length 4096 \
  --log_interval 10 \
  --save_total_limit 5
```

**核心算法流程** (SDFT 路线 = off-policy KD + prompt asymmetry):

```
三池数据 (train_data_v3.json, 31916 条):
  corr_answer (5456): Base 在 MATH train 上做对的题
  roll (2134):        fill_multi 自由采样救回 (Base 错 → 加首 token 提示后做对) 的题
  pool (24326):       池强制 fill 救回 (每题 376 候选 token 各试一次, 留对的) 的题

对每条样本 (源自上述三池之一):
  teacher_prompt = student_prompt = (corr)
    or
  teacher_prompt = 'Please start your answer with "{fill_token_text}". {q}'
  student_prompt = q       (roll/pool)

  共同 answer = candidate.answer  (该条样本对应的"对的"答案)

  teacher_logits = teacher.forward(teacher_prompt + answer)  [stop_gradient]
  student_logits = student.forward(student_prompt + answer)
  在 answer span 上对齐两者的 logits (序列长度对齐: 同 answer 但不同 prompt 长度, 两边分别取 answer 段)
  loss = sum over answer tokens of KL(student || teacher)

  三池 loss 纯 sum, w=1.0, 不归一化
```

**关键设计点**:
- corr 池 student 和 teacher prompt 一致 → 退化为标准 self-distillation on Base's correct outputs
- roll/pool 池 teacher 多 hint 一段 → teacher 输出"以特定 token 起头的对的解题"
- student forward 时**看不到 hint** → 学的是 "在原 prompt 下逼近 teacher 的 logits 分布"
- KL 对齐时序列长度不同, 但 answer span 长度相同, 分别取各自 answer span 的 logits

**超参** (v2 新):
- bs=8, grad_accum=4, world=2, 有效 bs=**64** (跟 V3 严格对齐)
- lr=**1e-4** (v1 是 1e-5, 跟 SDFT 论文对齐)
- num_epochs=**3** (v1 是 2, 折衷; 论文数学任务用 5)
- max_prompt=2048, max_answer=4096
- LoRA r=32, α=64, dropout=0.0, target=q/k/v/o_proj
- scheduler: cosine + warmup(min(200, total/10))
- 数据: 31916 条 train_data_v3.json

**预估**: ~11.4 小时 (单 epoch ~3.8h × 3)

**v1 vs v2 差异**:
| 项 | v1 (`sdft_v1_bs8_20260531_161415/`) | v2 (在跑) |
|---|---|---|
| lr | 1e-5 | **1e-4** |
| epoch | 2 | **3** |
| bs / grad_accum | 4 / 8 | **8 / 4** (有效 bs 仍 64) |
| 评测结果 | corr 93.27% / roll 33.72% / pool 0.40% / math500 72.60% / math_test 74.94% | 待训完 |

---

## 2. SDFT 论文总结 (Yang 2024, arXiv 2402.13669)

**论文标题**: Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning
**作者**: Yang et al., ACL 2024
**代码**: https://github.com/sail-sg/sdft

### 2.1 论文要解决的问题

标准 SFT 把 base model 直接拉到 task-specific 分布, 导致两个副作用:
1. **灾难性遗忘** (catastrophic forgetting): 训完任务 X, 通用能力/安全/对齐都掉
2. **分布鸿沟** (distribution gap): SFT target 跟 base model 自己的输出分布差距大, 训练不稳

### 2.2 SDFT 的核心做法

不是直接用人工标注的 ground-truth answer 训, 而是:
1. **先让 base model 自己 rephrase 原 response**, 用 base 自己的 distribution 重写出"等价但符合 base 风格"的 answer
2. **然后用这个 rephrased 版本作为训练 target**

效果: 训练 target 更靠近 base 自己的分布, 削弱了 distribution shift, 缓解遗忘 + 训得更稳。

### 2.3 SDFT 论文超参 (Llama-2-7b-chat + LoRA)

| 项 | 值 |
|---|---|
| base model | Llama-2-7b-chat (主), Llama-2-13b-chat, Llama-3-8B-Instruct |
| learning rate | **1e-4** |
| scheduler | cosine annealing to 0, 无 warmup |
| epochs | 2 (Alpaca/Dolly/OpenHermes), **5 (GSM8K 数学任务)** |
| batch_size | global 8 (per-device/grad_accum 未报) |
| LoRA | r=8, target=q/v only |
| α/dropout | 未报 |
| max_seq_len | 未报 |
| optimizer | "默认 Llama-2 配置" (AdamW 但 β/wd 未报) |
| 数据量 | > 10k 截到 2000 条; GSM8K 全集 ~7500 |
| 数学任务特殊处理 | **distill 答案对了才保留, 不对回退到原 response** (质量过滤) |

### 2.4 我们用的"SDFT" vs 论文 SDFT 的本质差异 ⚠

**完全不是同一个东西, 名字撞了**:

| 维度 | 论文 SDFT | 我们的 SDFT |
|---|---|---|
| **本质** | self-distillation (student 学自己 rephrase 的) | prompt-asymmetric KD (teacher 加 hint, student 不加) |
| distill source | 同一模型 rephrase 原 response | 同一模型 + hint prompt 生成 candidate answer |
| target | rephrased answer (软化版原答案) | candidate answer (fill_token 救回的对的解) |
| 假设 | base 重写的更靠近 base 自己分布 | hint 让 base 能产出更多对的解, 借此扩展 base 的能力边界 |
| 实际目标 | 缓解灾难性遗忘 + 缩小 SFT distribution shift | 让 student 学会"不需要 hint 也能从 pool token 起头解题" |

**为啥我们的方法名字叫 SDFT**: 用户拍板设计点时受论文启发, 借鉴了"用同模型的输出作为软目标"这个思路, 但 loss 形式跟论文完全不同。

### 2.5 v2 调参的依据

v1 (lr=1e-5, epoch=2) 评测 pool 池只有 0.40%, MATH 跨分布零提升, train_loss 收敛到 13.5 还在降, **lr 偏小**嫌疑最大。v2 改动:

| 改动 | 理由 |
|---|---|
| lr 1e-5 → 1e-4 | 跟论文对齐, LoRA 蒸馏一般需要更大 LR |
| epoch 2 → 3 | 论文数学任务用 5; 但我们数据量 31916 是论文 GSM8K 7500 的 4x, 折衷取 3 |
| bs 4 → 8 (grad_accum 8→4) | 有效 bs 仍 64 (不破坏 V3 对照), 单 step 数据更多压榨 H800 显存 |

**没改的 (跟论文相比仍是差异)**:
- LoRA r=32 (vs 论文 r=8): 容量大 4x, 不是瓶颈, 不动避免变量混淆
- 数据无质量过滤 (vs 论文数学任务有): pool 池 candidate 不全是对的, 但暂不动
- 机制级 loss (prompt-asymmetric KL vs self-rephrase): 这是路线本身的差异, 不是超参

---

## 3. 评测口径 (eval_v3.py)

**修复后默认**:
- `--max_prompt_length 6144` (vLLM 总窗口 = 2048 prompt + 4096 gen)
- `--max_new_tokens 4096`

**速跑选项**:
- `--skip_base` 跳过 Base pass (复用历史 Base 数字, 前提 max_prompt_length 一致)
- `--skip_lora` 跳过 LoRA pass (debug)
- `--skip_roll8` 跳过 roll-8 评测 (省 ~80% 时间)

**v2 训完后评测指令** (跟 SDFT v1 同口径):

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_v3.py \
  --lora_path output/sdft_v2_lr1e4_e3_bs8_<ts>/checkpoint_epoch_3/ \
  --model_path /workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B \
  --skip_roll8
```

参考 SDFT v1 (新口径) 数字, 看 v2 是否突破:

| Dataset | Base | SDFT v1 LoRA | Δ v1 | SDFT v2 LoRA (待) | Δ v2 |
|---|---|---|---|---|---|
| corr | 94.15% | 93.27% | -0.88% | ? | ? |
| roll | 30.17% | 33.72% | +3.55% | ? | ? |
| pool | 0.00% | 0.40% | +0.40% | ? | ? |
| math500 | 73.40% | 72.60% | -0.80% | ? | ? |
| math_test | 74.74% | 74.94% | +0.20% | ? | ? |

---

## 4. 关键 LoRA 路径速查

| 路线 | LoRA 路径 | 状态 |
|---|---|---|
| V3 三池 | `output/v3_4card_bs8_20260531_103921/` | 训完, 待用新口径重评 |
| SDFT v1 (lr=1e-5, epoch=2) | `output/sdft_v1_bs8_20260531_161415/checkpoint_epoch_2/` | 训完, 新口径评测完 |
| SDFT v2 (lr=1e-4, epoch=3) | `output/sdft_v2_lr1e4_e3_bs8_<ts>/checkpoint_epoch_3/` | **在跑** |
| GRPO v1 | `output/grpo_v1_<ts>/` | **在跑** |

---

## 5. 已知约束 / 踩过的坑

- **不擅作主张** ([[feedback-no-unilateral-decisions]]): 机制级选择 (loss/prompt/加权/调度) 必须先列方案问
- **不主动给"建议/思考/下一步"** 除非用户问
- **报数据只给数据, 不附加解读**
- **max_prompt_length 是 vLLM 总窗口** ([[feedback-eval-max-tokens]]): prompt + gen, 不是 prompt 单独预算
- **vLLM 报错先 pkill -9** ([[feedback-vllm-zombie-procs]])
- **vLLM v1 spawn 重新 import 主脚本** ([[feedback-vllm-spawn-heredoc]]): heredoc 必炸, 用 `if __name__ == "__main__"` 保护
- **use_worker()** 单进程占全部 GPU, DDP 训练完才能调
- **`from main import use_worker`** 在 launcher 里需要 `sys.path.insert(0, _PROJECT_ROOT)` 才能找到

---

**链接**: [[project-current-node]] [[project-v3-eval-result]] [[project-sdft-eval-result]] [[reference-v3-sdft-runbook]] [[feedback-eval-max-tokens]]

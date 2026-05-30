# 新算法流程(2026-05-30 起)

> **背景**:历史实验全作废(详见 [`docs/a_token_sdcl_why_mistake_capped.md`](./a_token_sdcl_why_mistake_capped.md) 归因诊断 + [`docs/s_grpo_analysis_and_alternatives.md`](./s_grpo_analysis_and_alternatives.md) S-GRPO 分析)。
>
> 旧算法 a_token_sdcl 在 V2 4k 口径下的 mistake 净增量天花板 ≈ +5pp,根因不是 bug,而是:
> 1. fill 数据流单候选 + 7:2 corr/fill 比例 → 信号过窄且被稀释
> 2. β 在 [0, 0.7] 不敏感 → 因为 anchor 信号相对"对 chain 全 span KD"是冗余的
> 3. self-distillation 无法突破 base model 能力上界
>
> 新算法的设计目标:**突破 +5pp 天花板**,在 V2 4k 口径下把 mistake 增量推到 +10pp 以上,同时不破坏 corr。
>
> **文档约定**:本文档**一步一步增量更新**,每一步落地后再写下一步;不写还没讨论过的设计。所有数字、文件路径、命令以这份文档为准。

---

## 0. 当前基线(冷启动锚点)

| 项目 | 值 | 来源 |
|---|---|---|
| Base model | `model/DS/DeepSeek-R1-Distill-Qwen-7B` | 项目内固定 |
| 数据集 | MATH train 7496 题(去重后) | `datasets/exam/Math_train_subset_*.json` |
| 评测协议 | 2048 prompt + 4096 gen(vLLM 总窗口 6144) | 与历史 V2 口径一致 |
| 解码 | greedy 单次(T=0) | 用于产出 mistake/corr 池 |
| Baseline mistake | **2030 题** | `datasets/exam/mistake_collection_book_4096.json` |
| Baseline corr | **5466 题** | `datasets/exam/corr_answer_4096.json` |
| Baseline acc | **72.92%**(5466 / 7496) | greedy 单次 |
| 产出日期 | 2026-05-30 | `scripts/rebuild_math_pool.py` |

**口径说明**:
- `max_prompt_length=6144` 是 **vLLM 总窗口**(prompt+gen 总和),不是 prompt 单独预算。详见 [[feedback-eval-max-tokens]]
- greedy 单次推理在 vLLM 不同 run 上有 ~1% 的自然 noise(KV cache 顺序、batching),所以"mistake 池"的边界本身有 ±70 题左右的抖动,任何后续 LoRA 增量必须**显著超过这个 noise**才算有效
- 论文口径(MATH-500 pass@1 averaged over 8 trials, T=0.6, top_p=0.95)和我们 mistake/corr 池的 greedy 口径**不直接可比**,不要混用

---

## Step 1 — Take Exam:产出 mistake 池和 corr 池

### 1.1 目的

用 baseline DeepSeek-R1-Distill-Qwen-7B(无 LoRA)在 MATH train 全集 7496 题上**单次 greedy 推理**,然后用规则评分(boxed 字符串相等)拆成:
- **mistake 池**:baseline 答错的题 → 后续算法要"救回"的目标集合
- **corr 池**:baseline 答对的题 → 后续算法不能破坏的"已有能力"

这两个池是**所有后续步骤的输入**:
- 训练数据合成的源(mistake 用于生成 fill / rollout-based 信号,corr 用于保持基线)
- 评测指标的口径(训练完后跑 mistake_acc / corr_acc 衡量净增量)

### 1.2 执行命令

```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1
python scripts/rebuild_math_pool.py
```

**为什么用 `rebuild_math_pool.py` 而不是 `main.py eval`**:
- `rebuild_math_pool.py` 是干净的一次性脚本,只做 take_exam + teacher_mark + 备份+覆盖三件事,没有 eval 链路的 math500/roll-K 副作用
- 自带 `try/except + finally use_worker()`,vLLM 子进程退出后能保活
- 不需要传 `--mistake_path / --corr_path` 等参数,所有路径写死在脚本里

### 1.3 内部流程(逐步)

**Stage A:take_exam**
- 入口:`student_take_exam_Math_sub(train=True, subset="all", lora_path=None, max_prompt_length=6144, max_new_tokens=4096)`
- 行为:对每道 train 题构造 chat-template prompt → vLLM 单次 greedy(T=0)生成 ≤ 4096 token 答案 → 写入 `datasets/exam/exam.json`
- prompt 模板:与训练侧 / fill 阶段完全一致(详见 [`docs/a_token_sdcl.md`](./a_token_sdcl.md) §3),`SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."`

**Stage B:teacher_mark_paper**
- 入口:`TeacherCorrecter.teacher_mark_paper_with_save()`
- 行为:**纯字符串相等比较**,不调用 LLM。提取 `\boxed{...}` 内容,经 `normalize_answer` 后与 ref_answer 比对
- 产出两个文件:
  - `datasets/exam/mistake_collection_book_4096.json`(错题)
  - `datasets/exam/corr_answer_4096.json`(对题)

**Stage C:同名拷贝**
- `shutil.copy2` 把上面两个文件改名为后续 pipeline 默认引用的名字:
  - `datasets/exam/mistake_DS_MATH_pool.json` ← `mistake_collection_book_4096.json`
  - `datasets/exam/corr_DS_MATH_pool.json` ← `corr_answer_4096.json`

### 1.4 产出清单

| 文件 | 条数 | 用途 |
|---|---|---|
| `datasets/exam/exam.json` | 7496 | 全量答卷(中间产物,后续不再用) |
| `datasets/exam/mistake_collection_book_4096.json` | 2030 | mistake 池原始 |
| `datasets/exam/corr_answer_4096.json` | 5466 | corr 池原始 |
| `datasets/exam/mistake_DS_MATH_pool.json` | 2030 | mistake 池(后续步骤默认引用) |
| `datasets/exam/corr_DS_MATH_pool.json` | 5466 | corr 池(后续步骤默认引用) |

### 1.5 验证

- [x] 总数 = 2030 + 5466 = 7496 ✅ 与 MATH train 全集一致(无丢题)
- [x] 4 个文件都是 `list[dict]`,每条含 `question / ref_answer / answer`(对错池)或 `question / ref_answer`(纯题目)
- [x] mistake_DS_MATH_pool / corr_DS_MATH_pool 与 mistake_collection_book / corr_answer 内容一致(只是改名)
- [x] baseline acc = 5466 / 7496 = **72.92%**

### 1.6 已知工程坑

- **`main.py:2030` use_worker AssertionError**:`torch.cuda.get_device_name(i)` 在 vLLM 子进程退出后炸 `Invalid device id`。**不影响数据产物**,只是保活循环本身崩。如果以后想修,在 `use_worker` 入口加 `try/except` 包住每张卡的 print。
- **`max_prompt_length` 命名陷阱**:它是 vLLM **总窗口**(prompt+gen 总和),不是 prompt 单独预算。详见 [[feedback-eval-max-tokens]]
- **vLLM 子进程僵尸**:任何 vLLM 报错后,必须 `pkill -9 -f vllm; pkill -9 -f python` 清理才能重启。详见 [[feedback-vllm-zombie-procs]]
- **vLLM v1 heredoc 陷阱**:不要用 `python <<'PY'` 跑 vLLM 代码,子进程 spawn 时找不到 `<stdin>`。要写成真的 `.py` 文件。详见 [[feedback-vllm-spawn-heredoc]]

### 1.7 Step 1 完成判据

- ✅ `mistake_DS_MATH_pool.json` 存在,条数 = 2030
- ✅ `corr_DS_MATH_pool.json` 存在,条数 = 5466
- ✅ acc = 72.92% 记录在册
- ✅ 文档(本文件 §0 + §1)写清楚

**Step 1 状态:已完成(2026-05-30)。**

---

## Step 2 — Build fill_multi pool:多轮多候选首 token 收集

### 2.1 目的

为 mistake 池(2030 题)每一题收集**多条对 rollout 的首 token id + 完整对答案**,作为后续训练的多候选 fill 数据源。这是旧 `fill_correct.json`(每题只保留 1 个 base-logprob 最大的对候选)的升级版,核心改动:

| 维度 | 旧 fill_correct(V2) | 新 fill_multi |
|---|---|---|
| 信号源 | base model 的 top-K logprob → 强制首 token rollout | **同一 base model 的 sampling rollout**(T=0.6, top_p=0.95) |
| 每题候选数 | 1(取 base logprob 最大的对) | **去重后保留所有救回的对 rollout 首 token** |
| 验证机制 | Phase A 取 K=400 候选 → Phase B 强制首 token rollout | 多轮 sampling rollout,任一轮救回即停 |
| 后续训练用法 | 单候选 KL + 后段 KD | 多候选混合软目标(见 §3 Step 3) |

### 2.2 算法

```
active = 全部 mistake (2030 题)
for round in 1..10:
    if active 空: break
    在 active 上跑 vLLM rolling-K (n=K=16, T=0.6, top_p=0.95)
    for 每题:
        收集本轮 K 条 rollout 中对的(boxed 字符串相等评分)
        按 first_token_id 去重(每个 token id 保留首次出现的对 rollout)
        如果去重后 ≥1 候选 → 这题救回:
            写入 candidates 列表 (token_id, token_text, answer, round)
            从 active 移除
        否则 → 留在 active,进入下一轮
```

**关键点**:
- 每轮 K=16 条 rollout
- 上限 10 轮,任一轮 ≥1 对就停 → 单题最多 K × 10 = 160 条 rollout
- **首 token 去重**:每个 token id 只保留首次出现的对 rollout(如果一轮里多条 rollout 首 token 相同,只留第一条)
- 不同轮种子不同(`seed = 42 + round_idx`),保证多样性

### 2.3 执行命令

**入口**:`scripts/build_fill_multi_pool.py`

#### 2 卡机器(0,1)

```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1
python scripts/build_fill_multi_pool.py \
    --k 16 \
    --max_rounds 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_prompt_length 2048 \
    --max_new_tokens 4096 \
    --save_every_round
```

#### 4 卡机器(0,1,2,3)

```bash
cd /workspace/SDCL_A_TOKEN
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/build_fill_multi_pool.py \
    --k 16 \
    --max_rounds 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_prompt_length 2048 \
    --max_new_tokens 4096 \
    --save_every_round
```

**两台机器跑同一题集(不需要分片)**:
- 因为同一题在两台机器上 sampling 结果不同(种子不同,且 vLLM 多 GPU 非确定),**两台机器跑出的 candidates 可以合并** → 数据更丰富
- 跑完后把两台机器的 `fill_multi_pool.json` 合并(按 question_idx group,合并 candidates 后再去重 token_id)
- 如果只想跑一次,选一台机器跑就行

**或者分片**(更经济,如果两台机器算力都吃满):
- 两台机器各跑一半 mistake 池(改 `--mistake_path` 指向预先切好的两个分片),最后合并
- 我可以追加切片脚本,需要的话告诉我

### 2.4 参数自动推导

- `tensor_parallel_size` = `len(CUDA_VISIBLE_DEVICES)`(从环境变量读)
- `max_model_len` = `max_prompt_length + max_new_tokens` = 6144

### 2.5 产出文件

| 文件 | 内容 |
|---|---|
| `datasets/exam/fill_multi_pool.json` | 救回的题列表,每题含 `candidates` 多候选 |
| `datasets/exam/fill_multi_unresolved.json` | 10 轮都没救回的题(候选为空) |

每条 fill_multi 记录格式:

```json
{
  "question_idx": 123,
  "question": "...",
  "ref_answer": "42",
  "candidates": [
    {"token_id": 2014, "token_text": "To",  "answer": "<完整对 rollout 1>", "round": 1},
    {"token_id": 281,  "token_text": "Let", "answer": "<完整对 rollout 2>", "round": 1}
  ],
  "n_rounds_used": 1,
  "total_correct_rollouts_this_round": 5,
  "total_rollouts": 16,
  "source": "fill_multi"
}
```

未救回的题简化记录:
```json
{"question_idx": 456, "question": "...", "ref_answer": "...", "n_rounds": 10, "total_rollouts": 160}
```

### 2.6 工程预估

- 单题平均 rollout 量(假设 50% 题第 1 轮救回 / 30% 第 2 轮 / 20% 后续) ≈ 1.5 × K = 24 rollout
- 全集总 rollout ≈ 2030 × 24 = ~48k rollout × 4096 token
- **2 卡 H100 wall-clock**:~120-180 分钟
- **4 卡 H100 wall-clock**:~60-90 分钟
- 显存:vLLM `gpu_memory_utilization=0.9`,7B bf16 + KV cache 充足

`--save_every_round` 开启后,每轮结束写一次盘,任何中断都能从产出文件恢复进度(虽然脚本本身不支持 resume,但人工可以基于 unresolved 重跑)。

### 2.7 验证

跑完后检查:

```bash
python -c "
import json
p = json.load(open('datasets/exam/fill_multi_pool.json'))
u = json.load(open('datasets/exam/fill_multi_unresolved.json'))
print(f'rescued: {len(p)} / {len(p)+len(u)} = {100*len(p)/(len(p)+len(u)):.2f}%')
ncs = [len(x['candidates']) for x in p]
print(f'candidates per question: min={min(ncs)} max={max(ncs)} avg={sum(ncs)/len(ncs):.2f}')
print(f'avg rounds used: {sum(x[\"n_rounds_used\"] for x in p)/len(p):.2f}')
"
```

预期(参考历史 V2 fill 救回率 62.42%):
- 救回率应**显著高于 62%**(因为 K=16 + 10 轮 ≫ 旧 K=400 single-shot)
- 大部分题应在 1-3 轮救回
- 平均 candidates 数应在 2-5 之间

### 2.8 已知工程坑(沿用 Step 1)

- vLLM v1 spawn 模式,**必须用真 .py 文件**(不要 heredoc)
- 任何 vLLM 报错后 `pkill -9 -f vllm` 清理僵尸进程
- `use_worker()` 在 vLLM 子进程退出后会 print 时炸 AssertionError → 已用 try/except 包住,**不影响产物**

### 2.9 Step 2 完成判据

- ✅ `fill_multi_pool.json` 存在,救回率 > 62%(显著超过旧 fill_correct V2 的 62.42%)
- ✅ 平均 candidates 数 > 1(至少有题给出多候选)
- ✅ 文档(本文件 §2)记录实际产出 counts
- ⏳ **未开始**

---

## Step 3 — (待定)

下一步设计将基于 Step 2 实际产出的数据(救回率、平均候选数)来定:

- 如果救回率高 + 平均候选数 ≥ 3 → 走"多候选软目标 fill_multi 训练"(纯 OPD 升级路线)
- 如果救回率仍 ~62% + 候选稀疏 → 考虑加 GRPO-on-policy 补足(混合 RL/OPD)
- 如果发现 mistake 池里相当一部分题完全 unresolvable → 这部分需要单独处理或舍弃

**下一步动作**:Step 2 跑完,数据回来再设计 Step 3。

---

## 附录 A — 文件位置速查

| 路径 | 用途 |
|---|---|
| `scripts/rebuild_math_pool.py` | Step 1 入口脚本 |
| `main.py` | take_exam / teacher_mark / use_worker 等函数定义 |
| `scripts/inference/take_exam.py` | take_exam 的 vLLM 推理实现 |
| `scripts/inference/teacher_correct.py` | teacher_mark_paper 实现(纯字符串判分) |
| `utils/data_utils.py:52,81` | `extract_boxed_content` / `normalize_answer` |
| `datasets/exam/` | 所有池文件存放位置 |
| `model/DS/DeepSeek-R1-Distill-Qwen-7B/` | base model |

## 附录 B — 与历史文档的关系

| 文档 | 状态 | 关系 |
|---|---|---|
| `EXPERIMENT_RESULTS.md` | **历史台账,作废** | 旧 V1/V2 实验数字仅作参考,不作为基线 |
| `EXPERIMENT_RESULTS_DEEPMATH.md` | **历史台账,作废** | 同上 |
| `docs/a_token_sdcl.md` | 算法审计,仍有效 | 旧算法的工程细节(prompt 模板、token 协议、防御性检查),新算法仍复用同一套基建 |
| `docs/a_token_sdcl_why_mistake_capped.md` | **归因诊断,本算法的设计动机来源** | 解释了为什么旧算法卡在 +5pp,新算法要针对性突破哪几层瓶颈 |
| `docs/s_grpo_analysis_and_alternatives.md` | **方法论参考** | S-GRPO 解构 + 5 个 2025 同期算法可借鉴点;新算法的"机制选型"从这里挑 |
| `docs/grpo_3pool_plan.md` | **作废** | 旧 GRPO 3 池方案,实施前思路已变;保留供参考但不作为当前计划 |

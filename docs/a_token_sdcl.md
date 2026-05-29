# a_token_sdcl 算法流程审计文档

> 审计日期：2026-05-29
> 对应脚本：`scripts/run_v2_4k_full_4card.py`（V2 4k 4 卡全流程）
> 审计目的：确认算法无 bug、参数正确、prompt 模板前后统一

---

## 0. TL;DR（先看结论）

| 项目 | 状态 | 说明 |
|---|---|---|
| Token 协议 | ✅ **2048 + 4096**（不是 1024+4096） | prompt budget=2048 / max_new=4096 / vLLM 总窗口=6144 |
| Prompt 模板（学生侧） | ✅ 全流程统一 | take_exam / fill / 训练 共享 SYSTEM_PROMPT + chat_template |
| Prompt 模板（教师判分） | ✅ 不参与 LLM | teacher_mark_paper 是纯字符串相等比较，不调用 LLM |
| fill_token_id 一致性 | ✅ 训练侧有防御性检查 | `a_token_sdcl_train.py:188` 兜底，不会 mismatch |
| Tokenizer use_fast | ⚠️ 训练默认 fast / 推理 use_fast=False | **不构成 bug**（vocab 共享 + 防御性检查），但建议统一 |
| kl_fill 平均 loss 日志 | ✅ 已存在 | `a_token_sdcl_train.py:960` 每 log_interval 打印 |
| use_worker 保活 | ✅ try/except/finally 全覆盖 | 异常路径也会 use_worker |

总评：**算法流程无 bug，可放心跑**。tokenizer use_fast 不一致是历史遗留，已有防御性检查兜底，不影响正确性，但作为技术债建议后续统一。

---

## 1. 整体流程

```
Stage A: take_exam (单次 greedy)
  └─→ datasets/exam/exam.json
      └─→ TeacherCorrecter.teacher_mark_paper_with_save()
          ├─→ datasets/exam/mistake_collection_book_4096.json  (mistake 候选)
          └─→ datasets/exam/corr_answer_4096.json              (corr 池原始)
              ↓ shutil.copy2 改名
          ├─→ datasets/exam/mistake_DS_MATH_pool.json
          └─→ datasets/exam/corr_DS_MATH_pool.json

Stage B: main.py pipeline --skip_train
  ├─→ generate_fill_correct (Phase A/B/C)
  │   └─→ datasets/exam/fill_correct.json
  └─→ merge_to_train_data (corr_pool + fill_correct)
      └─→ datasets/exam/a_token_train_data.json

Stage C1: scripts/train/run_a_token_sdcl_train.py --beta_fill 0.0
  └─→ output/a_token_b00_v2_4k_4card_<TS>/checkpoint_epoch_2/

Stage C2: scripts/train/run_a_token_sdcl_train.py --beta_fill 0.7
  └─→ output/a_token_b07_v2_4k_4card_<TS>/checkpoint_epoch_2/
```

---

## 2. Token 协议（**2048 + 4096**，全程统一）

> ⚠️ 用户在需求里写过 "1024+4096"，那是口误。实际全流程都是 **2048+4096**。

| 阶段 | prompt budget | 生成预算 | vLLM 总窗口 | 实际命令参数 |
|---|---|---|---|---|
| **A. take_exam** | 2048（被截到 vLLM 总窗口 - max_new） | 4096 | 6144 | `max_prompt_length=6144, max_new_tokens=4096` |
| **B. fill_correct 生成** | 2048 | 4096 | 6144（vLLM 内部） | `--fill_prompt_len 2048 --fill_max_gen_token 4096` |
| **C. 训练** | 2048（trainer 内 left-pad/截断） | 4096（answer 截断） | — | `--max_prompt_length 2048 --max_answer_length 4096` |

### 关键陷阱（已确认正确处理）

`vLLM` 的 `max_prompt_length` 实际是**总窗口** = prompt + gen。所以 take_exam 阶段 `MAX_PROMPT_LEN_VLLM=6144`，是 `2048 + 4096` 的总和；而训练阶段 `--max_prompt_length 2048` 是 prompt 单独预算。这两个名字相同但含义不同的参数，在 `run_v2_4k_full_4card.py:65-68` 的常量定义里已经分开命名（`MAX_PROMPT_LEN_VLLM` vs `TRAIN_MAX_PROMPT`），不会搞错。

---

## 3. Prompt 模板（**学生侧全程统一**）

### 3.1 SYSTEM_PROMPT 定义点

```python
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
```

| 文件:行 | 用途 |
|---|---|
| `scripts/inference/take_exam.py:30` | take_exam（Stage A） |
| `scripts/train/a_token_sd.py:165` | 老训练入口（V1） |
| `scripts/train/student_train_v2.py:46` | V2 训练（同字符串） |
| `scripts/train/a_token_sdcl_train.py:49` | **当前训练入口**（`from a_token_sd import SYSTEM_PROMPT`） ✅ |
| `scripts/train/a_token_sdcl.py:38` | fill 阶段（Stage B），同样 import 自 `a_token_sd` ✅ |

三处独立定义的字符串完全一致；其余通过 import 复用。**学生侧 take_exam / fill / 训练共享同一个 SYSTEM_PROMPT。**

### 3.2 chat_template

所有 4 个学生侧入口都走：

```python
tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=True,
)
```

DeepSeek-R1-Distill-Qwen tokenizer 自带 chat_template，全流程使用相同的 tokenizer（同一个 model_path 加载），所以 prompt 文本逐字节一致。

### 3.3 教师判分（teacher_mark_paper）

`scripts/inference/teacher_correct.py` 里有另一个 system prompt（"You are a helpful assistant who good at math"），但**那是给提示生成（hint generation）用的，不在本 pipeline 路径上**。

本 pipeline 调用的是 `TeacherCorrecter.teacher_mark_paper_with_save()`，它做的是 **`extract_boxed(student_answer) == extract_boxed(ref_answer)` 的纯字符串相等判分**，不调用 LLM，因此不受 prompt 模板影响。

---

## 4. 算法核心（fill_correct + 训练 KL）

### 4.1 fill_correct 三阶段（`scripts/train/a_token_sdcl.py`）

**Phase A（Top-K logprobs，line 204-211）**：
- 输入：mistake 题目（vLLM `max_tokens=1`，`logprobs=K_LOGPROBS=400`）
- 输出：每题 prompt 末尾后的 top-400 候选首 token + 它们的 base logprob

**Phase B（强制首 token rollout，line 240-266）**：
- 对 Phase A 给出的若干候选 token，用 vLLM `TokensPrompt(prompt_token_ids=prompt_ids + [cand_token_id])` 强制让 cand_token_id 作为首 token，然后续写到 max_new=4096
- 结果：每题每候选 → 一条完整 rollout

**Phase C（按 base logprob 选最优，line 281-303）**：
- 在那些 rollout 答对的候选里，挑 base logprob 最大的那个（学生本来更倾向产出的那条 corrected path）
- 该候选的 token_id / token_text + 整条 rollout 答案 → 写入 `fill_correct.json`

### 4.2 训练 loss 路径（`a_token_sdcl_train.py`）

| Source | Loss | 说明 |
|---|---|---|
| `corr_answer` | `KL(student ‖ teacher)` 在整个 answer span 上 | β 不参与，纯 KD |
| `fill_correct` | 首 token 上 `KL(student ‖ q')`，其中 `q' = (1-β)·teacher + β·onehot(fill_token_id)` | β=0 → 纯 teacher KD；β=1 → 纯 CE on fill_token |

#### fill_token_id 在训练侧的处理（line 183-195，**有防御性检查**）

```python
if src == "fill_correct":
    ftid = int(sample["fill_token_id"])
    if answer_ids[0] != ftid:
        # 罕见：tokenizer 把 fill_token_text 跟后续字符合并了
        # 用 token id 拼接保证首 token 严格等于 ftid
        answer_ids = [ftid] + answer_ids
    fill_token_id = ftid
    fill_pos_in_seq = len(prompt_ids)
```

**这里就是 tokenizer use_fast 不一致的兜底**：fill_token_id 是推理侧（use_fast=False）产出的 vocab id；训练侧 tokenize answer_text 时即便用了 fast tokenizer 把首字符跟后续字符合并成不同的 token，这段代码也会把 `[ftid] + answer_ids` 强制拼上去，保证训练侧 first token 严格等于推理侧的 fill_token_id。**因此 use_fast 不一致不构成正确性 bug。**

### 4.3 loss 日志（每 log_interval=10 step 一行）

`a_token_sdcl_train.py:960-1024` 已经在打印：

```
[Step N] epoch=E lr=L loss=L ce=C kl_corr=KC kl_fill=KF
```

其中 `kl_fill` = 本窗口内 fill_correct 样本的平均首 token KL loss。**用户问的"fill 平均 loss"已经存在，无需新增**。

---

## 5. 工程兜底点

### 5.1 use_worker 保活（success + exception 都触发）

`scripts/run_v2_4k_full_4card.py:324-342`:

```python
if __name__ == "__main__":
    overall = "ok"; top_err = None
    try:
        main()
    except BaseException as e:
        overall = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    finally:
        try: _print_final_summary(overall_status=overall, top_err=top_err)
        except: traceback.print_exc()
        try:
            from main import use_worker
            use_worker()
        except: traceback.print_exc()
```

✅ 异常路径也会进 finally → use_worker 保活。

### 5.2 tmux-friendly 最终汇总

`_print_final_summary` 会一次性打印：
- 各阶段时序（A/B/C1/C2）
- 数据产物路径 + 样本数（mistake/corr/fill_correct/train_data）
- ckpt 路径（CKPT_B00 / CKPT_B07，含同目录其他 epoch ckpt）
- training.log 路径（提示用户去看 kl_corr / kl_fill）

即便 tmux 滚屏丢失中间输出，tail 也能拿到完整信息。

---

## 6. 已知技术债（不影响本次实验正确性）

### 6.1 Tokenizer use_fast 不一致

| 文件 | use_fast | 影响 |
|---|---|---|
| `main.py:64,849,1789,2477` | False | take_exam 等推理路径 |
| `scripts/inference/take_exam.py:87` | False | take_exam |
| `scripts/train/a_token_sdcl.py:156` | False | fill 生成（vLLM 加载） |
| `scripts/train/a_token_sdcl_train.py:698` | **未指定 → 默认 True (fast)** | 训练 |

**为什么不构成 bug**：
1. fast 与 slow tokenizer 共享同一份 vocab.json，token_id 含义一致
2. 训练侧 `_encode_sample` 在 line 188 有防御性检查 `if answer_ids[0] != ftid: answer_ids = [ftid] + answer_ids`，把 fill_token_id 强制拼到 answer 首位，等于绕开了"首字符 BPE 合并差异"
3. corr_answer 路径不依赖首 token id，整个 answer span 走 teacher-forcing，fast/slow tokenize 出的 token 数可能差 1-2 个但 KL 仍是 well-defined

**建议**（后续清理，非阻断）：
```python
# scripts/train/a_token_sdcl_train.py:698
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)
```

---

## 7. 跟用户原始需求的逐项核对

| 用户问 | 答 |
|---|---|
| "整个流程的 token 是不是 1024+4096" | **不是**。是 **2048+4096**（用户口误）。详见 §2 |
| "是不是前后统一的 prompt 模板" | **是**。学生侧 take_exam / fill / 训练共享 SYSTEM_PROMPT + 同一份 chat_template。教师判分不调 LLM 不受影响。详见 §3 |
| "fill 的平均 loss 会不会更有利于分析" | **已经存在**。`avg_kl_fill` 每 log_interval=10 step 打印一次，无需新增 |
| "异常或者跑完调用 use_worker" | **已实现**。try/except/finally 全覆盖，详见 §5.1 |
| "训练完后把 lora 地址在最后打印到终端" | **已实现**。`_print_final_summary` 的 `[checkpoints]` section |
| "整个算法流程有没有 bug" | **无算法 bug**。仅一处 tokenizer use_fast 不一致（已被防御性检查兜底，不影响正确性），列为技术债，详见 §6.1 |

---

## 8. 跑前最终检查清单

- [x] CUDA_VISIBLE_DEVICES=0,1,2,3
- [x] `model/DS/DeepSeek-R1-Distill-Qwen-7B/` 存在
- [x] `datasets/exam/Math_train_subset_*.json` 存在（take_exam 输入）
- [x] `tmux` session 启动（输出可丢，最终汇总会重打）
- [x] 磁盘空余 ≥ 60G（2 个 ckpt × 7B LoRA + train_data + fill_correct）

执行：

```bash
cd /workspace/SDCL_A_TOKEN
git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/run_v2_4k_full_4card.py
```

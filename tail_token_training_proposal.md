# 随机首Token填充 + 混合蒸馏训练 需求文档

---

## 一、整体流程概览

```
mistake数据 ──┐
              ├──► [方法1] 随机首Token填充评测 ──► fill_correct.json
first_tokens ─┘

corr_answer.json ──┐
                   ├──► [方法2] 数据合并 ──► a_token_train_data.json
fill_correct.json ─┘

a_token_train_data.json ──► [方法3] 混合蒸馏训练 ──► 训练后模型
```

---

## 二、方法1：生成 fill_correct.json

### 参数

| 参数名 | 说明 |
|--------|------|
| `model_path` | 模型路径 |
| `mistake_path` | mistake数据文件路径 |
| `firt_token_list_path` | 首token候选池路径（默认 `datasets/first_tokens_test.json`） |
| `roll_n` | 每题随机抽取的候选token数量 |
| `max_gen_token` | 最大生成token数 |
| `prompt_len` | prompt最大长度，默认1024 |

> `max_model_len = prompt_len + max_gen_token`

### 算法流程

```
对 mistake 中的每道题 q:
    1. 构建 prompt（参考 take_exam 的 build_prompt）
    2. 先做一次无填充的 greedy 生成，获取模型在首token位置的 logits
    3. 从 first_tokens 候选池中随机抽取 roll_n 个 token
    4. 对每个候选 token_i:
        a. 将 token_i 的 text 作为首token强制填充到 prompt 末尾
           （参考 exam_with_hints 的做法）
        b. 模型自由生成后续内容
        c. 判断生成结果是否正确（提取 \boxed{} 与 ref_answer 比较）
    5. 收集所有"做对"的候选
    6. 如果没有任何候选做对 → 跳过该题
    7. 如果有多个候选做对:
        → 取步骤2中该 token_id 对应的首token logit 最大的那个
    8. 保存最终选中的 (题目, 生成答案(fill_token + gen_txt), fill_token_id, fill_token_text)
```

### 多卡并行要求

- 支持多卡 vLLM 并行处理数据加速
- 参考 `scripts/train/a_token_sd copy.py` 中的 vLLM 使用方式
- 可使用 `tensor_parallel_size` 或多进程分片

### 输出格式 fill_correct.json

```json
[
  {
    "question_idx": 42,
    "question": "...",
    "answer": "填充后生成的正确答案",
    "ref_answer": "参考答案",
    "ref_solution": "参考解答",
    "fill_token_id": 1654,
    "fill_token_text": "We",
    "source": "fill_correct"
  }
]
```

---

## 三、方法2：合并为 a_token_train_data.json

### 输入

- `corr_answer.json`：模型原本就做对的题
- `fill_correct.json`：通过填充首token后做对的题

### 合并规则

1. 读取 `corr_answer.json`，为每条数据添加 `"source": "corr_answer"`
2. 读取 `fill_correct.json`（已有 `source` 字段）
3. 两者拼接为一个列表
4. 写出为 `a_token_train_data.json`

### 输出格式 a_token_train_data.json

```json
[
  {
    "question_idx": 0,
    "question": "...",
    "answer": "...",
    "ref_answer": "...",
    "ref_solution": "...",
    "source": "corr_answer",
    "fill_token_id": null,
    "fill_token_text": null
  },
  {
    "question_idx": 42,
    "question": "...",
    "answer": "...",
    "ref_answer": "...",
    "ref_solution": "...",
    "source": "fill_correct",
    "fill_token_id": 1654,
    "fill_token_text": "We"
  }
]
```

> 关键：必须通过 `source` 字段标识数据来自哪个文件。

---

## 四、方法3：混合蒸馏训练

### 教师模型

统一使用**初始模型**（即 `model_path` 指向的原始模型）作为教师。

### 训练策略（按 source 区分）

#### 当 `source == "corr_answer"` 时

```
教师分布 = 初始模型对该序列的原始输出分布
loss = KL(学生分布 || 教师分布)   # 全序列正常KL
```

#### 当 `source == "fill_correct"` 时

**第一个生成token位置：**

```
loss_first = CE(学生首token分布, fill_token_id)   # 直接用交叉熵，target为fill_token_id
```

**后续token位置：**

```
teacher_dist = 初始模型的原始输出分布（不做修改）
loss_rest = KL(学生分布 || 教师分布)   # 正常KL
```

**总loss：**

```
loss = loss_first + loss_rest
```

### 设计意图

- `corr_answer` 样本：模型本来就做对了，直接用教师分布做常规蒸馏
- `fill_correct` 样本：模型原本做错，但填充某个首token后能做对
  - 在首token位置，直接用CE监督学生输出fill_token，简单高效
  - 后续token仍跟随教师的正常分布
  - 从而让学生学会：在易错题上选择更好的起始token

---

## 五、关键参考文件

| 文件 | 参考内容 |
|------|----------|
| `scripts/inference/take_exam.py` | `exam_with_hints` 的首token填充方式、多卡并行 |
| `scripts/train/a_token_sd copy.py` | 多卡vLLM使用、训练框架结构 |
| `scripts/train/a_token_sd_fill.py` | fill版首token KL构造逻辑 |
| `scripts/train/a_token_sd.py` | rollout版首token KL训练流程 |
| `scripts/train/extract_first_tokens.py` | first_tokens文件读取工具 |
| `datasets/first_tokens_test.json` | 首token候选池 |
| `datasets/exam/mistake_DS_MATH.json` | mistake数据样例 |
| `datasets/exam/corr_answer.json` | 正确答案数据样例 |

---

## 六、数据格式参考

### first_tokens_test.json 结构

```json
{
  "tokens": [
    {"token_id": 1654, "token_text": "We", "count": 945},
    {"token_id": 10267, "token_text": "To", "count": 812}
  ]
}
```

### mistake 数据单条结构

```json
{
  "question_idx": 42,
  "question": "题目文本...",
  "answer": "模型的错误答案...",
  "ref_solution": "参考解答过程...",
  "ref_answer": "正确答案",
  "entropy": ""
}
```

### corr_answer 数据单条结构

```json
{
  "question_idx": 0,
  "question": "题目文本...",
  "answer": "模型的正确答案...",
  "ref_solution": "参考解答过程...",
  "ref_answer": "正确答案",
  "entropy": ""
}
```

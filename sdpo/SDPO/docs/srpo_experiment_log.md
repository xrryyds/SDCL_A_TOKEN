# SRPO 实验记录

## Baseline 长度坍缩问题

### 现象

在 SRPO 训练 chemistry 数据集（Qwen3-8B）过程中，模型回复长度持续坍缩：

| 训练阶段 | response_length/mean | first_token/distinct | first_token/entropy |
|----------|---------------------|---------------------|---------------------|
| step 75（无格式约束） | ~17-23 tokens | 1-50 | ~0-1.6 |
| step 95（加 reasoning 扣分后） | ~23 tokens | 204 | 5.23 |

模型输出的典型样本：
```
<reasoning>
CO
</reasoning>
<answer>
A
</answer>
```

回复只有 ~23 tokens，reasoning 内容仅 "CO" 两个字。

### 根因分析

#### 1. Reward 函数只看答案对错，不奖励推理质量

MCQ reward 函数（`verl/utils/reward_score/feedback/mcq.py`）：
```python
reward = float(multiple_choice_answer == ground_truth)
```
正确答案 reward=1.0，无论 reasoning 长短。模型发现跳过推理直接给答案能拿满分，GRPO 强化这个捷径。

#### 2. SDPO 自蒸馏正反馈放大坍缩

SDPO 的流程（`_maybe_build_self_distillation_batch` → `compute_srpo_loss`）：

1. **找 correct sibling**：同 uid 组中答对的 rollout，取完整回复文本作为 "solution"
2. **构建 teacher prompt**：把 correct sibling 的短回复作为示范喂给 teacher
   ```
   [原问题]
   Correct solution:
   <reasoning>CO</reasoning><answer>A</answer>
   Correctly solve the original question.
   ```
3. **Teacher 前向**：`teacher_input_ids = [teacher_prompt + student_responses]`，EMA teacher 看着短示范算 logprob
4. **SDPO loss**：top-k JSD 蒸馏 + DW（熵动态加权）
   - `w = exp(-β·H_teacher)`，teacher 熵越低 → 权重越高 → 蒸馏越强
   - 实测 `teacher_entropy_mean: 0.08`（极低）→ 最强蒸馏

#### 3. EMA Teacher 退化

EMA 更新率 0.05，97 步后：
```
(1 - 0.05)^97 = 0.95^97 ≈ 0.007
```
teacher 中原始 base model 的贡献仅剩 0.7%，**已 99.3% 退化为 student 自身**。自我蒸馏坍缩行为。

#### 正反馈死循环

```
student 短回复 → correct sibling 短 → teacher prompt 短示范
→ EMA teacher 坍缩到低熵短输出 → DW 高权重强蒸馏
→ student 更短 → 循环放大
```

### 验证数据（step 95）

| 指标 | 值 | 说明 |
|------|-----|------|
| val acc mean@16 | 75.3% | 准确率不低，但靠猜不靠推理 |
| response_length/mean | 22.9 | 极短 |
| srpo/teacher_entropy_mean | 0.08 | EMA teacher 极度集中 |
| srpo/dw_weight_mean | 0.21 | DW 权重低（因为 normalizer 也低） |
| srpo/sdpo_frac | 0.21 | 21% 样本走 SDPO 蒸馏 |
| srpo/fill_p_student_mean | 0.00002-0.0004 | 强制 token 在 student 分布中概率极低 |
| first_token/top1_frac | 0.02 | 首 token 极度多样化（因 reroll 强制不同首 token） |
| first_token/entropy | 5.23 | 高熵（reroll 造成的，非自然推理多样性） |

### 尝试过的修复及效果

#### 修复 1：MCQ reward 加 reasoning 格式惩罚

- 无 `<reasoning>` 标签 → 扣 0.5 分
- 效果：模型加了 `<reasoning>` 标签，但内容极短（"CO"），只是跳过门槛
- 结论：**治标不治本**，模型总能找到最小代价路径

#### 修复 2：加最低 reasoning 长度要求（100 字符）

- `<reasoning>` 内容 < 100 字符 → 扣 0.5 分
- 预期：模型会写 100 字符的废话凑数
- 结论：**打地鼠**，不解决根本问题

### 待探索的方向

1. **冻结 base model 当 teacher**（不 EMA）—— teacher 永远保留原始推理能力
2. **大幅降低 EMA 更新率**（0.05 → 0.001）—— 保持 teacher 接近 base model
3. **correct sibling 过滤**—— 短的回复不当示范
4. **连续长度奖励**—— `reward = correct × min(1.0, reasoning_tokens / N)`，非门槛式
5. **Process Reward Model**—— 奖励每步推理而非只看最终答案

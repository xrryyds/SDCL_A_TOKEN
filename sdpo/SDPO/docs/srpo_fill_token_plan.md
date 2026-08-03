# SRPO Fill 首 Token 策略规划

## 核心目标

丰富模型 `<reasoning>\n` 后首 token 的选择，避免坍缩到单一 opener（如 "To"/"The"）。

## 当前机制

- 从 base model 首 token 分布文件（962 个 token）中均匀随机采样
- 问题：前 2 个 token（"To" p=0.75, "The" p=0.24）占 98.8%，其余 960 个概率 ≈ 0
- 均匀随机采样 → 95% 概率抽到垃圾 token → `fill_p_student_mean ≈ 6e-7`，fill KL 推不动

## 两阶段规划

### 阶段一：当前（占位池验证机制）

- **不纠结池子质量**，用现有 base model 分布的 962 token 池
- 目的：验证 fill 机制本身是否生效（fill_kl > 0、首 token distinct 上升、entropy 上升）
- **推理时屏蔽**：输出（rollout）阶段屏蔽 forced token，确保 forced token 只用于训练信号，不影响自然生成
- 判断标准：
  - 若 `first_token/distinct` 和 `first_token/entropy` 持续上升 → fill 机制有效
  - 若 `fill_p_student_mean` 从 1e-7 量级提升 → 模型在学会输出这些 token
  - 若无效 → 进入阶段二

### 阶段二：有意义的首 token 池

- 由用户提供一个有意义的首 token 池（如化学推理常见 opener、思维链启动词等）
- 替换 `output/after_reasoning_token_dist.json` 或新增配置路径
- 重新采收集分布文件，用有意义的 token 替代垃圾 subword 碎片
- 重新跑训练，观察 fill 效果是否改善

## 待实施

- [ ] 阶段一：rollout 输出阶段屏蔽 forced token（当前未实现）
- [ ] 阶段一：观察 fill 机制是否生效（fill_kl、distinct、entropy 趋势）
- [ ] 阶段二：用户提供有意义首 token 池
- [ ] 阶段二：替换池子重跑训练

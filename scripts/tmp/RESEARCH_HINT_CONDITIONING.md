# 调研: Hint-conditioned reasoning / 让 LLM 学会用 fill_token

调研日期: 2026-06-07
调研背景: 实验流程见 `EXP_PROGRESS_4096.md`。

## 问题定义

**目标**: 让 base 模型在 test-time 给 hint 后能做对硬题 (roll-8 全错的题)。

**已知现象**:
- Base (DeepSeek-R1-Distill-Qwen-7B) 首 token 输出**分布单一**, roll-8 (T=0.6, top_p=0.95) 实际只覆盖很少几个 token
- 但**强制塞** "模型不会自然 sample 出来的 token" 作为首 token + greedy 续写 → 86% 救回率 (`fill_multi_pool` 1221/1419)
- → **capability exists; sampling is the bottleneck** — 模型有能力, 只是采样把它锁在错误轨迹

**已试过无效**:
- vanilla SFT on rescue (fillonly LoRA): pass@1 -0.52% — 学不到
- SDFT v1 + v2 (`project-sdft-eval-result.md`): pool +0.40% — 学不到

## 5 个方向 (按契合度排序)

### 🔥 方向 3: Hindsight relabeling (最强契合)

**HIR — Hindsight Instruction Relabeling**
- arXiv: [2302.05206](https://arxiv.org/abs/2302.05206) (ICLR 2023)
- 方法: HER 在 LM 上的版本。任意 trajectory 反向标注成"模型该这么做的"; 不需 reward model
- 直接对应: `fill_multi_pool` 1221 题救回 = 现成的 hindsight 数据
- 落地: SFT `(question + [HINT:<rare_token>]) → rescued_trajectory`, test-time 让模型自己生成 `[HINT:?]`

**V-STaR**
- arXiv: [2402.06457](https://arxiv.org/abs/2402.06457) (COLM 2024)
- 方法: 同时学 generator 和 verifier, 保留 pos+neg 自生成轨迹做 DPO 训 verifier; +4-17% on math/code
- 落地: rescue = positive, base roll-8 失败 = negative, verifier rerank

**Amortizing Intractable Inference (GFlowNet-LM)**
- arXiv: [2310.04363](https://arxiv.org/abs/2310.04363) (ICLR 2024)
- 方法: GFlowNet 目标使采样频率 ∝ reward, **保证多模态覆盖**
- 落地: 理论最稳, 但工程贵, 留作 HIR 塌缩到单一 opener 时的兜底

### 🔥 方向 1: Diversifying — 防 mode collapse

**DivPO — Diverse Preference Optimization**
- arXiv: [2501.18101](https://arxiv.org/abs/2501.18101) (Meta, 2025-01)
- 方法: DPO 变种, chosen = "rare 但对", rejected = "common 但错"
- 落地: `(rare_first_token + 续写做对) vs (top-1 sample 做错)` → DPO LoRA, 防 HIR-SFT 后塌缩到单一 hint

**Adaptive Inference-Time Compute**
- arXiv: [2410.02725](https://arxiv.org/abs/2410.02725) (Stanford, 2024-10)
- 方法: 模型生成中间 self-eval token 预测 "restart 是否有用", 用作 prune/restart 信号
- 落地: 蒸馏一个 restart-head, 当前 run 判 hopeless 时从 376 token 池采样 restart, 比 roll-8 便宜

### 方向 2: Hint-augmented inference

**Buffer of Thoughts (BoT)**
- arXiv: [2406.04271](https://arxiv.org/abs/2406.04271) (NeurIPS 2024 Spotlight)
- 方法: meta-buffer 存 thought-templates, buffer-manager 检索并实例化; +11% Game24, +51% Checkmate
- 落地: 376 token 池 = 隐式 hint vocab, 形式化成 retrieval. `(problem embedding → winning opener)`, 不训练
- 用作: HIR/DivPO 都失败时的 fallback, 纯 test-time

### 方向 4: Lightweight conditioning (加新模块)

**Soft prefix / hint-encoder** (无单一 top-conf 论文, 自拼方案)
- 思路: 训小 encoder (e.g., 0.5B Qwen) 把 `problem → k 个 soft-prefix token` (在 base embedding 空间)
- 优化目标: prefix + base-greedy 重现 rescue trajectory
- 优点: base 权重不动 (实证已证 base 有能力)
- 工程: P-Tuning v2 mechanics + rescue data, 自己拼

### 方向 5: Self-distillation 改进版

**SDFT** (arXiv:[2402.13669](https://arxiv.org/abs/2402.13669), ACL 2024)
- 已试过, pool +0.40%, 无效

**SCoRe** (arXiv:[2409.12917](https://arxiv.org/abs/2409.12917), DeepMind 2024-09)
- +15.6% on MATH for Gemini 1.0 Pro, 多轮 RL 防 behavior collapse, 跟你诊断一致
- 工程**重**: 多轮 online RL + verifier 基础设施

**ReST-EM** (arXiv:[2312.06585](https://arxiv.org/abs/2312.06585), TMLR)
- generate→filter-by-correct→SFT 迭代, 经典 baseline; 用作对照

## 落地建议 (1-2 周)

**主路线**:
1. **HIR-style hint-token SFT** (方向 3) — 最高 ROI
   - 数据已现成: `(question + [HINT:<token>]) → rescued_trajectory` LoRA SFT
   - test-time: 让模型自己 generate `[HINT:?]`
2. **DivPO 叠加** (方向 1) — 防 hint 塌缩
   - chosen = rare opener rescue, rejected = common opener failure

**跳过**:
- SCoRe (太重, 多轮 RL 基础设施)
- GFlowNet-LM (工程贵, 留作 HIR 塌缩兜底)
- BoT (pure retrieval, 留作 fallback)

## 风险点 (我自己加的)

1. **HIR 的 test-time 难点**: 要求 hint token 能被自然采样, 但你的实证已经证明这正是问题源头。HIR 训练后能否让模型学会"自己生成 hint" — 这是关键, 失败的话只能在 test-time 给 hint
2. **DivPO 数据构造**: "common 但错"用 base roll-8 失败轨迹, 要选 first_token 高概率 + 答错的, 数据已有
3. **现有 fillonly 数据复用**: 3458 条样本接近 HIR 格式, 但训练目标差 — fillonly 是首 token CE + 后续 KL, HIR 是 prompt 加 hint 标签后续 SFT, 要改 trainer

## 完整 Sources

- [DivPO — arXiv:2501.18101](https://arxiv.org/abs/2501.18101) (Meta 2025-01)
- [V-STaR — arXiv:2402.06457](https://arxiv.org/abs/2402.06457) (COLM 2024)
- [SCoRe — arXiv:2409.12917](https://arxiv.org/abs/2409.12917) (DeepMind 2024-09)
- [SDFT — arXiv:2402.13669](https://arxiv.org/abs/2402.13669) (ACL 2024)
- [ReST-EM — arXiv:2312.06585](https://arxiv.org/abs/2312.06585) (TMLR)
- [GFlowNet-LM — arXiv:2310.04363](https://arxiv.org/abs/2310.04363) (ICLR 2024)
- [HIR — arXiv:2302.05206](https://arxiv.org/abs/2302.05206) (ICLR 2023)
- [Buffer of Thoughts — arXiv:2406.04271](https://arxiv.org/abs/2406.04271) (NeurIPS 2024 Spotlight)
- [Adaptive Inference-Time Compute — arXiv:2410.02725](https://arxiv.org/abs/2410.02725) (Stanford 2024-10)
- [Quiet-STaR — arXiv:2403.09629](https://arxiv.org/abs/2403.09629) (备用)

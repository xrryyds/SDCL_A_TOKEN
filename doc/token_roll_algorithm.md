# Token-Roll Policy Optimization（TRPO-branch / token_roll 模式）

> 本文档跟踪 token_roll 算法的设计、实现与实验记录。**算法有任何更新，同步更新本文档。**
>
> 最后更新：2026-07-29（v2：多轮强制尝试，成功即停，K = rollout n）

## 1. 动机与总体设计

参考 SRPO（`doc/grpo_router.md`：正确样本走 GRPO、失败样本走 SDPO 的样本级路由）。
本方法保留 SRPO 的整体框架，但**把失败样本的 SDPO 分支替换为 token 池强制首 token 重采样**：

1. 与 GRPO 相同：每个 prompt 采 G=8 条 rollout，按可验证奖励判分；
2. 对**错误样本**：最多进行 **K = rollout n（8）轮**强制重采样。每轮从首 token 池（`datasets/first_tokens_test.json`，376 个 token，取自 MATH 测试集人写解答的首 token 分布）随机抽 1 个 token，**强制作为模型输出的第一个真实 token**（非 chat 模板/think 标记），其后自由生成、重新判分；
3. **成功即停（first success wins）**：某轮变正确的样本立即退出重采样池、进入 token-roll 分支训练；每轮只对仍错误的样本继续重抽，K 轮后仍错误的样本丢弃。因此不存在"多个成功用哪条"的歧义，且算力随成功样本递减；
4. 计算损失时，token-roll 分支对**首 token 着重加权**（显式 CE），促使模型学会该首 token。

## 2. 样本路由

每步 batch = 32 prompt × 8 rollout = 256 条：

| 原始 rollout | 强制重roll结果（≤K=8 轮，成功即停） | 路由 | mask |
|---|---|---|---|
| 正确 | — | GRPO 分支 | `(1-token_roll_mask)(1-discard_mask)` |
| 错误 | 某轮变正确 | token-roll 分支（response 替换为该轮强制 rollout） | `token_roll_mask=1` |
| 错误 | K 轮后仍错误 | 丢弃 | `discard_mask=1` |

## 3. 损失函数

### 3.1 GRPO 分支（与 SRPO 论文 baseline 一致）

$$\mathcal{L}_{GRPO} = -\min\big(\rho_t A_i,\ \mathrm{clip}(\rho_t,\,1-\varepsilon,\,1+0.28)\,A_i\big)$$

- 组内优势 $A_i = r_i - \bar r$（`norm_adv_by_std_in_grpo=False`，不除 std）
- 序列级优势均匀分配到 token；带 token 级 rollout IS 修正（`rollout_is=token`，阈值 2.0）

### 3.2 Token-roll 分支

**首 token 交叉熵**（核心：显式监督强制 token，不经过 advantage）：

$$\mathcal{L}_{CE} = -\frac{1}{N_{tr}}\sum_{i\in tr}\log \pi_\theta(\text{forced\_token}_i \mid \text{prompt}_i)$$

在全词表 logits 上计算（token_roll 模式强制 `return_all_logps=True`）。

**其余 token 反向 KL**（第 2 个 token 起，锚向 ref 防漂移，不做强化）：

$$\mathcal{L}_{RKL} = \frac{1}{\sum mask}\sum_{t\ge 1}\big(\log\pi_\theta(y_t) - \log\pi_{ref}(y_t)\big)$$

**分支合计**：$\mathcal{L}_{TR} = w_{ce}\mathcal{L}_{CE} + w_{rkl}\mathcal{L}_{RKL}$，默认 $w_{ce}=w_{rkl}=1.0$。

### 3.3 总损失

$$\mathcal{L} = \mathcal{L}_{GRPO} + \mathcal{L}_{TR}$$

两分支 mask 互斥；丢弃样本不参与任何 loss。奖励信号通过"强制轨迹变正确才被选入训练"这一筛选事实进入 token-roll 分支。

## 4. 首 token 强制的实现（关键）

verl 该版本 `async_rollout_mode` 硬编码为 True（sync 已废弃），agent loop 从 `raw_prompt` 重建 prompt_ids、**忽略 input_ids 张量**。因此张量层面拼 token 无效，强制必须在 agent loop 内做：

- 新增 `ForcedFirstTokenAgentLoop`（注册名 `forced_first_token_agent`）：
  `prompt_ids = apply_chat_template(raw_prompt) + [forced_token_id]` 送引擎生成，
  返回时 `response_ids = [forced_token] + 自由生成`，使强制 token 归属 response 首位；
- `_build_forced_gen_batch` 通过 non-tensor 字段 `agent_name` / `forced_first_token_id` 逐样本传入；
- 强制 batch 走 `pad_dataproto_to_divisor`（worker 数对齐）；生成后补回 `reward_model`/`data_source` 等判分元信息并计算 `response_mask`。

## 5. 关键代码位置（`sdpo/SDPO/`）

| 功能 | 文件:位置 |
|---|---|
| 强制首 token agent loop | `verl/experimental/agent_loop/forced_first_token_agent_loop.py` |
| token 池加载 / 强制 batch 构建 / 路由与拼接 | `verl/trainer/ppo/ray_trainer.py` `_load_token_pool` / `_build_forced_gen_batch` / `_maybe_build_token_roll_batch` |
| fit() 中的调用点（判分之后） | `verl/trainer/ppo/ray_trainer.py` `fit()` token-roll 段 |
| 损失函数 | `verl/trainer/ppo/core_algos.py` `compute_token_roll_loss` |
| 分支路由与合并 | `verl/workers/actor/dp_actor.py` `update_policy`（`loss_mode == "token_roll"` 段） |
| 配置 dataclass | `verl/workers/config/actor.py` `TokenRollConfig` |
| 实验配置 / 启动脚本 | `verl/trainer/config/token_roll.yaml`、`run_local_token_roll.sh` |

## 6. 实验配置（当前）

- 模型：Qwen3-8B（thinking=False），2×H20，FSDP + vLLM async rollout（TP=2）
- 数据：SciKnowEval Chemistry（1890 train / 210 test，与 SDPO/SRPO 论文同 split）
- batch 32、rollout n=8、mini-batch 32、lr 5e-6、warmup 10、max_resp 8192
- token_roll：`success_reward_threshold=1.0`、`ce_loss_weight=1.0`、`reverse_kl_weight=1.0`、`num_forced_attempts=8`（=rollout n，逐样本成功即停）
- 本地适配：`attn_implementation=sdpa`（无 flash_attn）、`use_remove_padding=False`、actor param+optimizer offload、ref param offload（2 卡跑 8B 必需）、console 日志、40 步、每 10 步 val@16
- 监控指标：`token_roll/n_wrong`（路由到本分支数）、`n_correct_forced` 与 `forced_success_rate`（奖励信号数/比例）、`n_discarded`、`attempts_used`（实际轮数）、`n_forced_rollouts`（强制生成总条数=算力开销）、`ce_loss`、`reverse_kl_loss`；整体 `critic/rewards/mean`、`val-core/*`

## 7. 实验记录

### 2026-07-29 Chemistry 首跑（step 1 数据）

- 256 条 rollout 中 **145 条错误（56.6%）路由到 token-roll 分支**
- 强制首 token 后 **28 条变正确 → forced_success_rate = 19.3%**
- 117 条仍错丢弃；首 token CE = 3.70；反向 KL = 0（step 1 policy≡ref，正常）
- 结论：单次强制重roll即可为约 1/5 的失败样本恢复奖励信号，链路可用

## 8. 修复记录（实现层）

1. 补回缺失的 Hydra 配置组 `verl/trainer/config/model/hf_model.yaml`（否则所有 config 无法组合）
2. flash_attn 未安装 → `attn_implementation=sdpa` + `use_remove_padding=False`
3. 强制 rollout 判分 `KeyError: 'reward_model'` → 生成后补回 non-tensor 判分元信息并计算 `response_mask`
4. `only support equal chunk`（145/8）→ 强制 batch pad/unpad 到 worker 数整除
5. **首 token 强制实际未生效**（async agent loop 忽略 input_ids）→ 新增 `ForcedFirstTokenAgentLoop`（本算法语义的关键修复）
6. 指标汇总 `np.mean` inhomogeneous 崩溃 → 所有 micro-batch 发射相同 `token_roll/*` key 集合
7. step 2 vLLM wake_up OOM（优化器状态驻留）→ FSDP param/optimizer offload

## 9. 待观察 / 后续方向

- `forced_success_rate` 随训练的走势（上升 = 模型在内化首 token 分布；下降 = 错误样本变少后剩余为难例）
- token-roll 分支占比下降、GRPO 占比上升的自适应混合（与 SRPO 论文 Fig.5 同型的动态）
- 超参扫描：`ce_loss_weight`（学首 token 力度）、`reverse_kl_weight`（防漂移）
- token 池对比：`first_tokens_test.json`(376, MATH) vs `first_tokens_model_test.json`(254, 模型自生成)
- 扩展到其余 4 个基准（Physics / Biology / Materials / Tool Use）与 GRPO/SDPO 基线对比

## 10. 算法变更日志

- **v2（2026-07-29）**：单次强制尝试 → **最多 K=rollout n（8）轮，逐样本成功即停**。每轮仅对仍错误样本重抽 token；新增指标 `attempts_used`、`n_forced_rollouts`。预期把奖励信号率从单次 ~19% 提升到 ~1-(1-p)^8。实现：`ray_trainer.py::_maybe_build_token_roll_batch` 改为多轮循环；launcher 中 `NUM_FORCED_ATTEMPTS=$ROLLOUT_BATCH_SIZE`。
- **v1（2026-07-28）**：初版单次强制尝试；`ForcedFirstTokenAgentLoop` 实现真正的 token 级首 token 强制。

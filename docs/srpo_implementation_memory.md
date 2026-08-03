# SRPO 复现 + 全错组重 rollout —— 实现记忆

> 记录时间：2026-07-31。本文件是当前会话的进度记忆，供断开后续接。

## 一、目标

在 `sdpo/SDPO`（verl fork）里复现论文 `docs/srpo.md`（Sample-Routed Policy Optimization），
并在其基础上新增一个**全错组重 rollout 分支**（用户创新点）。

- **SRPO 论文**：正确 rollout → GRPO；错误且组内有 correct sibling → 熵动态加权 DW-SDPO 自蒸馏；
  全错组（无 teacher 信号）论文里只能 fallback GRPO 且 advantage=0（零信号）。
- **新分支**：全错组触发二次 rollout = 2 条自由采样 + 3 个互不相同的强制首 token × 各 2 条（共 8 条），
  强制首 token 取自**模型自身首 token 分布**（probe agent 采样统计 top-k distinct）；
  新 8 条整组替换原全错组，走标准 GRPO（有对有错就有信号；仍全错则 advantage 自然为 0）。

## 二、已实现（loss_mode: srpo）

| 文件 | 改动 |
|---|---|
| `verl/workers/config/actor.py` | 新增 `SRPOConfig`（enable_reroll / probe_* / dw_beta / dw_normalizer_scope），挂到 `ActorConfig.srpo` |
| `verl/experimental/agent_loop/first_token_probe_agent_loop.py` | 新 probe agent（max_tokens=1, T=1.0 采样统计首 token），已注册进 `__init__.py` |
| `verl/trainer/ppo/core_algos.py` | 新增 `compute_srpo_loss`：GRPO 分支(dual-clip) + DW-SDPO 分支(top-k JSD + 熵动态加权)，**共享同一 token 归一化分母**（论文式 144） |
| `verl/trainer/ppo/ray_trainer.py` | 新增 `_maybe_build_reroll_batch`（probe→重roll→整组替换 responses/mask/pos/rollout_log_probs/reward）+ fit() 钩子(2078后) + `_build_reroll_gen_batch` + `_run_reroll_generation`；改造 `_maybe_build_self_distillation_batch` 生成 `srpo_sdpo_mask = self_distillation_mask·(1-c)·(1-reroll_group_mask)`；新增 `_compute_first_token_stats`（首 token 塌缩监控） |
| `verl/workers/actor/dp_actor.py` | 新增 srpo loss 路由分支（teacher 前向复用 sdpo 路径→调 compute_srpo_loss）；`_update_teacher` 放宽到 srpo |
| `verl/workers/fsdp_workers.py` | EMA teacher 初始化条件放宽到 srpo（`loss_mode in ("sdpo","srpo")`） |
| `verl/trainer/main_ppo.py` | `self_distillation_needs_ref` 放宽到 srpo（走 colocated teacher / ActorRolloutRef） |
| `verl/trainer/config/srpo.yaml` + `config/actor/actor.yaml` | 论文超参配置 + srpo 子配置默认值 |
| `run_local_srpo.sh` | 启动脚本（console 日志、每 5 step 评测、step0 基线、支持追加 dry-run 覆盖参数、MODEL_PATH 可环境变量覆盖） |
| `tests/trainer/ppo/test_srpo_on_cpu.py` | CPU 单测（联合归一化 / 动态加权 / 路由 mask / reroll 槽位分配） |

**关键实现点**：
- reroll 钩子放在 old_log_prob/ref/advantage 之前 → 替换序列被后续前向天然覆盖。
- 替换组用 `srpo_reroll_group_mask` 强制走 GRPO，禁止进 SDPO 分支。
- reward 行替换只保证 per-sample sum 正确（GRPO advantage 只用 sum）。
- 联合归一化沿用 repo 既有 per-microbatch token-mean × loss_scale_factor 约定（与 sdpo/token_roll 一致）。
- probe distinct 不足降级：D 个 distinct → forced=min(3,D)，free=8-forced×2，组大小恒为 8。

## 三、监控指标 → 控制台字段

- 准确率：`val-core/.../acc`（step0 基线 + test_freq，avg@16，论文 Chemistry 基线≈41）
- 熵坍塌：`actor/entropy`↓、`first_token/entropy`↓、`first_token/top1_frac`→1
- 进入新分支数量：`srpo/n_all_wrong_groups`
- 新分支救回率：`srpo/reroll_new_correct_frac`
- 路由占比(图5)：`srpo/sdpo_frac`、`srpo/sdpo_sample_fraction`（论文初期 SDPO≈40%）
- teacher 信号：`srpo/teacher_entropy_mean`、`srpo/dw_weight_mean`
- 其他：`response_length/mean`、`srpo/probe_distinct_tokens_mean`、`srpo/reroll_degraded_groups`、`actor/pg_clipfrac`、`actor/grad_norm`

## 四、数据（与论文 Table 4 完全一致，官方划分，未重划分/打乱）

| benchmark | train/test |
|---|---|
| Chemistry | 1890 / 210 |
| Physics | 720 / 80 |
| Biology | 450 / 50 |
| Materials | 841 / 94 |
| Tool Use | 4046 / 68 |

prompt 模板即论文 Listing 1/2；reward `verl/utils/reward_score/feedback/`（mcq 提取 `<answer>` 字母 0/1）。

## 五、环境（environment.yml，name: srpo）

- torch 2.9.0+cu128 / vllm 0.12.0 / transformers 4.57.6 / tensordict 0.10 / ray 2.53
- **flash-attn 2.8.3.post1**：GitHub 直连不通，PyPI 无预编译 wheel，**源码编译**安装
  （`MAX_JOBS=20 TORCH_CUDA_ARCH_LIST="9.0" FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.8.3.post1 --no-build-isolation`）
- 注意 environment.yml 里 setuptools 82 删了 pkg_resources，需 `pip install "setuptools<81"` 修复。
- 建环境后 `cd sdpo/SDPO && pip install --no-deps -e .` 注册 verl。
- 模型 Qwen3-8B 在 `SDCL_A_TOKEN/model/Qwen/Qwen3-8B`（走 hf-mirror/modelscope 下的，16G）。

## 六、运行

```bash
conda activate srpo
cd sdpo/SDPO
python data/preprocess.py --data_source datasets/sciknoweval/chemistry   # 转 parquet
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# dry-run（小 batch/短 response/1 epoch）
MODEL_PATH=<repo>/model/Qwen/Qwen3-8B N_GPUS_PER_NODE=8 \
bash run_local_srpo.sh chem_dry \
  data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  data.max_response_length=1024 actor_rollout_ref.actor.srpo.probe_num_samples=8 \
  trainer.total_epochs=1 trainer.test_freq=2
# 全量：bash run_local_srpo.sh chem
```

## 七、dry-run 发现与首 token 注入点修正（2026-08-01）

**dry-run 结果（chemistry, 8×H20 单机, 短 response/1 epoch）**：
- ✅ 全流程稳定，无崩溃/NaN，`actor/grad_norm` 0.26~0.29，flash-attn 生效
- ✅ **step0 基线 `val-core/.../acc = 0.4125`**，精确命中论文 Chemistry Qwen3-8B 基线 41.1
- ✅ SRPO 路由正常：`srpo/sdpo_frac ≈ 0.35~0.40`，对上论文初期 SDPO≈40%
- ⚠️ **全错组重 roll 分支在 SciKnowEval 上失效**：`probe_distinct_tokens_mean≈1.5`、`reroll_degraded_groups`=全部、`reroll_new_correct_frac=0`

**根因**：SciKnowEval 强制 `<reasoning>...</reasoning><answer>` XML 格式，模型**首 token 恒为 `<`（id 27）**。
probe 探不出多样性；强塞别的首 token 破坏格式 → incorrect_format → 更错。

**修正（本次）**：注入点从"第 0 个输出 token"后移到**固定格式前缀之后的第一个有含义 token**。
- 新增 `SRPOConfig.reroll_prefix`（默认 `"<reasoning>\n"`；空串=位置0注入，兼容自由推理/数学任务）。
- probe：喂 `prompt + <reasoning>\n` 再采样 1 token → 首个真实推理 token 分布。
- forced：`prompt + <reasoning>\n + forced_token` 生成，response=`<reasoning>\n + forced_token + 续写`（格式合法且起点多样）。
- 经 per-sample `forced_prefix_ids` 传递；**token_roll 模式不设该字段，行为不变**。
- 改动：`workers/config/actor.py`、两个 agent_loop、`trainer/ppo/ray_trainer.py`、`config/srpo.yaml`、`config/actor/actor.yaml`。

> 该创新点更适合自由推理任务（原 MATH/DeepMath 线，首 token Okay/To/First 承载破局语义）；SciKnowEval 上收益可能有限。

## 八、强制 token 选择 + 方案C soft-teacher fill（2026-08-01，回应"GRPO 学不会"）

**问题**：救回的强制首 token 是模型自己 roll 不出来的（P≈0）。纯 GRPO 学不会它——
因为 rollout-IS 权重 = exp(old−rollout)，强制 token 的 old_log_prob≈−10、rollout 是镜像占位≈−1，
IS≈exp(−9)≈0 → 该位置策略梯度被压没。GRPO 只学到"给定开头怎么续写"，学不到"自己产生这个开头"。

**token 选择（回应用户）**：强制 token 从**模型该位置首 token 分布**里随机采（probe 用 `probe_temperature=1.5` 采出非默认的次优开头），
**排除该组 roll-8 已用过的默认首 token**（GRPO 已能 roll、且已证明无奖励）。每组随机取 `probe_num_forced_tokens=6` 个各不相同，各 roll 1 条 + 2 条自由 = 8。

**学习机制（复用旧线方案C，`a_token_sdcl_train.py:551-595`）**：对救回的强制样本，在首个有含义 token 位置加
`KL(student ‖ q')`，`q'=(1-β)·q_teacher + β·onehot(k)`，**β=`forced_fill_beta`=0.5 是上限**：
- 只把 P(k) 温和抬到 ~β，**不推向 1** → 不挤占其它 token → **无灾难遗忘**；softmax 耦合保留其余相对形状；
- 抬起后 **GRPO 主分支自然 roll 出 k 并用 advantage 强化**（两段式：先抬概率、后 RL 加强）。
- 实现用 {k, rest} 二元 KL（只需该位置 student/teacher 对 k 的逐 token logp，省全词表），`forced_fill_weight` 加权。
- 新增 config：`forced_fill_beta=0.5`、`forced_fill_weight=1.0`、`probe_temperature=1.5`；
  新增 metric：`srpo/fill_kl`、`srpo/fill_p_student_mean`、`srpo/fill_n`、`srpo/reroll_rescued_forced`、`srpo/reroll_excluded_first_tokens_mean`。
- 改动：`workers/config/actor.py`、两个 agent_loop、`trainer/ppo/{ray_trainer,core_algos}.py`、`workers/actor/dp_actor.py`、两处 yaml。

> 组成定为 6 forced×1 + 2 free；forced token 随机采自模型分布且排除默认。CE(硬 onehot)会 P→1 灾难遗忘，故用方案C 带上限的软化 KL。

## 九、fill-rescue 诊断 + 强制 token 来源改为 opener 池（2026-08-01）

**诊断脚本** `scripts/diag_fill_rescue.py`：chemistry train 前 200 题 roll-8(T=1.0) 找全错(无信号)题，
对同一批无信号题,两种来源各填充首 token(排除该组已用默认)后贪心生成判分。结果:

| 项 | 值 |
|---|---|
| 无信号题 | 43/200 (21.5%) |
| A 模型首 token 分布(读 top-20 logit,排除默认) | **46.5%** (20/43) |
| B 广撒 opener 池(top-50,排除默认) | **69.8%** (30/43) |
| 任一 | 72.1% (31/43)；A∩B=19，**A独有=1，B独有=11** |

**结论**：(1) 填充能救回 **72%** 的 GRPO 无信号数据,idea 强成立；
(2) **池 ≫ 模型分布(+23pp)**——能破局的 opener 大多**不在**模型自己的首 token 分布里(模型想不到),
必须强制分布外 token 再靠方案C 教会；(3) 训练里旧的采样 probe 只冒出那 2 个默认,比诊断里"读top-20 logit"的 46.5% 还弱,故 fill 几乎不触发。

**据此改动(数据驱动)**：强制 token 来源 **model 采样 probe → 外部 opener 池**：
- 新增 `srpo.forced_token_pool_path`(指向 `datasets/first_tokens_test.json`,376 opener)；
- `_maybe_build_reroll_batch`:**删除采样 probe**,候选=池−该组默认首token,随机取 `probe_num_forced_tokens`(6),靠跨 step/epoch 复现累积覆盖；
- 新增 `_load_forced_token_pool()`(带缓存)；池缺失则跳过 reroll(`srpo/forced_pool_loaded=0`)；
- 前缀 `<reasoning>\n` 注入 + 方案C 软化 fill(β=0.5)不变；`first_token_probe_agent` 保留但训练不再调用。
- 改动:`workers/config/actor.py`、`trainer/ppo/ray_trainer.py`、两处 yaml。

> 这条推翻了"强制 token 限定模型自己分布"的早期方向——诊断证明那样丢掉 ~23pp 可救回题。

## 十、当前状态 / 待办

- [x] 全部代码实现 + CPU 单测 + 编译通过
- [x] srpo conda 环境建好，flash-attn 已编译安装
- [x] Qwen3-8B 权重下载完成（16G，5 分片）
- [ ] **dry-run 未验证**：8×H20 上首次启动遇到 flash-attn 缺失（已装好），待重跑确认
      `srpo/*` 指标正常（sdpo_frac+grpo≈1、reroll 命中率非零、grad_norm 有限、loss 不 NaN）
- [ ] dry-run 通过后跑全量 Chemistry（30 epoch），再扩到 physics/biology/material/tooluse
- [ ] 与论文 baseline（Qwen3-8B 5 benchmark 平均 GRPO 74.0 / SDPO 71.1 / SRPO 77.4）对比，
      验证全错组重 roll 分支能否进一步提升

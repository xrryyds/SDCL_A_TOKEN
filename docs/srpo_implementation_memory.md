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

## 七、当前状态 / 待办

- [x] 全部代码实现 + CPU 单测 + 编译通过
- [x] srpo conda 环境建好，flash-attn 已编译安装
- [x] Qwen3-8B 权重下载完成（16G，5 分片）
- [ ] **dry-run 未验证**：8×H20 上首次启动遇到 flash-attn 缺失（已装好），待重跑确认
      `srpo/*` 指标正常（sdpo_frac+grpo≈1、reroll 命中率非零、grad_norm 有限、loss 不 NaN）
- [ ] dry-run 通过后跑全量 Chemistry（30 epoch），再扩到 physics/biology/material/tooluse
- [ ] 与论文 baseline（Qwen3-8B 5 benchmark 平均 GRPO 74.0 / SDPO 71.1 / SRPO 77.4）对比，
      验证全错组重 roll 分支能否进一步提升

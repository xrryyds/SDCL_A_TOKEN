#!/usr/bin/env bash
# 一键端到端跑 GRPO 三池流水线 Step 1-3,所有产出落到 logs/grpo_pipeline_<ts>/
# 用法:
#   bash scripts/run_grpo_pipeline.sh
#   bash scripts/run_grpo_pipeline.sh --skip-step1   # 跳 hot-swap harness(已绿过)
#   bash scripts/run_grpo_pipeline.sh --skip-step2   # 跳 grpo_pool 重建(已有就重用)
#
# 跑完把 LOG_DIR 整目录贴回来(或里面的 audit_*.txt + step*.log)。
#
# 关键:无论正常退出 / 报错 / Ctrl-C,最后都会 (1) cat 审计文件 (2) 调 use_worker 保活,
# 让 GPU 不被回收,可继续跑下一步 / 排查问题。

# 注意:这里故意不开 set -e。逐 step 自己检查并落日志,失败也要走到 finally。
set -u
cd "$(dirname "$0")/.."   # 回到 project root
PROJECT_ROOT="$(pwd)"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/grpo_pipeline_${TS}"
mkdir -p "${LOG_DIR}"

GRPO_POOL="datasets/exam/grpo_DS_MATH_pool.json"
MERGED="datasets/exam/a_token_train_data_with_grpo.json"
MODEL_PATH="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"

SKIP_STEP1=0
SKIP_STEP2=0
for arg in "$@"; do
  case "$arg" in
    --skip-step1) SKIP_STEP1=1 ;;
    --skip-step2) SKIP_STEP2=1 ;;
  esac
done

# ───────────────────────────────────────────────────────────────────
# finally:无论怎么退出都跑一次 (cat 审计 → use_worker 保活)
# ───────────────────────────────────────────────────────────────────
_pipeline_finally() {
  local exit_code=$?
  echo
  echo "================== PIPELINE FINALLY (exit=${exit_code}) =================="
  echo "logs in: ${LOG_DIR}"
  ls -lh "${LOG_DIR}" 2>/dev/null || true

  for f in env.log audit_grpo_pool.txt audit_merged.txt; do
    if [ -f "${LOG_DIR}/${f}" ]; then
      echo
      echo "--------- ${f} ---------"
      cat "${LOG_DIR}/${f}"
    fi
  done

  echo
  echo "================== use_worker 保活 (Ctrl-C 退出) =================="
  # 即使前面流程出错,也保活 GPU,方便排查/续跑
  python -u -c "
import traceback
try:
    from main import use_worker
    use_worker()
except BaseException:
    traceback.print_exc()
" 2>&1 | tee -a "${LOG_DIR}/use_worker.log"
}
trap _pipeline_finally EXIT

# 环境快照(便于事后核对版本)
{
  echo "========== ENV SNAPSHOT =========="
  echo "ts=${TS}"
  echo "pwd=${PROJECT_ROOT}"
  echo "git HEAD=$(git rev-parse HEAD 2>/dev/null || echo NA)"
  echo "git status:"; git status -s 2>/dev/null || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "python=$(which python)"
  python --version 2>&1
  echo "---- nvidia-smi (head) ----"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
  echo "---- pip key versions ----"
  python -c "import torch, vllm, transformers, peft; print('torch=', torch.__version__); print('vllm=', vllm.__version__); print('transformers=', transformers.__version__); print('peft=', peft.__version__)" 2>&1 || true
} | tee "${LOG_DIR}/env.log"

echo
echo "========== LOG_DIR = ${LOG_DIR} =========="
echo

# -------------------------------------------------------------------
# Step 1: hot-swap harness
# -------------------------------------------------------------------
if [ "${SKIP_STEP1}" -eq 0 ]; then
  echo "[step1] vLLM colocate + LoRA hot-swap harness ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_STEP1:-0}" \
    python -u scripts/train/test_grpo_rollout_engine.py --colocate_alloc \
    2>&1 | tee "${LOG_DIR}/step1_hotswap.log"
  if grep -q "DONE 全部检查通过" "${LOG_DIR}/step1_hotswap.log"; then
    echo "[step1] PASS" | tee -a "${LOG_DIR}/step1_hotswap.log"
  else
    echo "[step1] WARN: 未检测到 'DONE 全部检查通过' 字样,请人工查 step1_hotswap.log" \
      | tee -a "${LOG_DIR}/step1_hotswap.log"
  fi
else
  echo "[step1] SKIPPED (--skip-step1)" | tee "${LOG_DIR}/step1_hotswap.log"
fi

# -------------------------------------------------------------------
# Step 2: 构 grpo_pool
# -------------------------------------------------------------------
if [ "${SKIP_STEP2}" -eq 0 ] || [ ! -f "${GRPO_POOL}" ]; then
  echo "[step2] build grpo_pool (4-card) ..."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_STEP2:-0,1,2,3}" \
    python -u scripts/build_grpo_pool.py 2>&1 | tee "${LOG_DIR}/step2_build_grpo_pool.log"
else
  echo "[step2] SKIPPED (file exists, --skip-step2)" | tee "${LOG_DIR}/step2_build_grpo_pool.log"
  echo "       reuse: ${GRPO_POOL}" | tee -a "${LOG_DIR}/step2_build_grpo_pool.log"
fi

echo "[step2-audit] auditing grpo_pool ..."
python -u scripts/audit_grpo_pipeline.py \
  --stage grpo_pool \
  --grpo_pool_path "${GRPO_POOL}" \
  --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/step2_audit_console.log"

# -------------------------------------------------------------------
# Step 3: merge train_data
# -------------------------------------------------------------------
echo "[step3] merge train_data with grpo source ..."
python -u scripts/train/a_token_sdcl.py merge \
  --grpo_path "${GRPO_POOL}" \
  --output_path "${MERGED}" \
  2>&1 | tee "${LOG_DIR}/step3_merge.log"

echo "[step3-audit] auditing merged train_data ..."
python -u scripts/audit_grpo_pipeline.py \
  --stage merged \
  --merged_path "${MERGED}" \
  --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/step3_audit_console.log"

# -------------------------------------------------------------------
# Step 4 提示(不自动跑,smoke 由人按需启动)
# -------------------------------------------------------------------
{
  echo
  echo "================== ALL STEPS DONE =================="
  echo "logs in: ${LOG_DIR}"
  ls -lh "${LOG_DIR}"
  echo
  echo "下一步(单卡 5-step smoke,不在本脚本内,按需跑):"
  echo "  CUDA_VISIBLE_DEVICES=0 python scripts/train/a_token_sdcl_train.py \\"
  echo "      --model_path ${MODEL_PATH} \\"
  echo "      --data_path ${MERGED} \\"
  echo "      --output_dir output/grpo_smoke_\$(date +%Y%m%d_%H%M%S) \\"
  echo "      --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 1 \\"
  echo "      --max_prompt_length 2048 --max_answer_length 4096 \\"
  echo "      --enable_grpo --grpo_n 4 --grpo_max_tokens 512 \\"
  echo "      --grpo_lora_sync_every 2 --log_interval 1 --save_total_limit 0 \\"
  echo "      2>&1 | tee ${LOG_DIR}/step4_smoke.log"
} | tee -a "${LOG_DIR}/SUMMARY.log"

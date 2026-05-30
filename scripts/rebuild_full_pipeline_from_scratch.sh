#!/usr/bin/env bash
# 从零重生 SDCL 全套数据池：mistake / corr / fill_correct / grpo + merge train_data
#
# 阶段:
#   A. 备份并清空旧池
#   B. take_exam baseline + teacher_mark 拆 mistake / corr 池
#      (= scripts/rebuild_math_pool.py)
#   C. generate_fill_correct: 对 mistake 池跑随机首 token 填充评测 → fill_correct.json
#      (= scripts/train/a_token_sdcl.py 默认入口)
#   D. exam_roll_recheck_mistake: rolling-8 救回 → 增量进 corr + 写 grpo_pool
#      (= scripts/build_grpo_pool.py)
#   E. merge_to_train_data: corr + fill + grpo → a_token_train_data_with_grpo.json
#      (= scripts/train/a_token_sdcl.py merge --grpo_path)
#   F. audit grpo_pool + merged
#
# 用法 (4 卡 H800):
#   cd /workspace/SDCL_A_TOKEN
#   git pull origin main
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/rebuild_full_pipeline_from_scratch.sh
#
# 跳过某些已绿过的阶段(逗号分隔):
#   bash scripts/rebuild_full_pipeline_from_scratch.sh --skip A,B
#   --skip A   跳备份清空 (复用现有 _backup_<ts>/)
#   --skip B   跳 take_exam (复用现有 mistake/corr 池)
#   --skip C   跳 fill_correct 生成 (复用现有 fill_correct.json)
#   --skip D   跳 grpo_pool 生成 (复用现有 grpo_DS_MATH_pool.json)
#
# 退出策略: 任何阶段失败都不会中止后续 (set -e 不开),最后 finally 一律调
#          use_worker 保活,并 cat 各阶段 log 尾巴 + audit 报告。

set -u
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/rebuild_full_${TS}"
BACKUP_DIR="datasets/exam/_backup_${TS}"
mkdir -p "${LOG_DIR}"

EXAM_DIR="datasets/exam"
MODEL_PATH="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"

# 关键文件
F_MISTAKE_4096="${EXAM_DIR}/mistake_collection_book_4096.json"
F_CORR_4096="${EXAM_DIR}/corr_answer_4096.json"
F_MISTAKE_POOL="${EXAM_DIR}/mistake_DS_MATH_pool.json"
F_CORR_POOL="${EXAM_DIR}/corr_DS_MATH_pool.json"
F_FILL="${EXAM_DIR}/fill_correct.json"
F_GRPO_POOL="${EXAM_DIR}/grpo_DS_MATH_pool.json"
F_MERGED="${EXAM_DIR}/a_token_train_data_with_grpo.json"

# 解析 --skip
SKIP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip) SKIP="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
_skipped() { echo ",${SKIP}," | grep -q ",$1," ; }

# ───────────────────────────────────────────────────────────────────
# finally: 任何退出路径都跑
# ───────────────────────────────────────────────────────────────────
_finally() {
  local exit_code=$?
  echo
  echo "================== FINALLY (exit=${exit_code}) =================="
  echo "logs in: ${LOG_DIR}"
  ls -lh "${LOG_DIR}" 2>/dev/null || true

  for f in env.log stage_*.log audit_grpo_pool.txt audit_merged.txt SUMMARY.log; do
    if [ -f "${LOG_DIR}/${f}" ]; then
      echo
      echo "--------- ${f} ---------"
      tail -80 "${LOG_DIR}/${f}"
    fi
  done

  echo
  echo "================== use_worker 保活 (Ctrl-C 退出) =================="
  python -u -c "
import traceback
try:
    from main import use_worker
    use_worker()
except BaseException:
    traceback.print_exc()
" 2>&1 | tee -a "${LOG_DIR}/use_worker.log"
}
trap _finally EXIT

# ───────────────────────────────────────────────────────────────────
# 环境快照
# ───────────────────────────────────────────────────────────────────
{
  echo "========== ENV SNAPSHOT =========="
  echo "ts=${TS}"
  echo "pwd=${PROJECT_ROOT}"
  echo "git HEAD=$(git rev-parse HEAD 2>/dev/null || echo NA)"
  git status -s 2>/dev/null || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  python --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
  python -c "import torch, vllm, transformers, peft; print('torch=', torch.__version__); print('vllm=', vllm.__version__); print('transformers=', transformers.__version__); print('peft=', peft.__version__)" 2>&1 || true
  echo
  echo "SKIP=${SKIP}"
  echo "BACKUP_DIR=${BACKUP_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
} | tee "${LOG_DIR}/env.log"

echo
echo "========== LOG_DIR = ${LOG_DIR} =========="
echo

# ───────────────────────────────────────────────────────────────────
# Stage A: 备份并清空
# ───────────────────────────────────────────────────────────────────
if _skipped A; then
  echo "[stageA] SKIPPED" | tee "${LOG_DIR}/stage_A_backup.log"
else
  mkdir -p "${BACKUP_DIR}"
  {
    echo "[stageA] backup datasets/exam/*.json → ${BACKUP_DIR}"
    for f in "${F_MISTAKE_4096}" "${F_CORR_4096}" "${F_MISTAKE_POOL}" "${F_CORR_POOL}" \
             "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}" \
             "${EXAM_DIR}/exam.json" "${EXAM_DIR}/a_token_train_data.json"; do
      if [ -f "$f" ]; then
        cp -v "$f" "${BACKUP_DIR}/" 2>&1
      fi
    done
    echo
    echo "[stageA] removing live pool files (kept in ${BACKUP_DIR})"
    for f in "${F_MISTAKE_4096}" "${F_CORR_4096}" "${F_MISTAKE_POOL}" "${F_CORR_POOL}" \
             "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}"; do
      [ -f "$f" ] && rm -v "$f" 2>&1
    done
  } 2>&1 | tee "${LOG_DIR}/stage_A_backup.log"
fi

# ───────────────────────────────────────────────────────────────────
# Stage B: take_exam baseline + teacher_mark → mistake/corr 池
# ───────────────────────────────────────────────────────────────────
if _skipped B; then
  echo "[stageB] SKIPPED (reuse existing pools)" | tee "${LOG_DIR}/stage_B_takeexam.log"
else
  echo "[stageB] take_exam + teacher_mark (重生 mistake/corr 池, ~2-3h on 4 H800) ..."
  # 内联调用底层函数,绕开 rebuild_math_pool.py finally 里的 use_worker(会卡住外层流水线)
  python -u <<'PY' 2>&1 | tee "${LOG_DIR}/stage_B_takeexam.log"
import os, shutil, sys, traceback
sys.path.insert(0, os.getcwd())

EXAM_DIR = "datasets/exam"
INTER_MISTAKE = os.path.join(EXAM_DIR, "mistake_collection_book_4096.json")
INTER_CORR    = os.path.join(EXAM_DIR, "corr_answer_4096.json")
POOL_MISTAKE  = os.path.join(EXAM_DIR, "mistake_DS_MATH_pool.json")
POOL_CORR     = os.path.join(EXAM_DIR, "corr_DS_MATH_pool.json")

try:
    print("[stageB] step 1/3: student_take_exam_Math_sub (MATH train, 6144/4096) ...", flush=True)
    from main import student_take_exam_Math_sub
    student_take_exam_Math_sub(
        train=True, subset="all", lora_path=None,
        max_prompt_length=6144, max_new_tokens=4096,
    )

    print("[stageB] step 2/3: teacher_mark_paper_with_save ...", flush=True)
    from scripts import TeacherCorrecter
    t = TeacherCorrecter()
    t.teacher_mark_paper_with_save()
    del t

    print("[stageB] step 3/3: copy intermediate → pool names ...", flush=True)
    assert os.path.exists(INTER_MISTAKE) and os.path.exists(INTER_CORR), \
        f"中间文件缺失: {INTER_MISTAKE} 或 {INTER_CORR}"
    shutil.copy2(INTER_MISTAKE, POOL_MISTAKE)
    shutil.copy2(INTER_CORR,    POOL_CORR)
    import json
    m = json.load(open(POOL_MISTAKE)); c = json.load(open(POOL_CORR))
    tot = len(m) + len(c)
    acc = (len(c) / tot * 100) if tot else 0.0
    print(f"[stageB] DONE  mistake={len(m)}  corr={len(c)}  total={tot}  acc={acc:.2f}%", flush=True)
except BaseException:
    traceback.print_exc()
    # 不挂 use_worker,让外层 trap 接管
PY
fi

# 验证 stageB 产物
{
  echo "--- stageB outputs ---"
  for f in "${F_MISTAKE_POOL}" "${F_CORR_POOL}" "${F_MISTAKE_4096}" "${F_CORR_4096}"; do
    if [ -f "$f" ]; then
      n=$(python -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
      echo "  $f  entries=${n}"
    else
      echo "  $f  MISSING"
    fi
  done
} | tee -a "${LOG_DIR}/stage_B_takeexam.log"

# ───────────────────────────────────────────────────────────────────
# Stage C: generate_fill_correct
# ───────────────────────────────────────────────────────────────────
if _skipped C; then
  echo "[stageC] SKIPPED (reuse fill_correct.json)" | tee "${LOG_DIR}/stage_C_fill.log"
elif [ ! -f "${F_MISTAKE_POOL}" ]; then
  echo "[stageC] SKIP: ${F_MISTAKE_POOL} not found, can't generate fill_correct" \
    | tee "${LOG_DIR}/stage_C_fill.log"
else
  echo "[stageC] generate_fill_correct (随机首 token 填充评测) ..."
  python -u scripts/train/a_token_sdcl.py \
    --output_path "${F_FILL}" \
    2>&1 | tee "${LOG_DIR}/stage_C_fill.log"
fi

{
  echo "--- stageC output ---"
  if [ -f "${F_FILL}" ]; then
    n=$(python -c "import json; print(len(json.load(open('${F_FILL}'))))" 2>/dev/null || echo "?")
    echo "  ${F_FILL}  entries=${n}"
  else
    echo "  ${F_FILL}  MISSING"
  fi
} | tee -a "${LOG_DIR}/stage_C_fill.log"

# ───────────────────────────────────────────────────────────────────
# Stage D: exam_roll_recheck_mistake → grpo_pool + 增量 corr
# ───────────────────────────────────────────────────────────────────
if _skipped D; then
  echo "[stageD] SKIPPED (reuse grpo_pool)" | tee "${LOG_DIR}/stage_D_grpo.log"
elif [ ! -f "${F_MISTAKE_POOL}" ]; then
  echo "[stageD] SKIP: ${F_MISTAKE_POOL} not found" | tee "${LOG_DIR}/stage_D_grpo.log"
else
  echo "[stageD] build grpo_pool (rolling-8 救回, ~30min on 4 H800) ..."
  # 内联调用 exam_roll_recheck_mistake,绕开 build_grpo_pool.py finally 里的 use_worker
  GRPO_POOL_PATH="${F_GRPO_POOL}" python -u <<'PY' 2>&1 | tee "${LOG_DIR}/stage_D_grpo.log"
import os, sys, json, traceback
sys.path.insert(0, os.getcwd())

grpo_pool_path = os.environ["GRPO_POOL_PATH"]
print(f"[stageD] grpo_pool_path = {grpo_pool_path}", flush=True)
print(f"[stageD] k=8 T=0.6 top_p=0.95 max_prompt=6144 max_token=4096", flush=True)

try:
    from main import exam_roll_recheck_mistake
    exam_roll_recheck_mistake(
        use_lora=False, lora_path="",
        max_token=4096, max_prompt_length=6144,
        k=8, temperature=0.6, top_p=0.95,
        grpo_pool_path=grpo_pool_path,
    )
except BaseException:
    traceback.print_exc()

# 产出汇总
if os.path.exists(grpo_pool_path):
    g = json.load(open(grpo_pool_path))
    print(f"[stageD] grpo_pool entries = {len(g)}", flush=True)
    if g:
        s = g[0]
        print(f"[stageD] sample[0]: q_idx={s.get('question_idx')} "
              f"n_correct={s.get('n_correct_of_k')}/{s.get('k')} "
              f"first_id={s.get('anchor_first_token_id')} "
              f"first_text={s.get('anchor_first_token_text')!r}", flush=True)
else:
    print(f"[stageD] WARN: {grpo_pool_path} 未生成", flush=True)
PY
fi

# ───────────────────────────────────────────────────────────────────
# Stage E: merge → a_token_train_data_with_grpo.json
# ───────────────────────────────────────────────────────────────────
echo "[stageE] merge corr + fill + grpo → ${F_MERGED} ..."
python -u scripts/train/a_token_sdcl.py merge \
  --corr_answer_path "${F_CORR_POOL}" \
  --fill_correct_path "${F_FILL}" \
  --grpo_path "${F_GRPO_POOL}" \
  --output_path "${F_MERGED}" \
  2>&1 | tee "${LOG_DIR}/stage_E_merge.log"

# ───────────────────────────────────────────────────────────────────
# Stage F: audit
# ───────────────────────────────────────────────────────────────────
echo "[stageF-1] audit grpo_pool ..."
python -u scripts/audit_grpo_pipeline.py \
  --stage grpo_pool \
  --grpo_pool_path "${F_GRPO_POOL}" \
  --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/stage_F1_audit_grpo_console.log"

echo "[stageF-2] audit merged ..."
python -u scripts/audit_grpo_pipeline.py \
  --stage merged \
  --merged_path "${F_MERGED}" \
  --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/stage_F2_audit_merged_console.log"

# ───────────────────────────────────────────────────────────────────
# Final summary
# ───────────────────────────────────────────────────────────────────
{
  echo
  echo "================== ALL STAGES DONE =================="
  echo "logs in: ${LOG_DIR}"
  echo "backups in: ${BACKUP_DIR}"
  echo
  echo "Pool sizes:"
  for f in "${F_MISTAKE_POOL}" "${F_CORR_POOL}" "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}"; do
    if [ -f "$f" ]; then
      n=$(python -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "?")
      printf "  %-60s %s\n" "$f" "${n} entries"
    fi
  done
  echo
  echo "下一步 (smoke test):"
  echo "  CUDA_VISIBLE_DEVICES=0 python scripts/train/a_token_sdcl_train.py \\"
  echo "      --model_path ${MODEL_PATH} \\"
  echo "      --data_path ${F_MERGED} \\"
  echo "      --output_dir output/grpo_smoke_\$(date +%Y%m%d_%H%M%S) \\"
  echo "      --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 1 \\"
  echo "      --max_prompt_length 2048 --max_answer_length 4096 \\"
  echo "      --enable_grpo --grpo_n 4 --grpo_max_tokens 512 \\"
  echo "      --grpo_lora_sync_every 2 --log_interval 1 --save_total_limit 0"
} | tee "${LOG_DIR}/SUMMARY.log"

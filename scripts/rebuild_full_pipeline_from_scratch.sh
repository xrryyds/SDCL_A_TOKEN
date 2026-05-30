#!/usr/bin/env bash
# 从零重生 SDCL 全套数据池(quiet 版):控制台只打 stage 头 + 状态 + 计数;
# 详细输出全落到 logs/rebuild_full_<ts>/stage_*.log。
#
# 阶段:
#   A. 备份并清空旧池
#   B. take_exam baseline + teacher_mark → mistake/corr 池
#   C. generate_fill_correct → fill_correct.json
#   D. exam_roll_recheck_mistake → grpo_pool + 增量 corr
#   E. merge → a_token_train_data_with_grpo.json
#   F. audit grpo_pool + merged
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/rebuild_full_pipeline_from_scratch.sh
#   bash scripts/rebuild_full_pipeline_from_scratch.sh --skip A,B
#
# 退出策略: set -e 不开;任何 stage 失败都不中止;最后 trap finally 调 use_worker 保活。

set -u
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/rebuild_full_${TS}"
BACKUP_DIR="datasets/exam/_backup_${TS}"
mkdir -p "${LOG_DIR}"

EXAM_DIR="datasets/exam"
MODEL_PATH="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"

F_MISTAKE_4096="${EXAM_DIR}/mistake_collection_book_4096.json"
F_CORR_4096="${EXAM_DIR}/corr_answer_4096.json"
F_MISTAKE_POOL="${EXAM_DIR}/mistake_DS_MATH_pool.json"
F_CORR_POOL="${EXAM_DIR}/corr_DS_MATH_pool.json"
F_FILL="${EXAM_DIR}/fill_correct.json"
F_GRPO_POOL="${EXAM_DIR}/grpo_DS_MATH_pool.json"
F_MERGED="${EXAM_DIR}/a_token_train_data_with_grpo.json"

SKIP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip) SKIP="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
_skipped() { echo ",${SKIP}," | grep -q ",$1," ; }

# helpers ─────────────────────────────────────────────────────────────
_count() {
  # _count <json_path>  → echo len 或 "?" 或 "MISSING"
  local f="$1"
  if [ ! -f "$f" ]; then echo "MISSING"; return; fi
  python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f" 2>/dev/null || echo "?"
}
_say()  { printf "%s\n" "$*"; }
_head() { printf "\n=== %s ===\n" "$*"; }

# finally ─────────────────────────────────────────────────────────────
_finally() {
  local exit_code=$?
  _head "FINALLY (exit=${exit_code})"
  _say "logs:    ${LOG_DIR}"
  _say "backups: ${BACKUP_DIR}"
  _say
  _say "Pool sizes:"
  for f in "${F_MISTAKE_POOL}" "${F_CORR_POOL}" "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}"; do
    printf "  %-60s %s\n" "$f" "$(_count "$f")"
  done

  for af in audit_grpo_pool.txt audit_merged.txt; do
    if [ -f "${LOG_DIR}/${af}" ]; then
      _head "${af}"
      cat "${LOG_DIR}/${af}"
    fi
  done

  _head "use_worker 保活 (Ctrl-C 退出)"
  python -u -c "
import traceback
try:
    from main import use_worker
    use_worker()
except BaseException:
    traceback.print_exc()
" >> "${LOG_DIR}/use_worker.log" 2>&1 &
  WORKER_PID=$!
  _say "use_worker pid=${WORKER_PID}  (log: ${LOG_DIR}/use_worker.log)"
  wait "${WORKER_PID}" 2>/dev/null || true
}
trap _finally EXIT

# env snapshot ────────────────────────────────────────────────────────
{
  echo "ts=${TS}"
  echo "pwd=${PROJECT_ROOT}"
  echo "git HEAD=$(git rev-parse HEAD 2>/dev/null || echo NA)"
  git status -s 2>/dev/null || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  python --version 2>&1
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>&1 || true
  python -c "import torch, vllm, transformers, peft; print('torch=', torch.__version__); print('vllm=', vllm.__version__); print('transformers=', transformers.__version__); print('peft=', peft.__version__)" 2>&1 || true
  echo "SKIP=${SKIP}"
  echo "BACKUP_DIR=${BACKUP_DIR}"
  echo "LOG_DIR=${LOG_DIR}"
} > "${LOG_DIR}/env.log" 2>&1

_say "LOG_DIR = ${LOG_DIR}"

# Stage A ─────────────────────────────────────────────────────────────
_head "Stage A: backup + wipe"
if _skipped A; then
  _say "  SKIPPED"
  echo "[stageA] SKIPPED" > "${LOG_DIR}/stage_A_backup.log"
else
  mkdir -p "${BACKUP_DIR}"
  {
    echo "[stageA] backup → ${BACKUP_DIR}"
    for f in "${F_MISTAKE_4096}" "${F_CORR_4096}" "${F_MISTAKE_POOL}" "${F_CORR_POOL}" \
             "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}" \
             "${EXAM_DIR}/exam.json" "${EXAM_DIR}/a_token_train_data.json"; do
      [ -f "$f" ] && cp -v "$f" "${BACKUP_DIR}/"
    done
    echo
    echo "[stageA] removing live pool files"
    for f in "${F_MISTAKE_4096}" "${F_CORR_4096}" "${F_MISTAKE_POOL}" "${F_CORR_POOL}" \
             "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}"; do
      [ -f "$f" ] && rm -v "$f"
    done
  } > "${LOG_DIR}/stage_A_backup.log" 2>&1
  _say "  done  (log: ${LOG_DIR}/stage_A_backup.log)"
fi

# Stage B ─────────────────────────────────────────────────────────────
_head "Stage B: take_exam + teacher_mark"
if _skipped B; then
  _say "  SKIPPED"
  echo "[stageB] SKIPPED" > "${LOG_DIR}/stage_B_takeexam.log"
else
  _say "  running ... (~2-3h on 4 H800)"
  python -u <<'PY' > "${LOG_DIR}/stage_B_takeexam.log" 2>&1
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
PY
  rc=$?
  _say "  exit=${rc}  mistake=$(_count "${F_MISTAKE_POOL}")  corr=$(_count "${F_CORR_POOL}")  (log: ${LOG_DIR}/stage_B_takeexam.log)"
fi

# Stage C ─────────────────────────────────────────────────────────────
_head "Stage C: generate_fill_correct"
if _skipped C; then
  _say "  SKIPPED"
  echo "[stageC] SKIPPED" > "${LOG_DIR}/stage_C_fill.log"
elif [ ! -f "${F_MISTAKE_POOL}" ]; then
  _say "  SKIP: ${F_MISTAKE_POOL} not found"
  echo "[stageC] SKIP: mistake pool missing" > "${LOG_DIR}/stage_C_fill.log"
else
  _say "  running ..."
  python -u scripts/train/a_token_sdcl.py --output_path "${F_FILL}" \
    > "${LOG_DIR}/stage_C_fill.log" 2>&1
  rc=$?
  _say "  exit=${rc}  fill_correct=$(_count "${F_FILL}")  (log: ${LOG_DIR}/stage_C_fill.log)"
fi

# Stage D ─────────────────────────────────────────────────────────────
_head "Stage D: build grpo_pool (rolling-8)"
if _skipped D; then
  _say "  SKIPPED"
  echo "[stageD] SKIPPED" > "${LOG_DIR}/stage_D_grpo.log"
elif [ ! -f "${F_MISTAKE_POOL}" ]; then
  _say "  SKIP: ${F_MISTAKE_POOL} not found"
  echo "[stageD] SKIP: mistake pool missing" > "${LOG_DIR}/stage_D_grpo.log"
else
  _say "  running ... (~30min on 4 H800)"
  GRPO_POOL_PATH="${F_GRPO_POOL}" python -u <<'PY' > "${LOG_DIR}/stage_D_grpo.log" 2>&1
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
  rc=$?
  _say "  exit=${rc}  grpo_pool=$(_count "${F_GRPO_POOL}")  (log: ${LOG_DIR}/stage_D_grpo.log)"
fi

# Stage E ─────────────────────────────────────────────────────────────
_head "Stage E: merge"
python -u scripts/train/a_token_sdcl.py merge \
  --corr_answer_path "${F_CORR_POOL}" \
  --fill_correct_path "${F_FILL}" \
  --grpo_path "${F_GRPO_POOL}" \
  --output_path "${F_MERGED}" \
  > "${LOG_DIR}/stage_E_merge.log" 2>&1
rc=$?
_say "  exit=${rc}  merged=$(_count "${F_MERGED}")  (log: ${LOG_DIR}/stage_E_merge.log)"

# Stage F ─────────────────────────────────────────────────────────────
_head "Stage F: audit"
python -u scripts/audit_grpo_pipeline.py --stage grpo_pool \
  --grpo_pool_path "${F_GRPO_POOL}" --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" > "${LOG_DIR}/stage_F1_audit_grpo_console.log" 2>&1
_say "  grpo_pool audit done  (log: ${LOG_DIR}/stage_F1_audit_grpo_console.log)"

python -u scripts/audit_grpo_pipeline.py --stage merged \
  --merged_path "${F_MERGED}" --model_path "${MODEL_PATH}" \
  --out_dir "${LOG_DIR}" > "${LOG_DIR}/stage_F2_audit_merged_console.log" 2>&1
_say "  merged    audit done  (log: ${LOG_DIR}/stage_F2_audit_merged_console.log)"

# Summary ─────────────────────────────────────────────────────────────
{
  echo "logs in:    ${LOG_DIR}"
  echo "backups in: ${BACKUP_DIR}"
  echo
  echo "Pool sizes:"
  for f in "${F_MISTAKE_POOL}" "${F_CORR_POOL}" "${F_FILL}" "${F_GRPO_POOL}" "${F_MERGED}"; do
    printf "  %-60s %s\n" "$f" "$(_count "$f")"
  done
} > "${LOG_DIR}/SUMMARY.log"

_head "ALL STAGES DONE"
cat "${LOG_DIR}/SUMMARY.log"

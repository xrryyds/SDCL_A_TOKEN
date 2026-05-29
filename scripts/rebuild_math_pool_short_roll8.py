"""V2-short 实验 Phase A：rolling-8 严格筛选 mistake/corr 池（4096+2048 协议）。

两阶段流程（用户钦定，省算力）：

  A1) 单次 greedy take_exam（max_prompt=4096, max_new=2048, sample_n=1）
      → exam.json
      → TeacherCorrecter.teacher_mark_paper_with_save() 拆 mistake_collection_book / corr_answer

  A2) 对 mistake 候选 rolling-8（k=8, T=0.6, top_p=0.95, 同协议）
      → exam_roll_recheck_mistake：任一对救回追加到 corr_answer，全错重写 mistake_collection_book

  A3) 复制中间文件到带后缀的新池名，避免覆盖 V2 原池：
      mistake_collection_book_4096.json → mistake_DS_MATH_pool_short_roll8.json
      corr_answer_4096.json              → corr_DS_MATH_pool_short.json

跑完或异常都调 use_worker() 挂卡保活。

执行：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/rebuild_math_pool_short_roll8.py
"""

import os
import shutil
import sys
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

EXAM_DIR = os.path.join(_PROJECT_ROOT, "datasets", "exam")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# 中间文件（FileIOUtils 硬编码）
INTER_MISTAKE = os.path.join(EXAM_DIR, "mistake_collection_book_4096.json")
INTER_CORR = os.path.join(EXAM_DIR, "corr_answer_4096.json")

# V2 原池（不能覆盖）
V2_POOL_MISTAKE = os.path.join(EXAM_DIR, "mistake_DS_MATH_pool.json")
V2_POOL_CORR = os.path.join(EXAM_DIR, "corr_DS_MATH_pool.json")

# 新池（带后缀，避免覆盖）
NEW_POOL_MISTAKE = os.path.join(EXAM_DIR, "mistake_DS_MATH_pool_short_roll8.json")
NEW_POOL_CORR = os.path.join(EXAM_DIR, "corr_DS_MATH_pool_short.json")

# 协议
MAX_PROMPT_LENGTH = 4096   # = 2048 prompt + 2048 gen vLLM 总窗口
MAX_NEW_TOKENS = 2048
ROLL_K = 8
ROLL_TEMPERATURE = 0.6
ROLL_TOP_P = 0.95


def _backup(path: str) -> str | None:
    if not os.path.exists(path):
        print(f"[backup] skip (not exist): {path}", flush=True)
        return None
    bak = f"{path}.bak.{TS}"
    shutil.copy2(path, bak)
    print(f"[backup] {path} → {bak}", flush=True)
    return bak


def _count_json(path: str) -> int:
    import json
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return len(json.load(f))


def main():
    print("=" * 70, flush=True)
    print(f"[rebuild_math_pool_short_roll8] start ts={TS}", flush=True)
    print(f"[rebuild_math_pool_short_roll8] project_root={_PROJECT_ROOT}", flush=True)
    print(f"[rebuild_math_pool_short_roll8] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[rebuild_math_pool_short_roll8] protocol: max_prompt={MAX_PROMPT_LENGTH} max_new={MAX_NEW_TOKENS} k={ROLL_K} T={ROLL_TEMPERATURE} top_p={ROLL_TOP_P}", flush=True)
    print("=" * 70, flush=True)

    # 备份所有相关文件
    _backup(INTER_MISTAKE)
    _backup(INTER_CORR)
    _backup(V2_POOL_MISTAKE)
    _backup(V2_POOL_CORR)
    _backup(NEW_POOL_MISTAKE)
    _backup(NEW_POOL_CORR)

    # ============================================================
    # Phase A1：单次 greedy take_exam → 拆 mistake 候选 / corr
    # ============================================================
    print("\n" + "=" * 70, flush=True)
    print(f"[A1] {datetime.now().isoformat()}  greedy take_exam (sample_n=1)", flush=True)
    print(f"[A1] max_prompt_length={MAX_PROMPT_LENGTH} max_new_tokens={MAX_NEW_TOKENS}", flush=True)
    print("=" * 70, flush=True)

    from main import student_take_exam_Math_sub

    student_take_exam_Math_sub(
        train=True,
        subset="all",
        lora_path=None,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        sample_n=1,
        temperature=0.0,
        top_p=1.0,
    )

    print("\n[A1] teacher_mark_paper_with_save → 拆 mistake_collection_book / corr_answer ...", flush=True)
    from scripts import TeacherCorrecter

    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    del teacher

    n_mistake_a1 = _count_json(INTER_MISTAKE)
    n_corr_a1 = _count_json(INTER_CORR)
    print(f"[A1] DONE  mistake候选={n_mistake_a1}  corr={n_corr_a1}  total={n_mistake_a1 + n_corr_a1}", flush=True)

    # ============================================================
    # Phase A2：mistake 候选 rolling-8 严格筛
    # 任一对救回到 corr，8/8 全错留 mistake
    # ============================================================
    print("\n" + "=" * 70, flush=True)
    print(f"[A2] {datetime.now().isoformat()}  rolling-{ROLL_K} recheck on mistake候选", flush=True)
    print(f"[A2] mistake候选={n_mistake_a1}  T={ROLL_TEMPERATURE} top_p={ROLL_TOP_P}", flush=True)
    print("=" * 70, flush=True)

    from main import exam_roll_recheck_mistake

    exam_roll_recheck_mistake(
        use_lora=False,
        lora_path="",
        max_token=MAX_NEW_TOKENS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        k=ROLL_K,
        temperature=ROLL_TEMPERATURE,
        top_p=ROLL_TOP_P,
    )

    n_mistake_a2 = _count_json(INTER_MISTAKE)
    n_corr_a2 = _count_json(INTER_CORR)
    print(
        f"[A2] DONE  mistake(8/8 全错)={n_mistake_a2}  corr(A1+A2 救回)={n_corr_a2}  "
        f"救回数={n_corr_a2 - n_corr_a1}", flush=True,
    )

    # ============================================================
    # Phase A3：落盘到带后缀的新池名
    # ============================================================
    print("\n" + "=" * 70, flush=True)
    print(f"[A3] {datetime.now().isoformat()}  落盘到新池名", flush=True)
    print("=" * 70, flush=True)

    if not (os.path.exists(INTER_MISTAKE) and os.path.exists(INTER_CORR)):
        raise FileNotFoundError(
            f"中间文件缺失: {INTER_MISTAKE} 或 {INTER_CORR}"
        )

    shutil.copy2(INTER_MISTAKE, NEW_POOL_MISTAKE)
    shutil.copy2(INTER_CORR, NEW_POOL_CORR)
    print(f"[A3] {INTER_MISTAKE} → {NEW_POOL_MISTAKE}", flush=True)
    print(f"[A3] {INTER_CORR}    → {NEW_POOL_CORR}", flush=True)

    # ============================================================
    # 画像
    # ============================================================
    n_mistake = _count_json(NEW_POOL_MISTAKE)
    n_corr = _count_json(NEW_POOL_CORR)
    total = n_mistake + n_corr
    acc = (n_corr / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 70, flush=True)
    print(f"[rebuild_math_pool_short_roll8] DONE", flush=True)
    print(f"  protocol: max_prompt={MAX_PROMPT_LENGTH} max_new={MAX_NEW_TOKENS} roll-{ROLL_K}", flush=True)
    print(f"  mistake (硬骨头): {n_mistake} 题  →  {NEW_POOL_MISTAKE}", flush=True)
    print(f"  corr    (单次或救回): {n_corr} 题  →  {NEW_POOL_CORR}", flush=True)
    print(f"  total: {total} 题   acc={acc:.2f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

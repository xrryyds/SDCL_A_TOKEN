"""一次性脚本：用 2048+8192 干净口径重建 mistake / corr 池。

跟 rebuild_math_pool.py 同流程, 但生成参数改成 2048 prompt + 8192 gen.

Step:
  1. 备份旧 mistake_DS_MATH_pool.json / corr_DS_MATH_pool.json
  2. take_exam (MATH train, max_prompt_length=10240=2048+8192, max_new_tokens=8192)
     → datasets/exam/exam.json
  3. TeacherCorrecter(max_new=8192).teacher_mark_paper_with_save() →
     datasets/exam/mistake_collection_book_8192.json
     datasets/exam/corr_answer_8192.json
  4. 复制为 mistake_DS_MATH_pool.json / corr_DS_MATH_pool.json (覆盖)
  5. finally use_worker() 保活

期望: 加倍 gen 后, 一些原本因截断而 'mistake' 的题转入 corr,
      corr 池规模 ↑, mistake 池规模 ↓.

用法 (4 卡 H800):
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/rebuild_math_pool_8k.py
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

POOL_MISTAKE = os.path.join(EXAM_DIR, "mistake_DS_MATH_pool.json")
POOL_CORR = os.path.join(EXAM_DIR, "corr_DS_MATH_pool.json")

# 新口径 (max_new=8192) 下 teacher_mark_paper_with_save 写入的中间文件
INTER_MISTAKE = os.path.join(EXAM_DIR, "mistake_collection_book_8192.json")
INTER_CORR = os.path.join(EXAM_DIR, "corr_answer_8192.json")


def _backup(path: str):
    if not os.path.exists(path):
        print(f"[backup] skip (not exist): {path}", flush=True)
        return None
    bak = f"{path}.bak.{TS}"
    shutil.copy2(path, bak)
    print(f"[backup] {path} → {bak}", flush=True)
    return bak


def main():
    print("=" * 70, flush=True)
    print(f"[rebuild_math_pool_8k] start ts={TS}", flush=True)
    print(f"[rebuild_math_pool_8k] project_root={_PROJECT_ROOT}", flush=True)
    print(
        f"[rebuild_math_pool_8k] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    print(
        f"[rebuild_math_pool_8k] 口径: max_prompt=2048 + max_new=8192 (vLLM 总窗口 10240)",
        flush=True,
    )
    print("=" * 70, flush=True)

    # 1) 备份旧池
    _backup(POOL_MISTAKE)
    _backup(POOL_CORR)

    # 2) take_exam
    print(
        "\n[step 1/3] take_exam on MATH train (max_prompt=2048+8192=10240, "
        "max_new=8192) ...",
        flush=True,
    )
    from main import student_take_exam_Math_sub

    student_take_exam_Math_sub(
        train=True,
        subset="all",
        lora_path=None,           # Base, 无 LoRA
        max_prompt_length=10240,  # vLLM 总窗口 = 2048 prompt + 8192 gen
        max_new_tokens=8192,
    )

    # 3) teacher 判分 + 拆池 (用 max_new=8192 路径名)
    print(
        "\n[step 2/3] TeacherCorrecter(max_new=8192).teacher_mark_paper_with_save () "
        "拆 mistake/corr ...",
        flush=True,
    )
    from scripts import TeacherCorrecter

    teacher = TeacherCorrecter(max_new=8192)
    teacher.teacher_mark_paper_with_save()
    del teacher

    # 4) 复制成主池名
    print(
        "\n[step 3/3] copy → mistake_DS_MATH_pool.json / corr_DS_MATH_pool.json ...",
        flush=True,
    )
    if not (os.path.exists(INTER_MISTAKE) and os.path.exists(INTER_CORR)):
        raise FileNotFoundError(
            f"中间文件缺失: {INTER_MISTAKE} 或 {INTER_CORR}. teacher 判分可能未成功."
        )
    shutil.copy2(INTER_MISTAKE, POOL_MISTAKE)
    shutil.copy2(INTER_CORR, POOL_CORR)
    print(f"[ok] {INTER_MISTAKE} → {POOL_MISTAKE}", flush=True)
    print(f"[ok] {INTER_CORR} → {POOL_CORR}", flush=True)

    # 简单画像
    import json

    with open(POOL_MISTAKE, "r", encoding="utf-8") as f:
        m = json.load(f)
    with open(POOL_CORR, "r", encoding="utf-8") as f:
        c = json.load(f)
    total = len(m) + len(c)
    acc = (len(c) / total * 100) if total > 0 else 0.0
    print("\n" + "=" * 70, flush=True)
    print(f"[rebuild_math_pool_8k] DONE", flush=True)
    print(f"  mistake: {len(m)} 题  →  {POOL_MISTAKE}", flush=True)
    print(f"  corr:    {len(c)} 题  →  {POOL_CORR}", flush=True)
    print(f"  total:   {total} 题   acc={acc:.2f}%", flush=True)
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

"""V2 4k 4 卡全流程：take_exam → 拆 mistake/corr 池 → fill pipeline → 训练（β=0.0 + β=0.7 双 ckpt）。

实验目的（与 GRPO 三池设计前置实验对齐）：
  在 4 卡机上、V2 4k 协议（max_prompt=2048 prompt budget / max_new=4096 / 总窗口 6144）下走完整一次。
  - 不 roll，单次 greedy take_exam → mistake/corr 二池
  - fill + merge → 训练
  - β=0.0 + β=0.7 双 ckpt 保存

执行（4 卡机 / tmux）：
    cd /workspace/SDCL_A_TOKEN
    git pull
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/run_v2_4k_full_4card.py

阶段（全部串行，前一步失败后续不跑，但 use_worker 一定进入）：
    A) take_exam baseline → teacher_mark_paper → 落 mistake/corr 池（V2 4k 协议）
    B) main.py pipeline --skip_train（fill + merge_to_train_data）
    C1) β=0.0 训练 ep=2 lr=1e-5 max_prompt=2048 max_ans=4096 lora_r=32
    C2) β=0.7 训练 ep=2（同上其余参数）

prompt 模板全流程统一：所有阶段（take_exam / teacher_mark / fill / 训练）共享
  SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."
  + DeepSeek-R1-Distill-Qwen tokenizer 自带 chat_template
（这一致性已在 V2-short 当前节点 memory 里审计过，本次复用同样的入口）。

tmux-friendly：每阶段子进程输出仍直通 tmux，但脚本最末尾会一次性打印
所有产物（ckpt 路径 + 池规模 + train_data 规模 + stage 时序），异常时
也走 except → finally 同一份 _print_final_summary。

loss 日志：
  训练侧 a_token_sdcl_train.py 每 log_interval=10 step 已打印 [Step N] 行，
  含 ce / kl_corr / kl_fill 三列；fill 平均 loss 即 kl_fill，无需新增。
  日志文件位置：<output_dir>/training.log（torchrun 各 rank 各一份）。
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODEL_PATH = "/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B"

# V2 4k 池（覆盖式重建）
EXAM_DIR = os.path.join(_PROJECT_ROOT, "datasets", "exam")
INTER_MISTAKE = os.path.join(EXAM_DIR, "mistake_collection_book_4096.json")
INTER_CORR = os.path.join(EXAM_DIR, "corr_answer_4096.json")
POOL_MISTAKE = os.path.join(EXAM_DIR, "mistake_DS_MATH_pool.json")
POOL_CORR = os.path.join(EXAM_DIR, "corr_DS_MATH_pool.json")

# Pipeline 产物（V2 默认名）
FILL_CORRECT_PATH = "datasets/exam/fill_correct.json"
TRAIN_DATA_PATH = "datasets/exam/a_token_train_data.json"

# V2 4k 协议常量
MAX_PROMPT_LEN_VLLM = 6144   # vLLM 总窗口 = 2048 prompt + 4096 gen
MAX_NEW_TOKENS = 4096
TRAIN_MAX_PROMPT = 2048      # 训练侧 prompt 单独预算
TRAIN_MAX_ANSWER = 4096

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_v2_4k_4card_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_v2_4k_4card_{TS}"
CKPT_B00 = os.path.join(OUT_B00, "checkpoint_epoch_2")
CKPT_B07 = os.path.join(OUT_B07, "checkpoint_epoch_2")

# 全局：阶段时序 + 产物路径
STAGE_LOG: list[dict] = []
RESULT_PATHS: dict[str, str] = {}


def _record(key: str, path: str):
    RESULT_PATHS[key] = path


def _stage_done(stage: str, dt_min: float, status: str = "ok", err: str | None = None):
    STAGE_LOG.append({"stage": stage, "status": status, "duration_min": round(dt_min, 1), "err": err})


def _run(cmd: list[str], stage: str):
    print(f"\n[stage={stage}] START {datetime.now().isoformat()}  cmd={' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    except BaseException as e:
        _stage_done(stage, (time.time() - t0) / 60, status="exc", err=repr(e))
        raise
    dt = (time.time() - t0) / 60
    if proc.returncode != 0:
        _stage_done(stage, dt, status="fail", err=f"rc={proc.returncode}")
        raise RuntimeError(f"[stage={stage}] FAILED rc={proc.returncode} after {dt:.1f} min")
    _stage_done(stage, dt, status="ok")
    print(f"[stage={stage}] DONE duration={dt:.1f} min", flush=True)


def _count_json(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return None


def _backup(path: str):
    if not os.path.exists(path):
        return
    bak = f"{path}.bak.{TS}"
    shutil.copy2(path, bak)
    print(f"[backup] {path} → {bak}", flush=True)


# =====================================================
# Stage A：take_exam + teacher_mark → 落池
# =====================================================
def stage_a_rebuild_pool():
    """直接在主进程里调，不起 subprocess，因为 take_exam 会自己处理多卡 vLLM。"""
    print("\n" + "=" * 70, flush=True)
    print(f"[A] {datetime.now().isoformat()}  rebuild mistake/corr pool @ V2 4k", flush=True)
    print(f"[A] max_prompt_length={MAX_PROMPT_LEN_VLLM} (vLLM total) max_new_tokens={MAX_NEW_TOKENS}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        _backup(INTER_MISTAKE)
        _backup(INTER_CORR)
        _backup(POOL_MISTAKE)
        _backup(POOL_CORR)

        from main import student_take_exam_Math_sub
        student_take_exam_Math_sub(
            train=True,
            subset="all",
            lora_path=None,
            max_prompt_length=MAX_PROMPT_LEN_VLLM,
            max_new_tokens=MAX_NEW_TOKENS,
            sample_n=1,
            temperature=0.0,
            top_p=1.0,
            # device_ids=None → 用 CUDA_VISIBLE_DEVICES 全部
        )

        from scripts import TeacherCorrecter
        teacher = TeacherCorrecter()
        teacher.teacher_mark_paper_with_save()
        del teacher

        if not (os.path.exists(INTER_MISTAKE) and os.path.exists(INTER_CORR)):
            raise FileNotFoundError(
                f"Stage A 中间文件缺失: {INTER_MISTAKE} 或 {INTER_CORR}"
            )

        shutil.copy2(INTER_MISTAKE, POOL_MISTAKE)
        shutil.copy2(INTER_CORR, POOL_CORR)
        _record("pool_mistake", POOL_MISTAKE)
        _record("pool_corr", POOL_CORR)
        n_m = _count_json(POOL_MISTAKE)
        n_c = _count_json(POOL_CORR)
        print(f"[A] 落池 mistake={n_m} corr={n_c} acc={(n_c or 0) * 100 / max((n_m or 0) + (n_c or 0), 1):.2f}%", flush=True)
    except BaseException as e:
        _stage_done("A_rebuild_pool", (time.time() - t0) / 60, status="exc", err=repr(e))
        raise
    _stage_done("A_rebuild_pool", (time.time() - t0) / 60, status="ok")


# =====================================================
# Stage B：fill pipeline + merge_to_train_data
# =====================================================
def stage_b_fill_pipeline():
    _run(
        [
            "python", "main.py", "pipeline",
            "--mistake_path", os.path.relpath(POOL_MISTAKE, _PROJECT_ROOT),
            "--corr_answer_path", os.path.relpath(POOL_CORR, _PROJECT_ROOT),
            "--fill_correct_path", FILL_CORRECT_PATH,
            "--train_data_path", TRAIN_DATA_PATH,
            "--fill_prompt_len", str(TRAIN_MAX_PROMPT),     # 2048 prompt budget
            "--fill_max_gen_token", str(TRAIN_MAX_ANSWER),   # 4096 gen
            "--fill_epoch", "3",
            "--skip-train",
        ],
        stage="B_fill_pipeline",
    )
    _record("fill_correct", os.path.join(_PROJECT_ROOT, FILL_CORRECT_PATH))
    _record("train_data", os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH))


# =====================================================
# Stage C：双 ckpt 训练
# =====================================================
def _train_cmd(beta: str, output_dir: str) -> list[str]:
    return [
        "python",
        "scripts/train/run_a_token_sdcl_train.py",
        "--model_path", MODEL_PATH,
        "--data_path", TRAIN_DATA_PATH,
        "--output_dir", output_dir,
        "--num_epochs", "2",
        "--batch_size", "6",
        "--gradient_accumulation_steps", "3",
        "--learning_rate", "1e-5",
        "--max_prompt_length", str(TRAIN_MAX_PROMPT),
        "--max_answer_length", str(TRAIN_MAX_ANSWER),
        "--beta_fill", beta,
        "--use_lora",
        "--lora_r", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0.0",
        "--gradient_checkpointing",
        "--log_interval", "10",
        "--save_steps", "500",
        "--save_total_limit", "5",
        "--seed", "42",
    ]


def stage_c_train():
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="C1_train_b00")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[C1] 期望 ckpt 不存在: {CKPT_B00}")
    _record("ckpt_b00", CKPT_B00)

    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="C2_train_b07")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[C2] 期望 ckpt 不存在: {CKPT_B07}")
    _record("ckpt_b07", CKPT_B07)


# =====================================================
# 最终 tmux-friendly 汇总
# =====================================================
def _print_final_summary(overall_status: str, top_err: str | None = None):
    lines = []
    lines.append("=" * 78)
    lines.append(f"[run_v2_4k_full_4card] FINAL SUMMARY  ts={TS}  status={overall_status}")
    lines.append("=" * 78)

    # 阶段时序
    lines.append("\n[stages]")
    if not STAGE_LOG:
        lines.append("  (no stage executed)")
    for s in STAGE_LOG:
        tag = s["status"].upper()
        line = f"  [{tag:4}] {s['stage']:24} {s['duration_min']:>6.1f} min"
        if s.get("err"):
            line += f"   err={s['err']}"
        lines.append(line)

    # 数据产物
    lines.append("\n[data artifacts]")
    n_m = _count_json(POOL_MISTAKE)
    n_c = _count_json(POOL_CORR)
    n_fc = _count_json(os.path.join(_PROJECT_ROOT, FILL_CORRECT_PATH))
    n_td = _count_json(os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH))
    for label, path, n in [
        ("pool_mistake", POOL_MISTAKE, n_m),
        ("pool_corr", POOL_CORR, n_c),
        ("fill_correct", os.path.join(_PROJECT_ROOT, FILL_CORRECT_PATH), n_fc),
        ("train_data", os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH), n_td),
    ]:
        ok = "OK" if os.path.exists(path) else "MISSING"
        n_str = f"n={n}" if n is not None else "n=?"
        lines.append(f"  [{ok:7}] {label:14} {n_str:10} {path}")
    if n_m and n_c:
        acc_pool = n_c * 100 / (n_m + n_c)
        lines.append(f"           pool acc = {acc_pool:.2f}%  (mistake/total)")

    # ckpt 产物
    lines.append("\n[checkpoints]")
    for key, p in [("ckpt_b00", CKPT_B00), ("ckpt_b07", CKPT_B07)]:
        ok = "OK" if os.path.exists(p) else "MISSING"
        lines.append(f"  [{ok:7}] {key:10} {p}")
        parent = os.path.dirname(p)
        if os.path.isdir(parent):
            sibs = sorted(d for d in os.listdir(parent) if d.startswith("checkpoint_"))
            for sib in sibs:
                lines.append(f"  {'':9} {'  ↳':10} {os.path.join(parent, sib)}")
            for fname in ("training.log", "training_args.json"):
                p_log = os.path.join(parent, fname)
                if os.path.exists(p_log):
                    lines.append(f"  {'':9} {'  ↳ ' + fname:10} {p_log}")

    # 训练日志位置（提示用户去看 kl_corr / kl_fill）
    lines.append("\n[training log hints]")
    lines.append("  每 log_interval=10 step 一行，含: ce / kl_corr / kl_fill")
    lines.append("  fill 平均 loss = kl_fill (a_token_sdcl_train.py:1024 输出)")
    lines.append(f"  β=0.0 训练日志: {OUT_B00}/training.log")
    lines.append(f"  β=0.7 训练日志: {OUT_B07}/training.log")

    if top_err:
        lines.append("\n[top-level exception]")
        lines.append(top_err)

    lines.append("=" * 78)
    print("\n" + "\n".join(lines), flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"[run_v2_4k_full_4card] START ts={TS}", flush=True)
    print(f"[run_v2_4k_full_4card] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_v2_4k_full_4card] OUT_B00 = {OUT_B00}", flush=True)
    print(f"[run_v2_4k_full_4card] OUT_B07 = {OUT_B07}", flush=True)
    print(f"[run_v2_4k_full_4card] 协议: vLLM 总窗口=6144 prompt budget=2048 max_new=4096", flush=True)
    print("=" * 70, flush=True)

    stage_a_rebuild_pool()
    stage_b_fill_pipeline()
    stage_c_train()

    print(f"\n[run_v2_4k_full_4card] ALL STAGES OK", flush=True)


if __name__ == "__main__":
    overall = "ok"
    top_err = None
    try:
        main()
    except BaseException as e:
        overall = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    finally:
        try:
            _print_final_summary(overall_status=overall, top_err=top_err)
        except BaseException:
            traceback.print_exc()
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

"""V2 4k 4 卡 lr=5e-5 训练+评测：复用 V2 4k 已生成的 train_data，β=0.0 + β=0.7 双 ckpt。

实验动机（用户指定）：
  V2 4k 4 卡 baseline (lr=1e-5 ep=2) 复现了 V2 2 卡的 mistake ~20%（+5pp）瓶颈。
  用户决定直接拉 lr 1e-5 → 5e-5（5×）看 mistake 增量能否突破。
  注意：这是激进设置，可能训崩（KL loss 发散 / corr 暴跌），需要评测兜底确认。

前置条件：
  V2 4k 4 卡 Stage A+B 已完成，datasets/exam/a_token_train_data.json 就位。

执行（4 卡机 / tmux）：
    cd /workspace/SDCL_A_TOKEN
    git pull
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/run_v2_4k_lr5e5_train_eval_4card.py

阶段（全部串行）：
    1) β=0.0 训练 ep=2 lr=5e-5
    2) β=0.7 训练 ep=2 lr=5e-5
    3) β=0.0 评测（V2 4k 协议）
    4) β=0.7 评测（V2 4k 协议）

tmux-friendly：所有路径 + 指标缓存到末尾一次性打印；异常路径同样走 _print_final_summary。
异常或跑完都 use_worker 保活。
"""

import glob
import json
import os
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

TRAIN_DATA_PATH = "datasets/exam/a_token_train_data.json"
V2_MISTAKE_POOL = "datasets/exam/mistake_DS_MATH_pool.json"
V2_CORR_POOL = "datasets/exam/corr_DS_MATH_pool.json"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_v2_4k_lr5e5_4card_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_v2_4k_lr5e5_4card_{TS}"
CKPT_B00 = os.path.join(OUT_B00, "checkpoint_epoch_2")
CKPT_B07 = os.path.join(OUT_B07, "checkpoint_epoch_2")

STAGE_LOG: list[dict] = []
RESULT_PATHS: dict[str, str] = {}
EVAL_TS_MARK: dict[str, float] = {}


def _record_path(key: str, path: str):
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
        "--learning_rate", "5e-5",
        "--max_prompt_length", "2048",
        "--max_answer_length", "4096",
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


def _eval_cmd(adapter_path: str | None) -> list[str]:
    cmd = ["python", "main.py", "eval", "--model_path", MODEL_PATH]
    if adapter_path:
        cmd += ["--adapter_path", adapter_path]
    cmd += [
        "--mistake_path", V2_MISTAKE_POOL,
        "--corr_path", V2_CORR_POOL,
        "--max_prompt_length", "6144",
        "--max_new_tokens", "4096",
        "--math500_roll_k", "8",
        "--math500_roll_temperature", "0.6",
        "--math500_roll_top_p", "0.95",
        "--device_ids", "0,1,2,3",
    ]
    return cmd


def _find_latest_eval_dir(t_start: float) -> str | None:
    root = os.path.join(_PROJECT_ROOT, "output")
    cands = [
        p for p in glob.glob(os.path.join(root, "eval_*"))
        if os.path.isdir(p) and os.path.getmtime(p) >= t_start - 2
    ]
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def _read_summary(eval_dir: str) -> dict | None:
    s = os.path.join(eval_dir, "summary.json")
    if not os.path.exists(s):
        return None
    try:
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _format_summary_oneline(summary: dict | None) -> str:
    if not summary:
        return "(summary.json missing or unreadable)"
    parts = []
    for k in ("mistake_acc", "corr_acc", "math500_acc",
              "math500_roll_avg_pass1", "math500_roll_pass_at_k",
              "mistake_total", "corr_total"):
        if k in summary:
            v = summary[k]
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return " | ".join(parts) if parts else f"keys={list(summary.keys())}"


def _print_final_summary(overall_status: str, top_err: str | None = None):
    lines = []
    lines.append("=" * 78)
    lines.append(f"[run_v2_4k_lr5e5_train_eval_4card] FINAL SUMMARY  ts={TS}  status={overall_status}")
    lines.append("=" * 78)

    lines.append("\n[stages]")
    if not STAGE_LOG:
        lines.append("  (no stage executed)")
    else:
        for s in STAGE_LOG:
            tag = s["status"].upper()
            line = f"  [{tag:4}] {s['stage']:36} {s['duration_min']:>6.1f} min"
            if s.get("err"):
                line += f"   err={s['err']}"
            lines.append(line)

    lines.append("\n[checkpoints]")
    for key, p in [("ckpt_b00", CKPT_B00), ("ckpt_b07", CKPT_B07)]:
        ok = "OK" if os.path.exists(p) else "MISSING"
        lines.append(f"  [{ok:7}] {key:14} {p}")
        parent = os.path.dirname(p)
        if os.path.isdir(parent):
            sibs = sorted(d for d in os.listdir(parent) if d.startswith("checkpoint_"))
            for sib in sibs:
                lines.append(f"  {'':9} {'  ↳':14} {os.path.join(parent, sib)}")
            for fname in ("training.log", "training_args.json"):
                p_log = os.path.join(parent, fname)
                if os.path.exists(p_log):
                    lines.append(f"  {'':9} {'  ↳ ' + fname:14} {p_log}")

    lines.append("\n[eval outputs]")
    eval_keys = [k for k in RESULT_PATHS if k.startswith("eval_")]
    if not eval_keys:
        lines.append("  (no eval directory captured)")
    for k in eval_keys:
        d = RESULT_PATHS[k]
        ok = "OK" if os.path.isdir(d) else "MISSING"
        lines.append(f"  [{ok:7}] {k:14} {d}")
        if os.path.isdir(d):
            summary_path = os.path.join(d, "summary.json")
            lines.append(f"  {'':9} {'  ↳ summary':14} {summary_path}")
            lines.append(f"  {'':9} {'  ↳ metrics':14} {_format_summary_oneline(_read_summary(d))}")
            for fn in sorted(os.listdir(d)):
                if fn.startswith("items_") and fn.endswith(".jsonl"):
                    lines.append(f"  {'':9} {'  ↳ items':14} {os.path.join(d, fn)}")

    lines.append("\n[reference baselines (V2 4k 协议)]")
    lines.append("  baseline (V2 2卡):       mistake 0.1383 / math500_roll-8 0.7398 / corr 0.9046")
    lines.append("  V2 β=0.0 lr=1e-5 (2卡):  mistake 0.1891 (+5.08pp) / math500_roll-8 0.7355 / corr 0.8973")
    lines.append("  V2 4k 4卡 lr=1e-5 (本机): pool acc 73.20% (mistake=2009/corr=5487)")

    lines.append("\n[warning]")
    lines.append("  lr=5e-5 是激进设置（V2 baseline 1e-5 的 5×），")
    lines.append("  若 KL loss 发散 / corr 暴跌 / math500 跌幅 >5pp，需回退到 lr=3e-5")

    if top_err:
        lines.append("\n[top-level exception]")
        lines.append(top_err)

    lines.append("=" * 78)
    print("\n" + "\n".join(lines), flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] START ts={TS}", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] TRAIN_DATA = {TRAIN_DATA_PATH} (V2 4k 4卡 Stage A+B 产物)", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] OUT_B00    = {OUT_B00}", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] OUT_B07    = {OUT_B07}", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] lr=5e-5 (V2 baseline 的 5×，激进设置)", flush=True)
    print(f"[run_v2_4k_lr5e5_train_eval_4card] 协议: 训练 max_prompt=2048 max_ans=4096 / 评测 max_prompt=6144 max_new=4096", flush=True)
    print("=" * 70, flush=True)

    train_path_abs = os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH)
    if not os.path.exists(train_path_abs):
        raise FileNotFoundError(
            f"V2 4k train_data 缺失: {train_path_abs}\n"
            f"前置: V2 4k 4 卡 Stage A+B 必须先跑过"
        )
    with open(train_path_abs, "r", encoding="utf-8") as f:
        td = json.load(f)
    print(f"[check] train_data 共 {len(td)} 条", flush=True)

    # Step 1: β=0.0 训练
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="train_b00_lr5e5")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[train_b00] 期望 ckpt 不存在: {CKPT_B00}")
    _record_path("ckpt_b00", CKPT_B00)

    # Step 2: β=0.7 训练
    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="train_b07_lr5e5")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[train_b07] 期望 ckpt 不存在: {CKPT_B07}")
    _record_path("ckpt_b07", CKPT_B07)

    # Step 3: β=0.0 评测
    EVAL_TS_MARK["eval_b00"] = time.time()
    _run(_eval_cmd(adapter_path=CKPT_B00), stage="eval_b00_lr5e5")
    d = _find_latest_eval_dir(EVAL_TS_MARK["eval_b00"])
    if d:
        _record_path("eval_b00", d)

    # Step 4: β=0.7 评测
    EVAL_TS_MARK["eval_b07"] = time.time()
    _run(_eval_cmd(adapter_path=CKPT_B07), stage="eval_b07_lr5e5")
    d = _find_latest_eval_dir(EVAL_TS_MARK["eval_b07"])
    if d:
        _record_path("eval_b07", d)


if __name__ == "__main__":
    overall = "ok"
    top_err = None
    try:
        main()
    except BaseException as e:
        overall = "FAIL"
        top_err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    finally:
        # 兜底补抓 eval_* 目录
        for k, t in EVAL_TS_MARK.items():
            if k not in RESULT_PATHS:
                d = _find_latest_eval_dir(t)
                if d:
                    _record_path(k, d)
        try:
            _print_final_summary(overall_status=overall, top_err=top_err)
        except BaseException:
            traceback.print_exc()
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

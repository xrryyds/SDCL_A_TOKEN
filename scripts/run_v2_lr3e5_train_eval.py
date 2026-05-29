"""V2 lr 消融：训练集/协议/ep 全部沿用 V2 baseline，唯一改动 lr 1e-5 → 3e-5。

实验动机：
  V2 原 β=0.0 ep=2 lr=1e-5 在 mistake 池只 +5.08pp，用户怀疑"救回率少 = 没学到位"，
  最小改动假设：训练量没变、训练集没变、协议没变，仅 lr 拉高 3×，看 mistake Δ 能否
  从 +5pp 推到 +15pp+。

对照（V2 原口径，同池子同协议）：
  baseline:         mistake 13.83 / math500 roll-8 73.98 / corr 90.46
  V2 β=0.0 lr=1e-5: mistake 18.91 (+5.08) / math500 73.55 (-0.43) / corr 89.73 (-0.73)

本次（lr=3e-5，其余全等）：
  训练集 = datasets/exam/a_token_train_data.json （V2 原 6735 条 corr+fill_correct merged）
  ep=2 / bs=6 / gas=3 / lora_r=32 / α=64 / max_prompt=2048 / max_ans=4096
  β=0.0 + β=0.7 双 ckpt
  评测 V2 4k 协议 (max_prompt=6144 max_new=4096)

执行（2 卡机 / tmux）：
    cd /workspace/SDCL_A_TOKEN
    git pull
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/run_v2_lr3e5_train_eval.py

阶段（串行）：
    1) β=0.0 训练
    2) β=0.7 训练
    3) β=0.0 评测
    4) β=0.7 评测

tmux-friendly：所有需要用户看到的控制台输出（路径 + 指标）全部缓存到脚本最末尾一次性打印。
中间阶段子进程的输出仍直通 tmux（用于实时观察），但最终汇总会重打所有关键信息，
所以哪怕只看 tail 也能拿到完整产物清单。异常路径在 except + finally 中同样打印汇总后才挂卡。
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
OUT_B00 = f"/workspace/SDCL_A_TOKEN/output/a_token_b00_lr3e5_{TS}"
OUT_B07 = f"/workspace/SDCL_A_TOKEN/output/a_token_b07_lr3e5_{TS}"
CKPT_B00 = os.path.join(OUT_B00, "checkpoint_epoch_2")
CKPT_B07 = os.path.join(OUT_B07, "checkpoint_epoch_2")

# 全局：阶段时序 + 产物路径，跑完/异常都用 _print_final_summary 一次性 dump
STAGE_LOG: list[dict] = []      # [{stage, status, duration_min, err?}]
RESULT_PATHS: dict[str, str] = {}   # key → path
EVAL_TS_MARK: dict[str, float] = {}  # 评测开始时刻，用来扫 output/eval_* 找新目录


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
        "--learning_rate", "3e-5",
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
        "--device_ids", "0,1",
    ]
    return cmd


def _find_latest_eval_dir(t_start: float) -> str | None:
    """扫 output/eval_* 找 mtime ≥ t_start-2s 的最新目录。"""
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
    # 尽量挑常见字段，缺哪个跳哪个，不抛
    for k in ("mistake_acc", "corr_acc", "math500_acc",
              "math500_roll_avg_pass1", "math500_roll_pass_at_k",
              "mistake_total", "corr_total"):
        if k in summary:
            v = summary[k]
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return " | ".join(parts) if parts else f"keys={list(summary.keys())}"


def _print_final_summary(overall_status: str, top_err: str | None = None):
    """tmux-friendly：所有路径 + summary 一次性 dump，跑完/异常都调。"""
    lines = []
    lines.append("=" * 78)
    lines.append(f"[run_v2_lr3e5_train_eval] FINAL SUMMARY  ts={TS}  status={overall_status}")
    lines.append("=" * 78)

    # 1) 阶段时序
    lines.append("\n[stages]")
    if not STAGE_LOG:
        lines.append("  (no stage executed)")
    else:
        for s in STAGE_LOG:
            tag = s["status"].upper()
            line = f"  [{tag:4}] {s['stage']:32} {s['duration_min']:>6.1f} min"
            if s.get("err"):
                line += f"   err={s['err']}"
            lines.append(line)

    # 2) ckpt 路径
    lines.append("\n[checkpoints]")
    for key, p in [("ckpt_b00", CKPT_B00), ("ckpt_b07", CKPT_B07)]:
        ok = "OK" if os.path.exists(p) else "MISSING"
        lines.append(f"  [{ok:7}] {key:14} {p}")
        # 同时列 training output_dir 下的所有 ckpt（含 epoch_1）
        parent = os.path.dirname(p)
        if os.path.isdir(parent):
            sibs = sorted(d for d in os.listdir(parent) if d.startswith("checkpoint_"))
            for sib in sibs:
                lines.append(f"  {'':9} {'  ↳':14} {os.path.join(parent, sib)}")

    # 3) 评测产物（含 summary.json 关键指标）
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
            # 列出该目录下所有 items_*.jsonl
            for fn in sorted(os.listdir(d)):
                if fn.startswith("items_") and fn.endswith(".jsonl"):
                    lines.append(f"  {'':9} {'  ↳ items':14} {os.path.join(d, fn)}")

    # 4) 对照基线（方便直接和本次比）
    lines.append("\n[reference baselines (V2 同协议)]")
    lines.append("  baseline:           mistake 0.1383 / math500_roll-8 0.7398 / corr 0.9046")
    lines.append("  V2 β=0.0 lr=1e-5:   mistake 0.1891 (+5.08pp) / math500_roll-8 0.7355 / corr 0.8973")

    if top_err:
        lines.append("\n[top-level exception]")
        lines.append(top_err)

    lines.append("=" * 78)
    out = "\n".join(lines)
    print("\n" + out, flush=True)


def main():
    print("=" * 70, flush=True)
    print(f"[run_v2_lr3e5_train_eval] START ts={TS}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] TRAIN_DATA = {TRAIN_DATA_PATH} (V2 原 6735 条)", flush=True)
    print(f"[run_v2_lr3e5_train_eval] OUT_B00    = {OUT_B00}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] OUT_B07    = {OUT_B07}", flush=True)
    print(f"[run_v2_lr3e5_train_eval] 唯一改动 vs V2 baseline: lr 1e-5 → 3e-5", flush=True)
    print(f"[run_v2_lr3e5_train_eval] 评测协议: V2 4k (max_prompt=6144 max_new=4096)", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(os.path.join(_PROJECT_ROOT, TRAIN_DATA_PATH)):
        raise FileNotFoundError(f"V2 原训练集缺失: {TRAIN_DATA_PATH}")

    # Step 1: β=0.0 训练
    _run(_train_cmd(beta="0.0", output_dir=OUT_B00), stage="train_b00_lr3e5")
    if not os.path.exists(CKPT_B00):
        raise FileNotFoundError(f"[train_b00] 期望 ckpt 不存在: {CKPT_B00}")
    _record_path("ckpt_b00", CKPT_B00)

    # Step 2: β=0.7 训练
    _run(_train_cmd(beta="0.7", output_dir=OUT_B07), stage="train_b07_lr3e5")
    if not os.path.exists(CKPT_B07):
        raise FileNotFoundError(f"[train_b07] 期望 ckpt 不存在: {CKPT_B07}")
    _record_path("ckpt_b07", CKPT_B07)

    # Step 3: β=0.0 评测
    EVAL_TS_MARK["eval_b00"] = time.time()
    _run(_eval_cmd(adapter_path=CKPT_B00), stage="eval_b00_lr3e5")
    d = _find_latest_eval_dir(EVAL_TS_MARK["eval_b00"])
    if d:
        _record_path("eval_b00", d)

    # Step 4: β=0.7 评测
    EVAL_TS_MARK["eval_b07"] = time.time()
    _run(_eval_cmd(adapter_path=CKPT_B07), stage="eval_b07_lr3e5")
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
        # 不 re-raise：先打印汇总再 use_worker 保活
    finally:
        # 兜底：异常途中也尝试补抓 eval_* 目录
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

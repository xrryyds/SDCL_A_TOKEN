"""GRPO rollout engine 独立验证 harness（先打通再集成）。

跑这个 harness 应该全绿,trainer 集成（Stage 4）才安全:
  1) vLLM colocate + LoRA 加载
  2) hot-swap 不漏显存
  3) 每次 swap < 10s
  4) 与 student 7B bf16 colocate 不冲突
  5) rollout 输出格式正确

执行:
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0          # 单卡测试
    python scripts/train/test_grpo_rollout_engine.py --adapter_path <现有 V2 4k LoRA>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


SAMPLE_QUESTION = (
    "What is the value of $\\frac{1}{2} + \\frac{1}{3}$? "
    "Please reason step by step and put your final answer within \\boxed{}."
)
SYSTEM_PROMPT = "Please reason step by step and put your final answer within \\boxed{}."


def _build_prompt(tokenizer, question: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _save_dummy_lora(model_path: str, out_dir: str, lora_r: int = 32):
    """造一个 0 增量的 LoRA adapter,用于验证 hot-swap 在不同 adapter_dir 之间能切换。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print(f"[dummy_lora] building 0-init LoRA at {out_dir} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        bias="none",
    )
    peft = get_peft_model(base, cfg)
    peft.save_pretrained(out_dir)
    del peft, base
    torch.cuda.empty_cache()
    print(f"[dummy_lora] saved.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--adapter_path", default="", help="可选:现有 V2 4k LoRA 目录,作为 step=0 LoRA")
    ap.add_argument("--max_model_len", type=int, default=6144)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.22)
    ap.add_argument("--n_swaps", type=int, default=10, help="hot-swap 循环次数")
    ap.add_argument("--swap_threshold_sec", type=float, default=15.0)
    ap.add_argument("--colocate_alloc", action="store_true",
                    help="同时 alloc 一个 7B bf16 tensor 验证 colocate 不冲突")
    args = ap.parse_args()

    print("=" * 70, flush=True)
    print(f"[test_grpo_rollout_engine] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[test_grpo_rollout_engine] model_path={args.model_path}", flush=True)
    print(f"[test_grpo_rollout_engine] adapter_path={args.adapter_path or '(dummy 0-init)'}", flush=True)
    print(f"[test_grpo_rollout_engine] max_model_len={args.max_model_len} gpu_mem_util={args.gpu_memory_utilization}", flush=True)
    print("=" * 70, flush=True)

    from scripts.train.grpo_rollout_engine import GrpoRolloutEngine
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    sample_prompt = _build_prompt(tok, SAMPLE_QUESTION)

    # 准备 step=0 的 LoRA：用户给路径就用,否则造个 dummy
    tmp_root = os.path.join(_PROJECT_ROOT, "output", "_grpo_engine_test_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    if args.adapter_path and os.path.exists(args.adapter_path):
        step0_dir = args.adapter_path
    else:
        step0_dir = os.path.join(tmp_root, "lora_step0")
        if not os.path.exists(os.path.join(step0_dir, "adapter_config.json")):
            _save_dummy_lora(args.model_path, step0_dir)

    # ============================================================
    # 1) 引擎启动 + 同卡 colocate 验证
    # ============================================================
    print("\n[1] 启动 GrpoRolloutEngine on cuda:0 ...", flush=True)
    engine = GrpoRolloutEngine(
        model_path=args.model_path,
        device_id=0,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("[1] OK 引擎已启动", flush=True)

    if args.colocate_alloc:
        import torch
        print("\n[1.b] colocate 验证:在 cuda:0 alloc 7B bf16 tensor ...", flush=True)
        n_params = 7 * 1024 * 1024 * 1024
        try:
            ghost = torch.zeros(n_params, dtype=torch.bfloat16, device="cuda:0")
            print(f"[1.b] OK ghost tensor shape={ghost.shape} dtype={ghost.dtype}", flush=True)
            del ghost
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[1.b] FAIL colocate alloc: {e}", flush=True)
            traceback.print_exc()

    # ============================================================
    # 2) update_lora step=0 + rollout n=8
    # ============================================================
    print(f"\n[2] update_lora step=0 from {step0_dir} ...", flush=True)
    t0 = time.time()
    engine.update_lora(step0_dir, step=0)
    print(f"[2] update_lora ok, dt={time.time()-t0:.2f}s", flush=True)

    print("[2] rollout n=8 ...", flush=True)
    t0 = time.time()
    rollouts = engine.rollout([sample_prompt], n=8, T=0.6, top_p=0.95, max_tokens=512)
    print(f"[2] rollout ok, dt={time.time()-t0:.2f}s, n_results={len(rollouts[0])}", flush=True)

    assert len(rollouts) == 1
    assert len(rollouts[0]) == 8, f"期望 8 条 rollout,实得 {len(rollouts[0])}"
    distinct = len(set(rollouts[0]))
    print(f"[2] distinct rollouts = {distinct}/8", flush=True)
    # 至少不应该全部一致
    assert distinct >= 2, "rollout 全部相同,T=0.6 应有差异"

    # 至少一条能解出 boxed
    from utils.data_utils import extract_boxed_content
    n_boxed = sum(1 for t in rollouts[0] if extract_boxed_content(t))
    print(f"[2] rollouts with \\boxed{{}} = {n_boxed}/8", flush=True)

    # ============================================================
    # 3) 循环 hot-swap
    # ============================================================
    print(f"\n[3] hot-swap × {args.n_swaps} ...", flush=True)
    swap_dts = []
    for step in range(1, args.n_swaps + 1):
        # 每次都 reuse step0_dir(实战中是不同 adapter,但目录加载耗时类似)
        t0 = time.time()
        engine.update_lora(step0_dir, step=step)
        dt = time.time() - t0
        swap_dts.append(dt)
        # 每两次做一次小 rollout 确认引擎仍活
        if step % 2 == 0:
            _ = engine.rollout([sample_prompt], n=2, T=0.6, top_p=0.95, max_tokens=64)
        print(f"[3] step={step:3d} swap_dt={dt:.2f}s", flush=True)

    if swap_dts:
        avg = sum(swap_dts) / len(swap_dts)
        worst = max(swap_dts)
        print(f"[3] avg swap = {avg:.2f}s  worst = {worst:.2f}s  threshold = {args.swap_threshold_sec}s", flush=True)
        if worst > args.swap_threshold_sec:
            print(f"[3] WARN worst swap > threshold", flush=True)

    # ============================================================
    # 4) 完成
    # ============================================================
    print("\n[4] shutdown ...", flush=True)
    engine.shutdown()
    print("[4] DONE 全部检查通过", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise

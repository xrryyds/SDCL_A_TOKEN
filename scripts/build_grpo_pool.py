"""GRPO 三池：在现有 mistake_collection_book 上跑 rolling-K，写出 grpo_pool.json。

逻辑全在 main.exam_roll_recheck_mistake 里（已加 grpo_pool_path kwarg），
这里只做 CLI 胶水：固化 V2 4k 协议默认值 + 打印 counts + use_worker 保活。

前置：
  exam_paper（FileIOUtils）已绑定 V2 4k 文件名：
    datasets/exam/mistake_collection_book_4096.json
    datasets/exam/corr_answer_4096.json
  二者由前一阶段（V2 4k take_exam + teacher_mark）已就位。

执行（4 卡机）：
    cd /workspace/SDCL_A_TOKEN
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python scripts/build_grpo_pool.py
"""

import argparse
import os
import sys
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_GRPO_POOL = os.path.join(_PROJECT_ROOT, "datasets", "exam", "grpo_DS_MATH_pool.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_path", default="", help="可选 LoRA adapter 路径")
    parser.add_argument("--max_token", type=int, default=4096, help="V2 4k 协议默认 4096")
    parser.add_argument("--max_prompt_length", type=int, default=6144, help="V2 4k 协议 vLLM 总窗口")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--grpo_pool_path", default=DEFAULT_GRPO_POOL)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 70, flush=True)
    print(f"[build_grpo_pool] ts={ts}", flush=True)
    print(f"[build_grpo_pool] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"[build_grpo_pool] grpo_pool_path={args.grpo_pool_path}", flush=True)
    print(f"[build_grpo_pool] k={args.k} T={args.temperature} top_p={args.top_p}", flush=True)
    print(f"[build_grpo_pool] max_prompt_length={args.max_prompt_length} max_token={args.max_token}", flush=True)
    print("=" * 70, flush=True)

    from main import exam_roll_recheck_mistake

    exam_roll_recheck_mistake(
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        max_token=args.max_token,
        max_prompt_length=args.max_prompt_length,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        grpo_pool_path=args.grpo_pool_path,
    )


def _print_grpo_summary(grpo_pool_path: str):
    """把 grpo_pool 产出汇总打印到终端最后,正常 / 异常退出都跑一次。"""
    print("\n" + "=" * 70, flush=True)
    print("[build_grpo_pool] FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"[build_grpo_pool] grpo_pool_path = {grpo_pool_path}", flush=True)
    if not os.path.exists(grpo_pool_path):
        print(f"[build_grpo_pool] WARN  grpo_pool 未生成 (文件不存在)", flush=True)
        return
    try:
        import json
        with open(grpo_pool_path, "r", encoding="utf-8") as f:
            grpo = json.load(f)
        print(f"[build_grpo_pool] grpo_pool entries = {len(grpo)}", flush=True)
        if grpo:
            sample = grpo[0]
            print(
                f"[build_grpo_pool] sample[0] keys = {list(sample.keys())}\n"
                f"  question_idx           = {sample.get('question_idx')}\n"
                f"  ref_answer             = {sample.get('ref_answer')!r}\n"
                f"  anchor_answer          = {sample.get('anchor_answer')!r}\n"
                f"  anchor_first_token_id  = {sample.get('anchor_first_token_id')}\n"
                f"  anchor_first_token_text= {sample.get('anchor_first_token_text')!r}\n"
                f"  n_correct_of_k         = {sample.get('n_correct_of_k')}/{sample.get('k')}\n"
                f"  source                 = {sample.get('source')}",
                flush=True,
            )
    except Exception as e:
        print(f"[build_grpo_pool] summary 读取失败: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    grpo_pool_path_for_finally = DEFAULT_GRPO_POOL
    try:
        # 拿 args 里的真实路径(支持 --grpo_pool_path 覆盖默认)
        _ap_peek = argparse.ArgumentParser(add_help=False)
        _ap_peek.add_argument("--grpo_pool_path", default=DEFAULT_GRPO_POOL)
        _peek_args, _ = _ap_peek.parse_known_args()
        grpo_pool_path_for_finally = _peek_args.grpo_pool_path
        main()
    except BaseException:
        traceback.print_exc()
    finally:
        # 1) 产出汇总:正常 / 异常都打印
        try:
            _print_grpo_summary(grpo_pool_path_for_finally)
        except BaseException:
            traceback.print_exc()
        # 2) use_worker 保活:正常 / 异常都调
        try:
            from main import use_worker

            use_worker()
        except BaseException:
            traceback.print_exc()

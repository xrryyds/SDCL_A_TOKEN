"""GRPO 三池：vLLM colocate + LoRA hot-reload 引擎。

每个 DDP rank 一个实例，与 student 同卡（cuda:LOCAL_RANK）。
trainer 每 K step 调 `update_lora(adapter_dir, step)` 把当前 student LoRA 注入 vLLM。

显存预算（H100/H800 141G 单卡）：
  trainer student 7B+LoRA+grad+optim+act ~50G
  teacher frozen bf16                    ~15G
  vLLM colocate 7B bf16 + 8 rollout × 6k KV ~30G
  余量                                    ~10G

  → vLLM gpu_memory_utilization 默认 0.22 (≈ 30G of 141G)。

API:
  engine = GrpoRolloutEngine(model_path, device_id=local_rank)
  engine.update_lora(adapter_dir, step)        # rank-0 已 save_pretrained，所有 rank 调
  rollouts = engine.rollout(prompts, n=8, T=0.6, top_p=0.95, max_tokens=4096)
  engine.shutdown()                            # 结束训练时
"""

from __future__ import annotations

import collections
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class GrpoRolloutEngine:
    """vLLM colocate engine with LoRA hot-reload.

    - 用 enforce_eager=True 避免与 student 的 activation checkpointing 抢 CUDA graph
    - max_loras=2 + 队列式 GC：始终保留当前 + 上一次，避免引擎在 swap 那一帧抓不到 LoRA
    - update_lora 每次 step+1 单调递增 lora_int_id，避免 vLLM 内部 cache 命中旧权重
    """

    def __init__(
        self,
        model_path: str,
        device_id: int,
        max_model_len: int = 6144,
        max_lora_rank: int = 64,
        gpu_memory_utilization: float = 0.22,
        max_loras: int = 2,
        dtype: str = "bfloat16",
        seed: int = 42,
    ):
        # colocate 关键：把 vLLM 限定在与 student 同一张卡上
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        from vllm import LLM

        logger.info(
            "[GrpoRolloutEngine] init device=%s max_model_len=%d max_lora_rank=%d "
            "gpu_mem_util=%.3f max_loras=%d",
            device_id,
            max_model_len,
            max_lora_rank,
            gpu_memory_utilization,
            max_loras,
        )
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            dtype=dtype,
            seed=seed,
            disable_log_stats=True,
        )
        self.current_lora = None  # type: ignore[assignment]
        self._gc_history: "collections.deque" = collections.deque(maxlen=max_loras)

        # 推理停止 token 与 take_exam 保持一致
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        # 151645: <|im_end|>  151643: <|endoftext|>（DeepSeek-R1-Distill-Qwen tokenizer）
        eos = tok.eos_token_id
        self._stop_token_ids = [t for t in {eos, 151643, 151645} if t is not None]

    def update_lora(self, adapter_dir: str, step: int):
        """Rank 已经把最新 student LoRA save_pretrained 到 adapter_dir 下,
        引擎从该目录加载并切换为当前 LoRARequest。"""
        from vllm.lora.request import LoRARequest

        if not os.path.exists(adapter_dir):
            raise FileNotFoundError(f"adapter_dir 不存在: {adapter_dir}")

        # vLLM 不允许 lora_int_id=0,加 1 偏移
        lora_int_id = int(step) + 1
        new_req = LoRARequest(
            lora_name=f"trainee_{step}",
            lora_int_id=lora_int_id,
            lora_path=adapter_dir,
        )

        # 显式预热,vLLM 会在第一次 generate 时 lazy-load,我们提前推入
        try:
            self.llm.llm_engine.add_lora(new_req)
        except Exception as e:
            # 部分 vLLM 版本不暴露 add_lora;此时直接交给 generate 时 lazy-load
            logger.warning("add_lora not available or failed (will lazy-load): %s", e)

        # GC 旧 LoRA:队列满了就把最老的从引擎里删掉
        if len(self._gc_history) == self._gc_history.maxlen:
            old = self._gc_history.popleft()
            try:
                self.llm.llm_engine.remove_lora(old.lora_int_id)
            except Exception as e:
                logger.warning("remove_lora(%s) failed: %s", old.lora_int_id, e)
        self._gc_history.append(new_req)
        self.current_lora = new_req

    def rollout(
        self,
        prompts: List[str],
        n: int = 8,
        T: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
    ) -> List[List[str]]:
        """对 prompts 列表每条采样 n 条,返回 [B, n] 字符串。

        注意:输入 prompts 已经包含 chat-template,与 trainer 侧 _build_prompt 保持一致。
        """
        from vllm import SamplingParams

        sampling_kwargs = dict(
            n=n,
            temperature=T,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=self._stop_token_ids,
        )
        if seed is not None:
            sampling_kwargs["seed"] = seed
        sp = SamplingParams(**sampling_kwargs)

        outs = self.llm.generate(prompts, sp, lora_request=self.current_lora)
        # outs[i].outputs 是 List[CompletionOutput],长度为 n
        return [[c.text for c in o.outputs] for o in outs]

    def shutdown(self):
        try:
            del self.llm
        except Exception:
            pass
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

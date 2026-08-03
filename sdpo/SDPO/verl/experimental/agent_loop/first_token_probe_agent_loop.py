# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("first_token_probe_agent")
class FirstTokenProbeAgentLoop(AgentLoopBase):
    """Single-turn agent loop that samples only the first response token.

    Used by the SRPO re-rollout path to probe the model's own first-token
    distribution for a prompt. ``max_tokens`` is forced to 1 and temperature to
    1.0 so that repeated calls (via batch-level repeat) reveal which first tokens
    the model can produce. The single sampled token is returned as the response.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        forced_prefix_ids = kwargs.get("forced_prefix_ids")
        if forced_prefix_ids is not None:
            forced_prefix_ids = [int(t) for t in list(forced_prefix_ids)]

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # Skip the fixed format prefix (e.g. "<reasoning>\n") so we probe the first
        # *meaningful* reasoning token rather than the constant format opener.
        probe_prompt_ids = prompt_ids + forced_prefix_ids if forced_prefix_ids else prompt_ids

        # Probe only the first token: force max_tokens=1. Use a higher temperature so
        # non-default (plausible-but-not-preferred) first tokens surface in the samples.
        try:
            probe_temp = float(self.config.actor_rollout_ref.actor.srpo.probe_temperature)
        except Exception:
            probe_temp = 1.5
        probe_params = {**sampling_params, "max_tokens": 1, "temperature": probe_temp}

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=probe_prompt_ids,
                sampling_params=probe_params,
                image_data=images,
                video_data=videos,
            )

        response_ids = list(output.token_ids)[:1]
        response_mask = [1] * len(response_ids)

        response_logprobs = None
        if output.log_probs:
            response_logprobs = list(output.log_probs)[:1]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + 1]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

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


@register("forced_first_token_agent")
class ForcedFirstTokenAgentLoop(AgentLoopBase):
    """Single-turn agent loop whose first response token is forced to a given token id.

    The forced token is appended to the prompt fed to the engine so generation continues
    from it, then re-attributed to the response so that
    ``response_ids == [forced_token] + free_generation``.
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
        forced_first_token_id = int(kwargs["forced_first_token_id"])
        forced_prefix_ids = kwargs.get("forced_prefix_ids")
        if forced_prefix_ids is not None:
            forced_prefix_ids = [int(t) for t in list(forced_prefix_ids)]
        else:
            forced_prefix_ids = []

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # Condition on prompt + fixed format prefix (e.g. "<reasoning>\n") + forced token,
        # so the forced token is the first *meaningful* reasoning token while the response
        # still opens with the required format prefix.
        forced_segment = forced_prefix_ids + [forced_first_token_id]
        forced_prompt_ids = prompt_ids + forced_segment

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=forced_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

        # Re-attribute the forced segment (prefix + forced token) to the response.
        response_ids = (forced_segment + list(output.token_ids))[: self.response_length]
        response_mask = [1] * len(response_ids)

        response_logprobs = None
        if output.log_probs:
            # The forced segment has no logprobs; mirror the first generated one to keep lengths aligned.
            log_probs = list(output.log_probs)
            first_lp = log_probs[0] if log_probs else 0.0
            response_logprobs = ([first_lp] * len(forced_segment) + log_probs)[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

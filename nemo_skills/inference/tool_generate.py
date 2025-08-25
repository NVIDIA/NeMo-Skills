import json
import logging
import re
import sys
import textwrap
from dataclasses import asdict, field
from functools import partial
from typing import Optional

import hydra
from transformers import AutoTokenizer

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ToolGenerationConfig(GenerateSolutionsConfig):
    tokenizer_model: str
    """HuggingFace model ID, or path to tokenizer model folder"""

    inference: InferenceConfig = field(default_factory=InferenceConfig)

    prompt_config: Optional[str] = field(default=None)

    code_execution: bool = field(default=False)

    total_code_executions_in_prompt: int = field(default=10)

    override_max_code_executions: bool = field(default=False)

    tool_errors_in_context: bool = field(default=True)
    """When True, tool execution stderr goes in the context as tool output."""

    def _get_disallowed_params(self):
        return [
            ("code_execution", False),
        ]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="tool_generation_config", node=ToolGenerationConfig)


class ToolGenerationTask(GenerationTask):
    def __init__(self, cfg: ToolGenerationConfig):
        super().__init__(cfg)

        ## FIXME: make configurable.
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_model)
        self.prompt_formatter = partial(
            tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exa_websearch",
                        "description": "Search the web using Exa. Provide relevant links in your answer.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for Exa.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

    def setup_llm(self):
        self.sandbox = get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
        return super().setup_llm()

    def log_example_prompt(self, _):
        return

    ## FIXME: make configurable.
    async def parse_tool_call(self, generation):
        """Parse the last tool call."""

        tool_exp = re.compile(
            "\<\|start\|\>assistant\<\|channel\|\>commentary to=functions\.([\w\_]+).*\<\|message\|\>([\s\S]*)\<\|call\|\>$"
        )

        for match in tool_exp.finditer(generation):
            tool_name, tool_args = match.groups()
            try:
                tool_args = json.loads(tool_args)
            except json.decoder.JSONDecodeError as e:
                if self.cfg.tool_errors_in_context:
                    LOG.error(f"Unable to parse JSON arguments from generation: {tool_name}/{tool_args}")
                    LOG.error(e)

                    return "error", {"error": "Unable to parse JSON arguments"}
                else:
                    raise ValueError(f"Unable to parse JSON arguments from generation: {tool_name}/{tool_args}") from e

            return tool_name, tool_args

        if self.cfg.tool_errors_in_context:
            LOG.error(f"Unable to parse tool call from generation:\n{generation}")

            return "error", {"error": "Unable to parse tool call"}

        raise ValueError(f"Unable to parse tool call from generation:\n{generation}")

    ## FIXME: make configurable.
    async def execute_tool_call(self, tool_name, tool_args):
        if tool_name == "error":
            return tool_args
        elif tool_name == "exa_websearch":
            tool_code = textwrap.dedent(
                f"""
                from exa_py import Exa
                import os

                api_key = os.getenv("EXA_API_KEY")
                if api_key:
                    exa = Exa(api_key)
                    result = exa.answer({repr(tool_args["query"])})
                    print(result.answer)
                else:
                    print("Missing API key.")
            """
            )
        else:
            if self.cfg.tool_errors_in_context:
                LOG.error(f"Tool not available or unsupported: {tool_name}/{tool_args}")

                return {"error": "Tool not available or unsupported"}
            else:
                raise NotImplementedError(f"Tool not available or unsupported: {tool_name}/{tool_args}")

        tool_output, _ = await self.sandbox.execute_code(generated_code=tool_code, language="python")

        if tool_output["process_status"] != "completed":
            if self.cfg.tool_errors_in_context:
                LOG.error(f"Error executing tool {tool_name}/{tool_args}: {tool_output['stderr']}")

                return {"error": tool_output["stderr"]}
            else:
                raise ValueError(f"Error executing tool {tool_name}/{tool_args}:\n{tool_output['stderr']}")

        return {"result": tool_output["stdout"]}

    async def process_tool_output(self, tool_name, tool_args, tool_out):
        _tool_conv = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args},
                    }
                ],
            },
            {"role": "tool", "name": tool_name, "content": tool_out},
        ]
        tool_output_generation = self.prompt_formatter(_tool_conv)

        ## FIXME: hack to get the correct tool output format.
        _start_idx = [m.start() for m in re.finditer("\<\|start\|\>", tool_output_generation)][-2]
        tool_output_generation = tool_output_generation[_start_idx:]

        return tool_output_generation

    async def process_single_datapoint(self, data_point, all_data):
        generation_params = {
            **asdict(self.cfg.inference),
            **self.extra_generate_params,
            "stop_phrases": ["<|call|>", "<|return|>"],
            "include_response": True,
            "remove_stop_phrases": False,
        }

        ## Bookkeeping.
        num_tool_calls = 0
        generation_steps = []
        error_steps = []

        prompt = self.prompt_formatter(self.fill_prompt(data_point, all_data))

        generation = await self.llm.generate_async(prompt=prompt, **generation_params)
        stop_reason = generation["response"].choices[-1].stop_reason
        generation_steps.append(generation["generation"])

        ## FIXME: make configurable.
        while stop_reason in ["<|call|>", 200012] and num_tool_calls < self.cfg.total_code_executions_in_prompt:
            ## FIXME: this is a hacky fix. Unsure why the invariant is not satisfied.
            if not generation_steps[-1].endswith("<|call|>"):
                generation_steps[-1] += "<|call|>"

            tool_name, tool_args = await self.parse_tool_call(generation_steps[-1])
            tool_out = await self.execute_tool_call(tool_name, tool_args)
            tool_output_generation = await self.process_tool_output(tool_name, tool_args, tool_out)
            generation_steps.append(tool_output_generation)

            num_tool_calls += 1

            try:
                generation = await self.llm.generate_async(
                    prompt=prompt + "".join(generation_steps), **generation_params
                )
                stop_reason = generation["response"].choices[-1].stop_reason
            except Exception as e:
                LOG.error(e)
                error_steps.append(str(e))
                break

            generation_steps.append(generation["generation"])

        return {
            ## FIXME: what should the output format be for multi-turn?
            "generation": "".join(generation_steps),
            "num_tool_calls": num_tool_calls,
            "errors": error_steps,
        }


GENERATION_TASK_CLASS = ToolGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="tool_generation_config")
def tool_generation(cfg: ToolGenerationConfig):
    cfg = ToolGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ToolGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    ToolGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        tool_generation()

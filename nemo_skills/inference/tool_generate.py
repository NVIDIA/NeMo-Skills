import json
import logging
import os
import re
import sys
import traceback
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
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)
from nemo_skills.inference.wikipedia import (
    search_resources,
    pretty_outline_from_latex,
    html_to_latex,
    get_page_text,
    truncate_latex_to_id,
)

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
                        "name": "wikipedia_search",
                        "description": "A tool to get a list of Wikipedia page names, IDs, and descriptions for a given query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Query to get the list of relevant page names.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "wikipedia_content",
                        "description": "A tool to get the whole content of the specific Wikipedia page.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "page_id": {
                                    "type": "string",
                                    "description": "Id of the page.",
                                },
                                "section_id": {
                                    "type": "string",
                                    "description": "The index of the section to return. If not specified, will return 5000 characters of the page with no regard to sections.",
                                },
                                "offset": {
                                    "type": "string",
                                    "description": "The index of the character to start with. The tool returns only 5000 characters of the page/section.",
                                },
                            },
                            "required": ["page_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "wikipedia_outline",
                        "description": "A tool to get the outline of the specific Wikipedia page.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "page_id": {
                                    "type": "string",
                                    "description": "Id of the page.",
                                }
                            },
                            "required": ["page_id"],
                        },
                    },
                },
            ],
        )

    def setup_llm(self):
        self.sandbox = (
            get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
        )
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
                    LOG.error(
                        f"Unable to parse JSON arguments from generation: {tool_name}/{tool_args}"
                    )
                    LOG.error(e)

                    return "error", {"error": "Unable to parse JSON arguments"}
                else:
                    raise ValueError(
                        f"Unable to parse JSON arguments from generation: {tool_name}/{tool_args}"
                    ) from e

            return tool_name, tool_args

        if self.cfg.tool_errors_in_context:
            LOG.error(f"Unable to parse tool call from generation:\n{generation}")

            return "error", {"error": "Unable to parse tool call"}

        raise ValueError(f"Unable to parse tool call from generation:\n{generation}")

    ## FIXME: make configurable.
    async def execute_tool_call(self, tool_name, tool_args):
        if tool_name == "error":
            return tool_args
        elif tool_name == "wikipedia_search":
            if "query" not in tool_args:
                LOG.error(
                    f"Error executing tool {tool_name}/{tool_args}: query not defined"
                )
                raise {"error": "query not defined"}
            tool_result = search_resources(tool_args["query"])
        elif tool_name == "wikipedia_outline":
            if "page_id" not in tool_args:
                LOG.error(
                    f"Error executing tool {tool_name}/{tool_args}: page_id not defined"
                )
                raise {"error": "page_id not defined"}

            try:
                offset = tool_args.get("offset", 0)
                offset = int(offset) if offset else 0
            except:
                LOG.error(
                    f"Error executing tool {tool_name}/{tool_args}: offset can't be converted to integer"
                )
                raise {"error": "offset can't be converted to integer"}

            tool_result = pretty_outline_from_latex(
                html_to_latex(get_page_text(tool_args["page_id"])["html"])
            )[offset : offset + 5000]
        elif tool_name == "wikipedia_content":
            if "page_id" not in tool_args:
                LOG.error(
                    f"Error executing tool {tool_name}/{tool_args}: page_id not defined"
                )
                raise {"error": "page_id not defined"}
            try:
                offset = tool_args.get("offset", 0)
                offset = int(offset) if offset else 0
            except:
                LOG.error(
                    f"Error executing tool {tool_name}/{tool_args}: offset can't be converted to integer"
                )
                raise {"error": "offset can't be converted to integer"}
            if "section_id" not in tool_args or not tool_args["section_id"]:
                tool_result = html_to_latex(get_page_text(tool_args["page_id"])["html"])[
                    offset : offset + 5000
                ]
            else:
                try:
                    section_id = int(tool_args["section_id"])
                except:
                    LOG.error(
                        f"Error executing tool {tool_name}/{tool_args}: section_id can't be converted to integer"
                    )
                    raise {"error": "section_id can't be converted to integer"}

                tool_result = truncate_latex_to_id(
                    html_to_latex(get_page_text(tool_args["page_id"])["html"]),
                    section_id,
                )[offset : offset + 5000]
        else:
            if self.cfg.tool_errors_in_context:
                LOG.error(f"Tool not available or unsupported: {tool_name}/{tool_args}")

                return {"error": "Tool not available or unsupported"}
            else:
                raise NotImplementedError(
                    f"Tool not available or unsupported: {tool_name}/{tool_args}"
                )

        return {"result": tool_result}

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
        _start_idx = [
            m.start() for m in re.finditer("\<\|start\|\>", tool_output_generation)
        ][-2]
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
        while (
            stop_reason in ["<|call|>", 200012]
            and num_tool_calls < self.cfg.total_code_executions_in_prompt
        ):
            ## FIXME: this is a hacky fix. Unsure why the invariant is not satisfied.
            if not generation_steps[-1].endswith("<|call|>"):
                generation_steps[-1] += "<|call|>"

            tool_name, tool_args = await self.parse_tool_call(generation_steps[-1])
            try:
                tool_out = await self.execute_tool_call(tool_name, tool_args)
            except Exception as e:
                tool_out = {"error": traceback.format_exc()}
            tool_output_generation = await self.process_tool_output(
                tool_name, tool_args, tool_out
            )
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

from dataclasses import field
from functools import partial
import hydra
import json
import logging
from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    combine_stop_phrases,
)
from nemo_skills.inference.model import server_params
from nemo_skills.code_execution.utils import _extract_between_separators
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)
import sys
from transformers import AutoTokenizer


LOG = logging.getLogger(get_logger_name(__file__.replace("nemo_tir", "nemo_skills")))


@nested_dataclass(kw_only=True)
class ToolGenerationConfig(GenerateSolutionsConfig):
    tokenizer_model: str
    """HuggingFace model ID, or path to tokenizer model folder"""

    ## NOTE: Unused. Only to avoid errors in `setup_prompt`
    prompt_config: str = field(default="gpt-oss/code")

    ## NOTE: Unused. Only to avoid errors in `setup_prompt`.
    prompt_template: str = field(default="gpt-oss-high")

    code_execution: bool = field(default=False)

    total_code_executions_in_prompt: int = field(default=10)

    override_max_code_executions: bool = False

    def _get_disallowed_params(self):
        return [("code_execution", False)]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="tool_generation_config", node=ToolGenerationConfig)


class ToolGenerationTask(GenerationTask):
    def __init__(self, cfg: ToolGenerationConfig):
        super().__init__(cfg)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_model)
        self.prompt_formatter = partial(
            tokenizer.apply_chat_template,
            tokenize=False,
            add_generation_prompt=True,
            builtin_tools=["browser"],
        )

    def setup_llm(self):
        self.sandbox = (
            get_sandbox(**self.cfg.sandbox) if self.cfg.sandbox is not None else None
        )
        return super().setup_llm()

    def log_example_prompt(self, _):
        return

    def setup_prompt(self):
        ## NOTE: should remain unused, only to avoid errors for now.
        return super().setup_prompt()

    ## FIXME: Make more general.
    async def execute_tool_call(self, generation):
        browser_search_query = _extract_between_separators(
            generation,
            (
                "<|start|>assistant<|channel|>analysis to=browser.search code<|message|>",
                "<|call|>",
            ),
        )

        if browser_search_query:
            browser_search_query = json.loads(browser_search_query)["query"]
        else:
            return ""

        browser_search_code = f"""
from exa_py import Exa
import os

exa = Exa(os.getenv("EXA_API_KEY"))

result = exa.answer(
    "{browser_search_query}",
    text=True
)

print(result)
"""

        browser_search_output, _ = await self.sandbox.execute_code(
            generated_code=browser_search_code, language="python"
        )

        return f"<|start|>browser.search to=assistant<|channel|>commentary<|message|>{browser_search_output}<|end|>"

    async def process_single_datapoint(self, data_point, all_data):
        ## Bookkeeping.
        num_tool_calls = 0

        conversation = [{"role": "user", "content": data_point["problem"]}]

        generation_params = {
            "stop_phrases": combine_stop_phrases(
                self.prompt.stop_phrases if self.prompt is not None else None,
                self.extra_stop_phrases,
            ),
            "include_response": True,
            "remove_stop_phrases": False,
            **self.cfg.inference,  ## FIXME: should have been a dataclass, but not?
            **self.extra_generate_params,
        }

        for _ in range(self.cfg.total_code_executions_in_prompt):
            prompt = self.prompt_formatter(conversation)

            ##
            # NOTE: Directly use the model in CodeExecutionWrapper,
            #  as we only need it for sandbox creation.
            #
            generation_step = await self.llm.generate_async(
                prompt=prompt, **generation_params
            )

            conversation.append(
                {"role": "assistant", "content": generation_step["generation"]}
            )

            if generation_step["response"].choices[-1].stop_reason in [
                "<|call|>",
                200012,
            ]:
                tool_call_output = self.execute_tool_call(generation_step["generation"])

                conversation[-1]["content"] += tool_call_output

                num_tool_calls += 1

                continue

            break

        return {
            "generation": "".join([c["content"] for c in conversation[1:]]),
            "num_tool_calls": num_tool_calls,
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

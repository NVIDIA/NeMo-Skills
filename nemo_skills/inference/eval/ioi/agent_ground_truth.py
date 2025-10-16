import asyncio
import logging
import re
import sys
import time
from dataclasses import field

import hydra
import litellm

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_block(text: str):
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# Helper to extract a detailed bug report or solution section from an LLM response
def extract_detailed_solution(solution: str, marker: str = "Detailed Verification", after: bool = True):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    solution = solution.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    report_matches = re.findall(r"<report>(.*?)</report>", solution, re.DOTALL)
    if report_matches:
        # Return the last (most recent) report block, stripped of leading/trailing whitespace.
        return report_matches[-1].strip()
    else:
        raise ValueError(f"No report found in solution: {solution}")


def _extract_boxed_verdict(text: str) -> str:
    """Return the lowercase verdict ('yes' or 'no') found **inside** the latest <report> block.

    If no <report> block is present fall back to searching the whole text. Returns
    an empty string when no boxed verdict is found.
    """

    # Try to focus on the report section first
    try:
        search_area = extract_detailed_solution(text)
    except ValueError:
        # No report block – fall back to full text.
        search_area = text

    # Match one-or-more backslashes before 'boxed' and allow optional spaces
    # around the braces and content to be robust to model formatting.
    m = re.search(r"\\+boxed\s*\{\s*([^}]*)\s*\}", search_area)
    return m.group(1).strip().lower() if m else ""


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/solver"
    self_improve_prompt_config: str = "eval/ioi/agent/self_improve"
    verify_prompt_config: str = "eval/ioi/agent/verify"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_after_verify"
    total_steps: int = 30
    num_self_improve: int = 1
    num_verify: int = 10
    num_majority_verify: int = 5


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOISolutionExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "code_tags": cfg.code_tags,
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "self_improve_solution": get_prompt(cfg.self_improve_prompt_config, **prompt_kwargs),
            "verify_solution": get_prompt(cfg.verify_prompt_config, **prompt_kwargs),
            "improve_after_verify_solution": get_prompt(cfg.improve_after_verify_prompt_config, **prompt_kwargs),
        }
        self.sandbox = LocalSandbox()

    def log_example_prompt(self, data):
        pass

    async def _call_llm(self, data_point, all_data, prompt_key, **extra_data):
        combined_dp = {**data_point, **extra_data}
        filled_prompt = self.fill_prompt(combined_dp, all_data, prompt=self.prompts[prompt_key])
        start_t = time.time()
        try:
            llm_out = await super().process_single_datapoint(combined_dp, all_data, prompt=self.prompts[prompt_key])
        except (litellm.exceptions.OpenAIError, Exception) as e:
            print(f"LLM call failed: {e}\nPrompt causing failure:\n{filled_prompt}")
            raise
        gen_time = time.time() - start_t
        return filled_prompt, llm_out, gen_time

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []

        all_model_solutions = data_point["correct_solutions"]
        latest_generation_response = ""

        for x, latest_generation_response in enumerate(all_model_solutions):
            print(f"Processing models solution {x}/{len(all_model_solutions)}")
            solution = latest_generation_response

            # Launch verifier calls concurrently
            verify_tasks = [
                self._call_llm(
                    data_point,
                    all_data,
                    "verify_solution",
                    solution=solution,
                )
                for _ in range(self.cfg.num_verify)
            ]
            verify_results = await asyncio.gather(*verify_tasks)
            yes_votes = sum(1 for _, ver, _ in verify_results if _extract_boxed_verdict(ver["generation"]) == "yes")
            for _, ver, _ in verify_results:
                print(f"VOTE: {_extract_boxed_verdict(ver['generation'])}")
            print(f"Verification yes votes: {yes_votes}/{self.cfg.num_verify}")
            chat_history.append(
                {
                    "solution": solution,
                    "verdicts": [_extract_boxed_verdict(ver["generation"]) for _, ver, _ in verify_results],
                    "generations": [ver["generation"] for _, ver, _ in verify_results],
                }
            )

        return {
            "generation": latest_generation_response,
            "results": chat_history,
        }


GENERATION_TASK_CLASS = IOISolutionExecutionGenerationTask


@hydra.main(version_base=None, config_name="base_ioi_generation_config")
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Note: IOI Module is being used.")
    LOG.info("Config used: %s", cfg)
    task = IOISolutionExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()

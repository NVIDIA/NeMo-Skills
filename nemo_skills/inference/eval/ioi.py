import asyncio
import logging
import os
import re
import sys
import tempfile
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


async def compile_and_run_cpp(code_string: str, data_point: dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        for original_path, content in data_point.get("grader_files", []):
            filename = os.path.basename(original_path)
            if "checker" in filename or "grader" in filename or not filename.endswith(".h"):
                continue
            with open(os.path.join(temp_dir, filename), "w") as f:
                f.write(content)

        executable_path = os.path.join(temp_dir, "a.out")
        compile_command = ["g++", "-I", temp_dir, "-x", "c++", "-o", executable_path, "-"]
        compiler_process = await asyncio.create_subprocess_exec(
            *compile_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, compile_stderr = await compiler_process.communicate(input=code_string.encode())

        if compiler_process.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{compile_stderr.decode()}\nCode:{code_string}")

        run_process = await asyncio.create_subprocess_exec(
            executable_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        run_stdout, run_stderr = await run_process.communicate()
        return run_stdout.decode(), run_stderr.decode()


def extract_code_block(text: str):
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# Helper to extract a detailed bug report or solution section from an LLM response
def extract_detailed_solution(solution: str, marker: str = "Detailed Verification", after: bool = True):
    # First, handle the new format where the report is enclosed in ```report ``` code fences.
    report_matches = re.findall(r"```report(.*?)```", solution, re.DOTALL)
    if report_matches:
        # Return the last (most recent) report block, stripped of leading/trailing whitespace.
        return report_matches[-1].strip()
    else:
        raise ValueError(f"No report found in solution: {solution}")


def _extract_boxed_verdict(text: str) -> str:
    """Return the lowercase verdict ('yes' or 'no') found **inside** the latest ```report``` block.

    If no ```report``` block is present fall back to searching the whole text. Returns
    an empty string when no boxed verdict is found.
    """

    # Try to focus on the report section first
    try:
        search_area = extract_detailed_solution(text)
    except ValueError:
        # No report block – fall back to full text.
        search_area = text

    m = re.search(r"\\boxed\{([^}]+)\}", search_area, re.IGNORECASE)
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
    num_self_improve: int = 5
    num_verify: int = 10


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
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
        num_steps_completed = 0

        prompt_txt, solution_response, gen_time = await self._call_llm(data_point, all_data, "initial")
        latest_generation_response = solution_response["generation"]
        chat_history.append(
            {"prompt": prompt_txt, "response": latest_generation_response, "generation_time": gen_time}
        )
        print("[Initial] Generated initial solution.")
        try:
            solution = extract_code_block(latest_generation_response)

            for improve_idx in range(self.cfg.num_self_improve):
                print(f"[Self-Improve] Attempt {improve_idx + 1}/{self.cfg.num_self_improve}")
                prompt_txt, improve_response, gen_time = await self._call_llm(
                    data_point,
                    all_data,
                    "self_improve_solution",
                    solution=solution,
                )
                chat_history.append(
                    {"prompt": prompt_txt, "response": improve_response["generation"], "generation_time": gen_time}
                )
                solution = extract_code_block(improve_response["generation"])
                if not solution:
                    raise ValueError(f"Failed to generate an improved solution: {improve_response}")

            for step_num in range(self.cfg.total_steps):
                print(f"[Step {step_num + 1}/{self.cfg.total_steps}] Starting verification phase")
                consecutive_yes = 0
                first_fail_report = None

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
                yes_votes = sum(
                    1 for _, ver, _ in verify_results if _extract_boxed_verdict(ver["generation"]).startswith("y")
                )
                print(f"[Step {step_num + 1}] Verification yes votes: {yes_votes}/{self.cfg.num_verify}")

                for prompt_txt, verify_resp, gen_time in verify_results:
                    chat_history.append(
                        {"prompt": prompt_txt, "response": verify_resp["generation"], "generation_time": gen_time}
                    )
                    ver_out = verify_resp["generation"]

                    # Extract verdict from inside the report block.
                    verdict = _extract_boxed_verdict(ver_out)

                    # Ensure verdict is explicitly 'yes' or 'no'.
                    if verdict not in ("yes", "no"):
                        print(
                            f"[Warning] Invalid verdict extracted (expected 'yes' or 'no', got '{verdict}'). Full output:\n{ver_out}"
                        )

                    if verdict == "yes":
                        consecutive_yes += 1
                        if consecutive_yes >= 5:
                            print(
                                f"[Success] Solution verified correct with {consecutive_yes} consecutive 'yes' votes."
                            )
                            latest_generation_response = solution
                            return {
                                "generation": latest_generation_response,
                                "steps": chat_history,
                                "num_steps_completed": num_steps_completed,
                            }
                    else:  # Treat 'no' or invalid verdict as failure
                        if first_fail_report is None:
                            first_fail_report = ver_out
                        consecutive_yes = 0  # reset streak

                # If we reach here, solution deemed incorrect -> improve using first fail report
                if first_fail_report is not None:
                    verification_log = extract_detailed_solution(first_fail_report, "Detailed Verification", False)
                else:
                    raise ValueError("No fail report found")

                prompt_txt, sol_resp, gen_time = await self._call_llm(
                    data_point,
                    all_data,
                    "improve_after_verify_solution",
                    solution=solution,
                    verification=verification_log,
                )

                new_solution = extract_code_block(sol_resp["generation"])
                if not new_solution:
                    raise ValueError(f"Failed to extract improved solution. Response: {sol_resp}")

                latest_generation_response = sol_resp["generation"]
                solution = new_solution
                chat_history.append(
                    {"prompt": prompt_txt, "response": sol_resp["generation"], "generation_time": gen_time}
                )
                num_steps_completed += 1
        except Exception as e:
            print(f"Agent loop failed: {e}")

        return {
            "generation": latest_generation_response,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
        }


GENERATION_TASK_CLASS = IOIExecutionGenerationTask


@hydra.main(version_base=None, config_name="base_ioi_generation_config")
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Note: IOI Module is being used.")
    LOG.info("Config used: %s", cfg)
    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()

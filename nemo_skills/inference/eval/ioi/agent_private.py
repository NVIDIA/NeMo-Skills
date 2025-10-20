import asyncio
import logging
import os
import re

# Use our own run directory instead of tempfile for better control
# and cleanup handling.
import shutil
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


async def compile_and_run_cpp(code_string: str, data_point: dict, timeout: int = 30):
    """Compile the provided C++ code and run it inside a temporary directory.

    A soft timeout (default 30 s) is applied to the execution phase. On timeout
    the process is killed and a timeout message is returned as stderr.
    """

    # Create a unique directory for this run – it helps with debugging and keeps
    # compilation artefacts isolated. It is explicitly removed in the finally
    # clause so we don't rely on garbage-collection.
    run_dir = f"/tmp/cpp_run_{os.getpid()}_{time.time_ns()}"
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Write supplementary header files supplied in the data point.
        for original_path, content in data_point.get("grader_files", []):
            filename = os.path.basename(original_path)
            if ("checker" in filename) or ("grader" in filename) or (not filename.endswith(".h")):
                continue
            with open(os.path.join(run_dir, filename), "w") as f:
                f.write(content)

        # Compile the solution.
        executable_path = os.path.join(run_dir, "a.out")
        compile_command = ["g++", "-I", run_dir, "-x", "c++", "-o", executable_path, "-"]
        compiler_process = await asyncio.create_subprocess_exec(
            *compile_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, compile_stderr = await compiler_process.communicate(input=code_string.encode())

        if compiler_process.returncode != 0:
            raise RuntimeError(f"C++ compilation failed:\n{compile_stderr.decode()}\nCode:{code_string}")

        # Run the compiled binary with a timeout.
        run_process = await asyncio.create_subprocess_exec(
            executable_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            run_stdout, run_stderr = await asyncio.wait_for(run_process.communicate(), timeout=timeout)
            return run_stdout.decode(), run_stderr.decode()
        except asyncio.TimeoutError:
            # Kill the process group to avoid lingering processes.
            run_process.kill()
            await run_process.wait()
            return "", f"Execution timed out after {timeout} seconds."
    finally:
        # Ensure we never leave temporary artefacts behind.
        shutil.rmtree(run_dir, ignore_errors=True)


def extract_code_block(text: str):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


# Extract a C++ test script wrapped in ```script ... ``` fences
def extract_script_block(text: str):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
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
    testgen_prompt_config: str = "eval/ioi/multiagent/code/generate_test"
    simple_verify_prompt_config: str = "eval/ioi/agent/verify_simple_test"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_with_private_test"
    total_steps: int = 30
    num_self_improve: int = 1
    num_verify: int = 10
    num_majority_verify: int = 5
    # Maximum wall-clock seconds allowed for running compiled C++ code.
    run_timeout_seconds: int = 30


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
            "generate_test_script": get_prompt(cfg.testgen_prompt_config, **prompt_kwargs),
            "simple_verify_solution": get_prompt(cfg.simple_verify_prompt_config, **prompt_kwargs),
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
        cur_generation_response = solution_response["generation"]
        chat_history.append({"prompt": prompt_txt, "response": cur_generation_response, "generation_time": gen_time})

        print("[Initial] Generated initial solution.")

        try:
            for step_num in range(self.cfg.total_steps):
                # Evaluate the current solution using the external evaluator.
                eval_results = await self.evaluator.eval_single({**data_point, "generation": cur_generation_response})
                test_case_results = eval_results["test_case_results"]

                # Check if all subtasks passed fully (score == 1 for every output)
                if all(all(o["score"] == 1 for o in v["outputs"]) for v in test_case_results.values()):
                    print(f"[Success] All test cases passed at step {step_num}.")
                    return {
                        "generation": cur_generation_response,
                        "steps": chat_history,
                        "num_steps_completed": num_steps_completed,
                    }

                print(f"[Step {step_num + 1}/{self.cfg.total_steps}] Improving based on evaluator feedback.")

                # Prepare a concise failure summary (only non-perfect cases)
                failure_lines = []
                for subtask, info in test_case_results.items():
                    for out in info["outputs"]:
                        if out["score"] != 1:
                            failure_lines.append(
                                f"{subtask}:{out['test_name']} score={out['score']} msg={out.get('run_stderr', '').strip()}"
                            )
                failure_summary = "\n".join(failure_lines)

                # Ask the LLM to improve the solution given the evaluator feedback.
                prompt_txt, improve_resp, gen_time = await self._call_llm(
                    data_point,
                    all_data,
                    "improve_after_verify_solution",
                    solution=extract_code_block(cur_generation_response),
                    test_case_results=failure_summary,
                )

                num_steps_completed += 1
                cur_generation_response = improve_resp["generation"]

                chat_history.append(
                    {"prompt": prompt_txt, "response": cur_generation_response, "generation_time": gen_time}
                )
                print(f"Prompt: {prompt_txt}")

            # Reached maximum steps without passing all tests.
            print("[Failure] Reached max improvement steps without passing all tests.")
            return {
                "generation": cur_generation_response,
                "steps": chat_history,
                "num_steps_completed": num_steps_completed,
            }
        except Exception as e:
            print(f"Agent loop failed: {e}")
            return {
                "generation": cur_generation_response,
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

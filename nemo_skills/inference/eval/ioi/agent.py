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
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_after_verify"
    total_steps: int = 30
    num_self_improve: int = 1
    num_verify: int = 10
    num_majority_verify: int = 5


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

                async def run_single_verification():
                    # 1) Generate C++ test script wrapped in ```script fences
                    test_prompt_txt, test_gen_resp, test_gen_time = await self._call_llm(
                        data_point,
                        all_data,
                        "generate_test_script",
                        solution=solution,
                    )
                    test_script_full = test_gen_resp["generation"]
                    test_script_code = extract_script_block(test_script_full)
                    if not test_script_code:
                        raise ValueError(f"Failed to extract test script. Response: {test_gen_resp}")

                    # 2) Execute the script
                    exec_stdout = ""
                    exec_stderr = ""
                    compile_error = ""
                    try:
                        exec_stdout, exec_stderr = await compile_and_run_cpp(test_script_code, data_point)
                    except Exception as e:
                        compile_error = str(e)

                    # Debug prints for execution output
                    print(f"[Execution] stdout:\n{exec_stdout}")
                    print(f"[Execution] stderr:\n{exec_stderr}")
                    if compile_error:
                        print(f"[Execution] compile_error:\n{compile_error}")

                    # 3) Simple verification using outputs
                    verify_prompt_txt, verify_resp, verify_time = await self._call_llm(
                        data_point,
                        all_data,
                        "simple_verify_solution",
                        solution=solution,
                        stdout=exec_stdout,
                        stderr=exec_stderr,
                        compile_error=compile_error,
                    )

                    return {
                        "test_gen": {
                            "prompt": test_prompt_txt,
                            "response": test_gen_resp["generation"],
                            "generation_time": test_gen_time,
                            "script": test_script_code,
                        },
                        "simple_verify": {
                            "prompt": verify_prompt_txt,
                            "response": verify_resp["generation"],
                            "generation_time": verify_time,
                            "stdout": exec_stdout,
                            "stderr": exec_stderr,
                            "compile_error": compile_error,
                        },
                    }

                # Launch verifier attempts concurrently
                verify_results = await asyncio.gather(*[run_single_verification() for _ in range(self.cfg.num_verify)])

                # Tally votes
                yes_votes = 0
                first_fail_log = None
                for res in verify_results:
                    ver_out = res["simple_verify"]["response"]
                    verdict = _extract_boxed_verdict(ver_out)
                    # Debug prints for each generation and vote
                    print(f"GENERATION: {ver_out}")
                    print(f"VOTE: {verdict}")
                    if verdict == "yes":
                        yes_votes += 1
                    else:
                        if first_fail_log is None:
                            # Build a verification log from execution outputs and verifier response
                            first_fail_log = (
                                "stdout:\n"
                                + res["simple_verify"]["stdout"]
                                + "\n\nstderr:\n"
                                + res["simple_verify"]["stderr"]
                                + "\n\ncompile_error:\n"
                                + res["simple_verify"]["compile_error"]
                                + "\n\nverifier_response:\n"
                                + ver_out
                            )

                    # Track prompts/responses in chat history
                    chat_history.append(
                        {
                            "prompt": res["test_gen"]["prompt"],
                            "response": res["test_gen"]["response"],
                            "generation_time": res["test_gen"]["generation_time"],
                        }
                    )
                    chat_history.append(
                        {
                            "prompt": res["simple_verify"]["prompt"],
                            "response": res["simple_verify"]["response"],
                            "generation_time": res["simple_verify"]["generation_time"],
                        }
                    )

                print(f"[Step {step_num + 1}] Verification yes votes: {yes_votes}/{self.cfg.num_verify}")

                # Accept if total yes votes meet or exceed majority threshold
                if yes_votes >= self.cfg.num_majority_verify:
                    print(
                        f"[Success] Solution verified correct with {yes_votes} 'yes' votes (threshold: {self.cfg.num_majority_verify})."
                    )
                    latest_generation_response = solution
                    return {
                        "generation": latest_generation_response,
                        "steps": chat_history,
                        "num_steps_completed": num_steps_completed,
                    }

                # If we reach here, solution deemed incorrect -> improve using first failure execution log
                if first_fail_log is None:
                    raise ValueError("No failure verification log found")

                prompt_txt, sol_resp, gen_time = await self._call_llm(
                    data_point,
                    all_data,
                    "improve_after_verify_solution",
                    solution=solution,
                    verification=first_fail_log,
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

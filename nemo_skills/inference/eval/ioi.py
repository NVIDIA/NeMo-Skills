import asyncio
import logging
import os
import re
import sys
import tempfile
from dataclasses import field

import hydra

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


async def compile_and_run_cpp(code_string: str, data_point: dict):
    with tempfile.TemporaryDirectory() as temp_dir:
        for original_path, content in data_point.get('grader_files', []):
            filename = os.path.basename(original_path)
            if 'checker' in filename or 'grader' in filename or not filename.endswith('.h'):
                continue
            with open(os.path.join(temp_dir, filename), 'w') as f:
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


def extract_test_input(text: str):
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/codegen"
    improve_prompt_config: str = "eval/ioi/codegen_improve"
    test_prompt_config: str = "eval/ioi/codegen_tests"
    llm_select_test: bool = True
    llm_select_test_config: str = "eval/ioi/codegen_select_test"
    language: str = "cpp"
    total_steps: int = 2
    num_test_generations: int = 5


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "prompt_template": cfg.prompt_template,
            "code_tags": cfg.code_tags,
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "improve_solution": get_prompt(cfg.improve_prompt_config, **prompt_kwargs),
            "test_generation": get_prompt(cfg.test_prompt_config, **prompt_kwargs),
            "select_test": get_prompt(cfg.llm_select_test_config, **prompt_kwargs),
        }
        self.sandbox = LocalSandbox()

    def log_example_prompt(self, data):
        pass

    async def _call_llm(self, data_point, all_data, prompt_key, **extra_data):
        return await super().process_single_datapoint(
            {**data_point, **extra_data}, all_data, prompt=self.prompts[prompt_key]
        )

    async def _find_successful_script(self, test_responses, data_point, all_data):
        compiled = []
        first_error = None
        for r in test_responses:
            script = extract_test_input(r['generation'])
            if not script:
                continue
            try:
                stdout, stderr = await compile_and_run_cpp(script, data_point)
                compiled.append((r, script, stdout + stderr))
            except RuntimeError as e:
                if first_error is None:
                    first_error = (str(e), script)

        if not compiled:
            if first_error:
                error_msg, failed_script = first_error
                raise ValueError(
                    f"All test scripts failed. First error:\n{error_msg}\nFailed script:\n{failed_script}"
                )
            raise ValueError(f"Failed to extract a valid test script from {len(test_responses)} attempts.")

        if self.cfg.llm_select_test and len(compiled) > 1:
            parts = []
            for idx, (_, script, output) in enumerate(compiled, 1):
                parts.append(f"## Test script {idx}\n\n{script}\n\n## Output\n\n{output}\n")
            test_scripts_text = "\n\n".join(parts)
            select_resp = await self._call_llm(
                data_point,
                all_data,
                "select_test",
                test_scripts=test_scripts_text,
                question=data_point.get('question', ''),
            )
            m = re.search(r"\\boxed\{(\d+)\}", select_resp['generation'])
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= len(compiled):
                    return compiled[idx - 1][0], compiled[idx - 1][1], len(compiled)

        return compiled[0][0], compiled[0][1], len(compiled)

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []
        num_steps_completed = 0

        solution_response = await self._call_llm(data_point, all_data, "initial")
        latest_generation_response = solution_response['generation']
        chat_history.append(solution_response)

        try:
            solution = extract_code_block(latest_generation_response)
            if not solution:
                raise ValueError(f"Failed to generate an initial solution: {solution_response}")
        except Exception as e:
            LOG.warning("Failed to extract valid code from initial generation: %s", e)
            return {
                'generation': latest_generation_response,
                'steps': chat_history,
                'num_steps_completed': num_steps_completed,
            }

        test_gen_tasks = [
            self._call_llm(data_point, all_data, "test_generation", solution=solution)
            for _ in range(self.cfg.num_test_generations)
        ]
        test_responses = await asyncio.gather(*test_gen_tasks)
        try:
            _, test_script, _ = await self._find_successful_script(test_responses, data_point, all_data)
        except Exception as e:
            LOG.warning("Failed to generate a successful test input initially: %s", e)
            return {
                'generation': latest_generation_response,
                'steps': chat_history,
                'num_steps_completed': num_steps_completed,
            }

        try:
            for _ in range(self.cfg.total_steps):
                stdout, stderr = await compile_and_run_cpp(test_script, data_point)
                output = stdout + stderr

                sol_resp = await self._call_llm(
                    data_point,
                    all_data,
                    "improve_solution",
                    solution=solution,
                    script=test_script,
                    output=output,
                )

                new_solution = extract_code_block(sol_resp['generation'])
                if not new_solution:
                    raise ValueError(f"Failed to extract improved solution. Response: {sol_resp}")

                latest_generation_response = sol_resp['generation']
                solution = new_solution
                chat_history.append(sol_resp)
                num_steps_completed += 1
        except Exception:
            pass

        return {
            'generation': latest_generation_response,
            'steps': chat_history,
            'num_steps_completed': num_steps_completed,
        }


GENERATION_TASK_CLASS = IOIExecutionGenerationTask


@hydra.main(version_base=None, config_name='base_ioi_generation_config')
def ioi_generation(cfg: IOIExecutionConfig):
    cfg = IOIExecutionConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = IOIExecutionGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(IOIExecutionConfig, server_params=server_params())

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_generation()

import asyncio
import json
import logging
import shutil
import textwrap
from dataclasses import field
from pathlib import Path

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

# Define constants for better readability and maintenance
LIVECODEBENCH_PYTHON_GIT_URL = (
    "git+https://github.com/wasiahmad/livecodebench.git@f285640c20aaf18df1ee5917621a596af4630b5e"
)
LIVECODEBENCH_PYPY3_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git"


@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    language: str = "python"  # "cpp" is another option now
    test_file: str = None
    interpreter: str = "python"  # use either "python" or pypy3
    timeout: int = 6


async def install_livecodebench(sandbox, interpreter: str):
    """Installs the livecodebench package once inside the provided sandbox."""
    LOG.info(f"Installing livecodebench using {interpreter}...")
    pip_command = "pip" if interpreter == "python" else "pypy3 -m pip"
    GIT_URL = LIVECODEBENCH_PYTHON_GIT_URL if interpreter == "python" else LIVECODEBENCH_PYPY3_GIT_URL
    install_command = f"{pip_command} install {GIT_URL}"

    result, _ = await sandbox.execute_code(install_command, language="shell", timeout=300)

    if result["process_status"] != "completed":
        LOG.warning(f"Failed to install livecodebench: {result.get('stderr', 'Unknown error')}")
        return False

    LOG.info("Successfully installed livecodebench.")
    return True


async def eval_livecodebench_async(cfg):
    """Asynchronous core logic for evaluating LiveCodeBench."""
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    if eval_config.language == "python" and eval_config.interpreter not in ["python", "pypy3"]:
        raise ValueError("Python interpreter must be 'python' or 'pypy3'.")
    if eval_config.language == "cpp" and eval_config.test_file is None:
        raise ValueError("C++ evaluation requires a test_file.")

    async with get_sandbox(**eval_config.sandbox) as sandbox:
        # Install dependencies only once
        if not await install_livecodebench(sandbox, eval_config.interpreter):
            return  # Stop if installation fails

        release_version = None
        for jsonl_file_path_str in unroll_files(cfg.input_files):
            jsonl_file = Path(jsonl_file_path_str)
            LOG.info(f"Processing file: {jsonl_file.name}")

            with open(jsonl_file, "r", encoding="utf-8") as f_in:
                samples = [preprocess_code(json.loads(line), eval_config.language) for line in f_in]

            for sample in samples:
                sample["code_list"] = [sample["completion"]]
                current_version = sample["release_version"]
                if release_version is None:
                    release_version = current_version
                elif release_version != current_version:
                    raise ValueError(
                        f"All samples should have the same release version. "
                        f"Found {release_version} and {current_version}"
                    )

            # Use a temporary file for the evaluation harness to avoid overwriting source
            temp_eval_file = jsonl_file.with_suffix(".temp.jsonl")
            with open(temp_eval_file, "w", encoding="utf-8") as f_out:
                for sample in samples:
                    f_out.write(json.dumps(sample) + "\n")

            # 2. Run the evaluation harness in the sandbox
            test_file_arg = f'"{eval_config.test_file}"' if eval_config.test_file else "None"
            python_script_to_run = textwrap.dedent(f"""
                from livecodebench.evaluate import evaluate

                evaluate(
                    custom_output_file='{temp_eval_file.name}',
                    release_version='release_{release_version}',
                    test_file={test_file_arg},
                    k_list=[1],
                    language='{eval_config.language}',
                    num_process_evaluate=12,
                    timeout={eval_config.timeout},
                )
            """)

            interpreter = eval_config.interpreter
            shell_command = f"{interpreter} -c {repr(python_script_to_run)}"
            output_dict, _ = await sandbox.execute_code(
                shell_command,
                language="shell",
                timeout=eval_config.timeout * len(samples) + 60,
                max_output_characters=100000,
            )

            if output_dict.get("process_status") != "completed":
                LOG.error(f"Evaluation failed for {jsonl_file.name}. Stderr: {output_dict.get('stderr')}")
                continue

            eval_results_file = temp_eval_file.with_name(f"{temp_eval_file.stem}_eval_results.json")
            if not eval_results_file.exists():
                LOG.warning(f"Expected results file not found: {eval_results_file}")
                continue

            with open(eval_results_file, "r", encoding="utf-8") as f_in:
                eval_grades = json.load(f_in)

            # Write the final, graded output back to the original file
            with open(jsonl_file, "w", encoding="utf-8") as f_out:
                for sample in samples:
                    sample["graded_list"] = eval_grades["eval"][sample["task_id"]]["graded_list"]
                    f_out.write(json.dumps(sample) + "\n")

            # Clean up by moving the results file
            saved_results_file = eval_results_file.with_name(f"{jsonl_file.stem}_eval_results-saved.json")
            shutil.move(str(eval_results_file), str(saved_results_file))
            temp_eval_file.unlink()  # Remove the temporary file
            LOG.info(f"Finished processing and saved results for {jsonl_file.name}")


def eval_livecodebench(cfg):
    """Synchronous wrapper to run the async evaluation."""
    asyncio.run(eval_livecodebench_async(cfg))

import asyncio
import json
import logging
import textwrap
from contextlib import asynccontextmanager
from dataclasses import field
from pathlib import Path

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))

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
    num_processes: int = 12


@asynccontextmanager
async def sandbox_context(config: dict):
    sandbox = get_sandbox(**config)
    try:
        yield sandbox
    finally:
        LOG.info("Closing sandbox...")
        await sandbox.close()


async def install_packages(eval_config: LiveCodeBenchEvaluatorConfig) -> bool:
    """
    Installs required packages in a temporary sandbox.
    Returns True on success, False on failure.
    """
    async with sandbox_context(eval_config.sandbox) as sandbox:
        LOG.info(f"Installing livecodebench with {eval_config.interpreter}...")
        pip_cmd = "pip" if eval_config.interpreter == "python" else "pypy3 -m pip"
        git_url = LIVECODEBENCH_PYTHON_GIT_URL if eval_config.interpreter == "python" else LIVECODEBENCH_PYPY3_GIT_URL
        cmd = f"{pip_cmd} install {git_url}"

        result, _ = await sandbox.execute_code(cmd, language="shell", timeout=300)
        if result.get("process_status") != "completed":
            LOG.warning(f"Failed to install livecodebench: {result.get('stderr', 'Unknown error')}")
            return False

        LOG.info("Successfully installed livecodebench.")
        return True


async def eval_livecodebench_async(cfg):
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg.eval_config)

    if eval_config.language == "python" and eval_config.interpreter not in ["python", "pypy3"]:
        raise ValueError("Python interpreter must be 'python' or 'pypy3'.")
    if eval_config.language == "cpp" and eval_config.test_file is None:
        raise ValueError("C++ evaluation requires a test_file.")

    if not await install_packages(eval_config):
        return

    async with sandbox_context(eval_config.sandbox) as sandbox:
        for jsonl_path in map(Path, unroll_files(cfg.input_files)):
            LOG.info(f"Processing file: {jsonl_path.name}")

            with jsonl_path.open("r", encoding="utf-8") as f_in:
                samples = [preprocess_code(json.loads(line), eval_config.language) for line in f_in]

            versions = {s["release_version"] for s in samples}
            if len(versions) > 1:
                raise ValueError(f"All samples should have the same release version. Found: {versions}")
            release_version = versions.pop()

            for s in samples:
                s["code_list"] = [s["completion"]]

            temp_eval = jsonl_path.with_suffix(".temp.jsonl")
            temp_eval.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")

            test_file_arg = repr(eval_config.test_file) if eval_config.test_file else "None"
            eval_code = textwrap.dedent(f"""from livecodebench.evaluate import evaluate
                evaluate(
                    custom_output_file='{temp_eval.name}',
                    release_version='release_{release_version}',
                    test_file={test_file_arg},
                    k_list=[1],
                    language='{eval_config.language}',
                    num_process_evaluate={eval_config.num_processes},
                    timeout={eval_config.timeout}
                )
            """)

            cmd = f"{eval_config.interpreter} -c {repr(eval_code)}"
            output, _ = await sandbox.execute_code(
                cmd,
                language="shell",
                timeout=eval_config.timeout * len(samples) + 60,
                max_output_characters=100_000,
            )

            if output.get("process_status") != "completed":
                LOG.error(f"Evaluation failed for {jsonl_path.name}. Stderr: {output.get('stderr')}")
                temp_eval.unlink(missing_ok=True)
                continue

            results_path = temp_eval.with_name(f"{temp_eval.stem}_eval_results.json")
            if not results_path.exists():
                LOG.warning(f"Results file missing: {results_path}")
                temp_eval.unlink(missing_ok=True)
                continue

            eval_grades = json.loads(results_path.read_text(encoding="utf-8"))

            with jsonl_path.open("w", encoding="utf-8") as f_out:
                for s in samples:
                    s["graded_list"] = eval_grades["eval"][s["task_id"]]["graded_list"]
                    f_out.write(json.dumps(s) + "\n")

            results_path.rename(results_path.with_name(f"{jsonl_path.stem}_eval_results-saved.json"))
            temp_eval.unlink(missing_ok=True)
            LOG.info(f"Finished {jsonl_path.name}, results saved.")


def eval_livecodebench(cfg):
    """Synchronous wrapper to run the async evaluation."""
    asyncio.run(eval_livecodebench_async(cfg))

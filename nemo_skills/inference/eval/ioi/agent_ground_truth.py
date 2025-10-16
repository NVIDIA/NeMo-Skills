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


model_solution = """
#include "worldmap.h"
#include <vector>
#include <iostream>
using namespace std;

std::vector<std::vector<int> > create_map(int N, int M, std::vector<int> A, std::vector<int> B) {
	// step #1. preparation
	vector<vector<int> > G(N);
	for (int i = 0; i < M; i++) {
		A[i]--;
		B[i]--;
		G[A[i]].push_back(B[i]);
		G[B[i]].push_back(A[i]);
	}

	// step #2. make spanning tree
	vector<bool> vis(N, false);
	vector<int> depth(N);
	vector<int> tour, RA, RB;
	auto dfs = [&](auto& self, int x) -> void {
		vis[x] = true;
		tour.push_back(x);
		for (int i : G[x]) {
			if (!vis[i]) {
				depth[i] = depth[x] + 1;
				self(self, i);
				tour.push_back(x);
			} else if (depth[x] < depth[i]) {
				RA.push_back(x);
				RB.push_back(i);
			}
		}
	};
	dfs(dfs, 0);

	// step #3. construction
	vector<int> rank(N, -1), holder(N, -1);
	for (int i = 0; i < 2 * N - 1; i++) {
		int d = min(i, (2 * N - 2) - i);
		if (rank[tour[i]] < d) {
			rank[tour[i]] = d;
			holder[tour[i]] = i;
		}
	}
	vector<vector<int> > H(N);
	for (int i = 0; i < M - (N - 1); i++) {
		if (rank[RA[i]] < rank[RB[i]]) {
			swap(RA[i], RB[i]);
		}
		H[RA[i]].push_back(RB[i]);
	}
	vector<vector<int> > ans(2 * N, vector<int>(2 * N, 0));
	int cur = 0, parity = 0;
	for (int i = 0; i < 2 * N - 1; i++) {
		if (i == holder[tour[i]]) {
			int pos = 0;
			for (int j = 0; j < 2 * N; j++) {
				int ya = cur - j;
				if (0 <= ya && ya < 2 * N) {
					ans[j][ya] = tour[i];
				}
				int yb = (cur + 1) - j;
				if (0 <= yb && yb < 2 * N) {
					if (pos < int(H[tour[i]].size())) {
						ans[j][yb] = H[tour[i]][pos];
						pos++;
					} else {
						ans[j][yb] = tour[i];
					}
				}
				int yc = (cur + 2) - j;
				if (0 <= yc && yc < 2 * N) {
					ans[j][yc] = tour[i];
				}
			}
			cur += 3;
			parity ^= 1;
		} else {
			for (int j = 0; j < 2 * N; j++) {
				int ya = cur - j;
				if (0 <= ya && ya < 2 * N) {
					ans[j][ya] = tour[i];
				}
			}
			cur += 1;
		}
	}

	// step #4. final step
	for (int i = 0; i < int(ans.size()); i++) {
		for (int j = 0; j < int(ans.size()); j++) {
			ans[i][j]++;
		}
	}

	return ans;
}
"""


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
        num_steps_completed = 0

        all_model_solutions = data_point["correct_solutions"]

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

        return {
            "generation": latest_generation_response,
            "steps": chat_history,
            "num_steps_completed": num_steps_completed,
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

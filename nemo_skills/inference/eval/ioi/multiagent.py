import asyncio
import logging
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import field
from typing import List, Optional, Tuple

import hydra
import litellm

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_cpp_block(text: str) -> Optional[str]:
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_script_block(text: str) -> Optional[str]:
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```script(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_detailed_report(solution: str) -> str:
    solution = solution.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    report_matches = re.findall(r"<report>(.*?)</report>", solution, re.DOTALL)
    if report_matches:
        return report_matches[-1].strip()
    raise ValueError(f"No report found in solution: {solution}")


def extract_boxed_yes_no(text: str) -> str:
    try:
        search_area = extract_detailed_report(text)
    except ValueError:
        search_area = text
    m = re.search(r"\\+boxed\s*\{\s*([^}]*)\s*\}", search_area)
    return m.group(1).strip().lower() if m else ""


def extract_boxed_index(text: str) -> Optional[int]:
    try:
        search_area = extract_detailed_report(text)
    except ValueError:
        search_area = text
    m = re.search(r"\\+boxed\s*\{\s*(\d+)\s*\}", search_area)
    return int(m.group(1)) if m else None


def render_solutions_markdown(solutions: List[str]) -> str:
    parts: List[str] = []
    for idx, sol in enumerate(solutions, start=1):
        parts.append(f"### Proposed Solution {idx} ###\n{sol}\n")
    return "\n".join(parts)


def build_question_with_history(question: str, prev_solutions: List[str]) -> str:
    if not prev_solutions:
        return question
    prev = render_solutions_markdown(prev_solutions)
    return f"{question}\n\n Here are previous proposed solutions for your reference (improve upon or propose a new one if believe none of them are useful):\n{prev}"


@nested_dataclass(kw_only=True)
class MultiAgentConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    # Prompts
    solver_prompt_config: str = "eval/ioi/multiagent/code/solver"
    generate_test_prompt_config: str = "eval/ioi/multiagent/code/generate_test"
    improve_prompt_config: str = "eval/ioi/multiagent/code/improve"
    terminate_prompt_config: str = "eval/ioi/multiagent/code/terminate"
    select_prompt_config: str = "eval/ioi/multiagent/code/select"

    # Orchestration
    total_steps: int = 10
    min_steps_before_terminate: int = 2
    majority_terminate_n: int = 10
    majority_selection_n: int = 10
    skip_termination: bool = False

    # Sub-agent configuration
    execution_steps: int = 3
    test_timeout_s: float = 30.0
    execution_max_output_characters: int = 1000
    agents: List[str] = field(default_factory=lambda: [])


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_multiagent_config", node=MultiAgentConfig)


class BaseSubAgent:
    def __init__(self, task: "MultiAgentGenerationTask"):
        self.task = task
        self.tir_enabled: bool = False

    async def _call(self, prompt_key: str, data_point: dict, all_data: dict, **extra):
        combined = {**data_point, **extra}
        filled = self.task.fill_prompt(combined, all_data, prompt=self.task.prompts[prompt_key])
        start_t = time.time()
        try:
            # When TIR is enabled, pass builtin_tools via generation params override
            generation_params_override = None
            if self.tir_enabled:
                generation_params_override = {"builtin_tools": self.task.cfg.builtin_tools}
            out = await super(MultiAgentGenerationTask, self.task).process_single_datapoint(
                combined,
                all_data,
                prompt=self.task.prompts[prompt_key],
                generation_params_override=generation_params_override,
            )
        except (litellm.exceptions.OpenAIError, Exception) as e:
            print(f"LLM call failed: {e}\nPrompt causing failure:\n{filled}")
            raise
        gen_time = time.time() - start_t
        return filled, out, gen_time

    async def run(self, data_point: dict, all_data: dict, prev_solutions: List[str]) -> Tuple[str, List[dict]]:
        raise NotImplementedError


class SolverAgent(BaseSubAgent):
    async def run(self, data_point: dict, all_data: dict, prev_solutions: List[str]) -> Tuple[str, List[dict]]:
        logs: List[dict] = []
        question_with_history = build_question_with_history(data_point.get("question", ""), prev_solutions)
        filled, out, t = await self._call("solver", {**data_point, "question": question_with_history}, all_data)
        logs.append({"prompt": filled, "response": out["generation"], "generation_time": t})
        sol = extract_cpp_block(out["generation"]) or ""
        if not sol:
            raise ValueError(f"SolverAgent failed to produce a C++ solution block: {out['generation']}")
        return sol, logs


class ExecutionAgent(BaseSubAgent):
    def __init__(self, task: "MultiAgentGenerationTask", steps: int, test_timeout_s: float):
        super().__init__(task)
        self.steps = steps
        self.test_timeout_s = test_timeout_s

    async def _generate_test(self, question: str, solution: str, all_data: dict):
        filled, out, t = await self._call("generate_test", {"question": question, "solution": solution}, all_data)
        return filled, out, t

    async def _improve(self, question: str, solution: str, script: str, output: str, all_data: dict):
        filled, out, t = await self._call(
            "improve",
            {"question": question, "solution": solution, "script": script, "output": output},
            all_data,
        )
        return filled, out, t

    async def _compile_and_run_cpp(self, code_string: str, data_point: dict):
        # Compile the provided C++ code from stdin and run the produced executable, returning stdout and stderr.
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

    async def run(
        self, data_point: dict, all_data: dict, prev_solutions: List[str], starting_solution: Optional[str] = None
    ) -> Tuple[str, List[dict]]:
        logs: List[dict] = []

        # Initial solution
        current_solution: Optional[str] = starting_solution
        question_with_history = build_question_with_history(data_point.get("question", ""), prev_solutions)
        if not current_solution:
            filled_init, out_init, t_init = await self._call(
                "solver", {**data_point, "question": question_with_history}, all_data
            )
            logs.append({"prompt": filled_init, "response": out_init["generation"], "generation_time": t_init})
            current_solution = extract_cpp_block(out_init["generation"]) or ""
            if not current_solution:
                raise ValueError("ExecutionAgent initial solution extraction failed")

        # Iterative execution-improve loop
        for step_idx in range(self.steps):
            print(f"[ExecutionAgent] Step {step_idx + 1}/{self.steps}: Generating test script")
            filled_test, out_test, t_test = await self._generate_test(
                question_with_history, current_solution, all_data
            )
            logs.append({"prompt": filled_test, "response": out_test["generation"], "generation_time": t_test})
            print("[ExecutionAgent] Full test generation (raw):\n" + out_test["generation"])
            script = extract_script_block(out_test["generation"]) or ""
            if not script:
                print(
                    "[ExecutionAgent] Failed to extract test script. Full generation was:\n" + out_test["generation"]
                )
                raise ValueError("Failed to extract test script from generate_test output")

            # Execute test script (standalone C++) via custom compile-and-run helper
            print("[ExecutionAgent] Compiling and executing generated test script (custom runner)")
            try:
                run_stdout, run_stderr = await self._compile_and_run_cpp(script, data_point)
            except Exception:
                print("[ExecutionAgent] Compilation/execution error. Test script was:\n" + script)
                print("[ExecutionAgent] Corresponding full test generation was:\n" + out_test["generation"])
                raise
            # Truncate outputs for stability and prompt cleanliness
            max_chars = max(0, int(self.task.cfg.execution_max_output_characters))
            if max_chars > 0:
                if len(run_stdout) > max_chars:
                    run_stdout = run_stdout[:max_chars] + "<output cut>"
                if len(run_stderr) > max_chars:
                    run_stderr = run_stderr[:max_chars] + "<output cut>"
            exec_output = f"STDOUT:\n{run_stdout}\nSTDERR:\n{run_stderr}"
            print("[ExecutionAgent] Execution result (captured):\n" + exec_output)

            # Improve using feedback
            print("[ExecutionAgent] Improving solution based on execution feedback")
            filled_imp, out_imp, t_imp = await self._improve(
                question_with_history, current_solution, script, exec_output, all_data
            )
            logs.append({"prompt": filled_imp, "response": out_imp["generation"], "generation_time": t_imp})
            new_solution = extract_cpp_block(out_imp["generation"]) or ""
            if not new_solution:
                raise ValueError("Failed to extract improved C++ solution", out_imp["generation"])
            current_solution = new_solution

        return current_solution, logs


class SolverTIRAgent(SolverAgent):
    def __init__(self, task: "MultiAgentGenerationTask"):
        super().__init__(task)
        self.tir_enabled = True


class ExecutionTIRAgent(ExecutionAgent):
    def __init__(self, task: "MultiAgentGenerationTask", steps: int, test_timeout_s: float):
        super().__init__(task, steps=steps, test_timeout_s=test_timeout_s)
        self.tir_enabled = True


class ChainedAgent(BaseSubAgent):
    def __init__(self, task: "MultiAgentGenerationTask", steps: int, test_timeout_s: float):
        super().__init__(task)
        self.exec_agent = ExecutionAgent(task, steps=steps, test_timeout_s=test_timeout_s)

    async def run(self, data_point: dict, all_data: dict, prev_solutions: List[str]) -> Tuple[str, List[dict]]:
        all_logs: List[dict] = []

        # First: a quick solver pass (acts as an improver by providing a strong seed)
        solver = SolverAgent(self.task)
        seed_solution, logs = await solver.run(data_point, all_data, prev_solutions)
        all_logs.extend(logs)

        # Then: execution-driven improvements starting from the seed
        final_solution, exec_logs = await self.exec_agent.run(
            data_point, all_data, prev_solutions, starting_solution=seed_solution
        )
        all_logs.extend(exec_logs)
        return final_solution, all_logs


class MultiAgentGenerationTask(GenerationTask):
    def __init__(self, cfg: MultiAgentConfig):
        super().__init__(cfg)
        self.cfg: MultiAgentConfig
        prompt_kwargs = {"code_tags": cfg.code_tags, "examples_type": cfg.examples_type}
        self.prompts = {
            "solver": get_prompt(cfg.solver_prompt_config, **prompt_kwargs),
            "generate_test": get_prompt(cfg.generate_test_prompt_config, **prompt_kwargs),
            "improve": get_prompt(cfg.improve_prompt_config, **prompt_kwargs),
            "terminate": get_prompt(cfg.terminate_prompt_config, **prompt_kwargs),
            "select": get_prompt(cfg.select_prompt_config, **prompt_kwargs),
        }

        self.sandbox = LocalSandbox()

        # Instantiate sub-agents
        self.available_agents = {
            "solver": lambda: SolverAgent(self),
            "execution": lambda: ExecutionAgent(
                self, steps=self.cfg.execution_steps, test_timeout_s=self.cfg.test_timeout_s
            ),
            "solver_tir": lambda: SolverTIRAgent(self),
            "execution_tir": lambda: ExecutionTIRAgent(
                self, steps=self.cfg.execution_steps, test_timeout_s=self.cfg.test_timeout_s
            ),
            "chained": lambda: ChainedAgent(
                self, steps=self.cfg.execution_steps, test_timeout_s=self.cfg.test_timeout_s
            ),
        }

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

    async def _should_terminate(
        self, data_point: dict, all_data: dict, solutions: List[str]
    ) -> Tuple[bool, List[dict]]:
        logs: List[dict] = []
        solutions_md = render_solutions_markdown(solutions)
        tasks = [
            self._call_llm(
                data_point, all_data, "terminate", question=data_point.get("question", ""), solutions=solutions_md
            )
            for _ in range(self.cfg.majority_terminate_n)
        ]
        results = await asyncio.gather(*tasks)
        yes_votes = 0
        for prompt_txt, resp, gen_time in results:
            logs.append({"prompt": prompt_txt, "response": resp["generation"], "generation_time": gen_time})
            verdict = extract_boxed_yes_no(resp["generation"])  # "yes" / "no"
            print(f"[Terminate] Vote: {verdict}")
            if verdict == "yes":
                yes_votes += 1
        majority = math.floor(self.cfg.majority_terminate_n / 2) + 1
        print(f"[Terminate] Yes votes: {yes_votes}/{self.cfg.majority_terminate_n} (threshold {majority})")
        return yes_votes >= majority, logs

    async def _select_best(self, data_point: dict, all_data: dict, solutions: List[str]) -> Tuple[int, List[dict]]:
        logs: List[dict] = []
        solutions_md = render_solutions_markdown(solutions)
        tasks = [
            self._call_llm(
                data_point, all_data, "select", question=data_point.get("question", ""), solutions=solutions_md
            )
            for _ in range(self.cfg.majority_selection_n)
        ]
        results = await asyncio.gather(*tasks)
        votes = [0] * len(solutions)
        for prompt_txt, resp, gen_time in results:
            logs.append({"prompt": prompt_txt, "response": resp["generation"], "generation_time": gen_time})
            idx = extract_boxed_index(resp["generation"])  # 1-based index
            print(f"[Select] Vote: {idx}")
            if idx is not None and 1 <= idx <= len(solutions):
                votes[idx - 1] += 1
        best_idx = max(range(len(solutions)), key=lambda i: votes[i]) if solutions else 0
        print(f"[Select] Votes: {votes} -> selected {best_idx + 1}")
        return best_idx, logs

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history: List[dict] = []

        # Prepare agents for this run (list of agent names)
        agents: List[BaseSubAgent] = []
        agent_names = [a.strip() for a in self.cfg.agents]
        for a in agent_names:
            factory = self.available_agents.get(a)
            if factory is not None:
                agents.append(factory())
            else:
                print(f"[Warning] Unknown agent type '{a}', skipping")

        if not agents:
            raise ValueError(
                "No agents configured. Please set cfg.agents to include at least one of: solver, execution, solver_tir, execution_tir, chained"
            )

        # Step 0: ask all agents for an initial solution (no history)
        print(f"[MultiAgent] Step 0: Gathering initial solutions from {len(agents)} agents")
        initial_tasks = [agent.run(data_point, all_data, prev_solutions=[]) for agent in agents]
        results = await asyncio.gather(*initial_tasks)
        current_solutions = []
        for sol, logs in results:
            current_solutions.append(sol)
            chat_history.extend(logs)

        # Iterate steps
        for step_num in range(1, self.cfg.total_steps + 1):
            print(f"[MultiAgent] Step {step_num}: Producing new solutions based on prior step outputs")

            # Provide previous solutions to each agent and get new proposals
            step_tasks = [agent.run(data_point, all_data, prev_solutions=current_solutions) for agent in agents]
            step_results = await asyncio.gather(*step_tasks)
            new_solutions: List[str] = []
            for sol, logs in step_results:
                new_solutions.append(sol)
                chat_history.extend(logs)

            current_solutions = new_solutions

            # Check termination after minimum number of steps (unless skipped)
            if (not self.cfg.skip_termination) and step_num + 0 >= max(1, self.cfg.min_steps_before_terminate):
                terminate, term_logs = await self._should_terminate(data_point, all_data, current_solutions)
                chat_history.extend(term_logs)
                if terminate:
                    print("[MultiAgent] Termination condition satisfied. Selecting final solution...")
                    if len(agents) == 1:
                        final_solution = current_solutions[0]
                        return {"generation": final_solution, "steps": chat_history, "num_steps_completed": step_num}
                    best_idx, sel_logs = await self._select_best(data_point, all_data, current_solutions)
                    chat_history.extend(sel_logs)
                    final_solution = current_solutions[best_idx]
                    return {"generation": final_solution, "steps": chat_history, "num_steps_completed": step_num}

        # If not terminated within steps, still select the best at the end
        print("[MultiAgent] Reached maximum steps without termination. Selecting final solution anyway...")
        if len(agents) == 1:
            final_solution = current_solutions[0]
            return {"generation": final_solution, "steps": chat_history, "num_steps_completed": self.cfg.total_steps}
        best_idx, sel_logs = await self._select_best(data_point, all_data, current_solutions)
        chat_history.extend(sel_logs)
        final_solution = current_solutions[best_idx]
        return {"generation": final_solution, "steps": chat_history, "num_steps_completed": self.cfg.total_steps}


GENERATION_TASK_CLASS = MultiAgentGenerationTask


@hydra.main(version_base=None, config_name="base_ioi_multiagent_config")
def ioi_multiagent(cfg: MultiAgentConfig):
    cfg = MultiAgentConfig(_init_nested=True, **cfg)
    LOG.info("Note: IOI MultiAgent Module is being used.")
    LOG.info("Config used: %s", cfg)
    task = MultiAgentGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(MultiAgentConfig, server_params=server_params())


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        ioi_multiagent()

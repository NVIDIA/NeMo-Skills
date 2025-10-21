import logging
import re
import sys
import time
from dataclasses import field

import hydra
import litellm

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def extract_code_block(text: str):
    # todo (sean): this is a hack to prevent catching report tags in the CoT, causing parsing errors for gpt-oss.
    text = text.split("<|end|><|start|>assistant<|channel|>final<|message|>")[-1]
    matches = re.findall(r"```cpp(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


@nested_dataclass(kw_only=True)
class IOIExecutionConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_config: str = "eval/ioi/agent/solver"
    improve_after_verify_prompt_config: str = "eval/ioi/agent/improve_with_private_test"
    total_steps: int = 60


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_ioi_generation_config", node=IOIExecutionConfig)


class IOIExecutionGenerationTask(GenerationTask):
    def __init__(self, cfg: IOIExecutionConfig):
        super().__init__(cfg)
        prompt_kwargs = {
            "examples_type": cfg.examples_type,
        }
        self.prompts = {
            "initial": get_prompt(cfg.prompt_config, **prompt_kwargs),
            "improve_after_verify_solution": get_prompt(cfg.improve_after_verify_prompt_config, **prompt_kwargs),
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

    async def process_single_datapoint(self, data_point, all_data, prompt=None):
        chat_history = []
        num_steps_completed = 0

        # Attempt to resume from latest intermediate state
        async_pos = data_point[self.cfg.async_position_key]
        saved_state = self.load_latest_state(async_pos)
        if saved_state and saved_state.get("_intermediate", False):
            chat_history = saved_state.get("steps", [])
            cur_generation_response = saved_state.get("generation")
            num_steps_completed = int(saved_state.get("num_steps_completed", 0))
            print(f"[Resume] Restoring pos={async_pos} from step {num_steps_completed}")
        else:
            prompt_txt, solution_response, gen_time = await self._call_llm(data_point, all_data, "initial")
            cur_generation_response = solution_response["generation"]
            chat_history.append(
                {"prompt": prompt_txt, "response": cur_generation_response, "generation_time": gen_time}
            )

            print("[Initial] Generated initial solution.")
            # Save checkpoint after initial solution
            print(f"[Checkpoint] Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                },
            )

        for step_num in range(num_steps_completed, self.cfg.total_steps):
            # Evaluate the current solution using the external evaluator.
            # Time the external evaluator to capture evaluation latency.
            eval_start_t = time.time()
            eval_results = await self.evaluator.eval_single({**data_point, "generation": cur_generation_response})
            eval_time = time.time() - eval_start_t

            # Record evaluation time for the solution that was just evaluated (last entry in chat_history).
            if chat_history:
                chat_history[-1]["evaluation_time"] = eval_time
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

            # Save checkpoint after each improvement step
            print(f"[Checkpoint] Saving intermediate pos={async_pos}, step={num_steps_completed}")
            await self.save_intermediate_state(
                async_pos,
                {
                    "generation": cur_generation_response,
                    "steps": chat_history,
                    "num_steps_completed": num_steps_completed,
                },
            )

        # Reached maximum steps without passing all tests.
        print("[Failure] Reached max improvement steps without passing all tests.")
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

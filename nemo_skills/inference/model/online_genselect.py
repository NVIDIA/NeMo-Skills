# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import random
import re
from dataclasses import field
from typing import Dict, List, Optional, Union

from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, nested_dataclass, remove_thinking

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class OnlineGenSelectConfig:
    max_concurrent_requests: int = 8
    max_num_solutions: int = 8
    prompt_config: str = "generic/genselect"
    prompt_template: str | None = None
    temperature: float = 0.6
    tokens_to_generate: int = 2048
    comparison_key: str = "generation"  # Key used for comparing the different solutions
    regex: str = r"Judg[e]?ment: (\d+)"
    remove_thinking: bool = True  # Remove thinking tokens from the comparison key
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"


class OnlineGenSelectWrapper:
    """
    Wrapper that generates multiple completions for a datapoint and uses GenSelect
    to choose the best one.
    """

    def __init__(self, model: BaseModel, cfg: OnlineGenSelectConfig):
        self.model = model
        self.cfg = cfg

        # Load GenSelect prompt
        self.genselect_prompt = get_prompt(self.cfg.prompt_config, prompt_template=self.cfg.prompt_template)
        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

    def close(self):
        """Close the wrapped model."""
        if hasattr(self.model, 'close'):
            self.model.close()

    def _extract_judgment(self, generation: str, max_idx: int) -> Optional[int]:
        """Extract the judgment index from GenSelect generation."""
        judgment = None

        try:
            matches = re.findall(self.cfg.regex, generation)
            if matches:
                number = matches[-1]
                judgment = int(number)
                if judgment > max_idx:
                    judgment = None
            else:
                judgment = None
        except Exception:
            judgment = None

        if judgment is not None and judgment > max_idx:
            judgment = None

        return judgment

    def _format_solutions_for_genselect(self, solutions: List[Dict]) -> str:
        """Format solutions for GenSelect prompt."""
        formatted_solutions = []
        for i, solution in enumerate(solutions):
            formatted_solutions.append(f"Solution {i}: {solution[self.cfg.comparison_key]}")
        return "\n\n".join(formatted_solutions)

    async def _run_genselect(
        self, prompt: str, solutions: List[Dict], local_random: random.Random
    ) -> tuple[int, Dict]:
        """Run GenSelect to choose the best solution."""
        # Step 1: Format the solutions for GenSelect
        num_solutions = len(solutions)
        max_idx = num_solutions - 1
        solutions_text = self._format_solutions_for_genselect(solutions)

        genselect_input = {
            'problem': prompt,
            'solutions': solutions_text,
            'num_solutions': num_solutions,
            'max_idx': max_idx,
        }
        genselect_prompt = self.genselect_prompt.fill(genselect_input)

        # Step 2: Run Self-GenSelect
        genselect_result = await self.model.generate_async(
            prompt=genselect_prompt,
            tokens_to_generate=self.cfg.tokens_to_generate,
            temperature=self.cfg.temperature,
            remove_stop_phrases=True,
        )

        # Step 3: Extract the judgment from the GenSelect result
        judgment = self._extract_judgment(genselect_result['generation'], max_idx)
        if judgment is None:
            LOG.warning("GenSelect failed to produce valid judgment, falling back to random selection")
            judgment = local_random.randint(0, max_idx)

        return judgment, genselect_result

    async def generate_async(
        self,
        prompt: Union[str, List],
        random_seed: int = 0,
        **solution_kwargs,
    ) -> Dict:
        """
        Generate multiple solutions and use Self-GenSelect to choose the best one.
        """
        # Step 1: Generate multiple solutions
        local_random = random.Random(random_seed)
        tasks = []
        for _ in range(self.cfg.max_num_solutions):
            # Generate solutions with different seeds for diversity
            cur_random_seed = local_random.getrandbits(32)
            # Create a copy to avoid mutation issues
            current_kwargs = solution_kwargs.copy()
            current_kwargs['random_seed'] = cur_random_seed

            task = self.model.generate_async(prompt=prompt, **current_kwargs)
            tasks.append(task)

        generation_results = await asyncio.gather(*tasks)
        solutions = []
        for generation_result in generation_results:
            if self.cfg.remove_thinking:
                remove_thinking(
                    generation_result,
                    generation_key=self.cfg.comparison_key,
                    thinking_begin=self.cfg.thinking_begin,
                    thinking_end=self.cfg.thinking_end,
                )

            solutions.append(
                {
                    self.cfg.comparison_key: generation_result[self.cfg.comparison_key],
                    "output_dict": generation_result,
                }
            )

        local_random.shuffle(solutions)

        # Step 2: Run GenSelect to choose the best solution
        best_index, genselect_result = await self._run_genselect(prompt, solutions, local_random)
        best_solution = solutions[best_index]

        # Return the best solution in the expected format
        result = {
            self.cfg.comparison_key: best_solution[self.cfg.comparison_key],
            "solution_list": [solution[self.cfg.comparison_key] for solution in solutions],
            "genselect_comparison": genselect_result["generation"],
        }

        total_num_generated_tokens = 0
        for solution in solutions:
            total_num_generated_tokens += solution["output_dict"].get('num_generated_tokens', 0)
        result["total_solution_generated_tokens"] = total_num_generated_tokens

        # Add the tokens for genselect
        result["genselect_num_generated_tokens"] = genselect_result.get("num_generated_tokens", 0)

        # Add the tokens for all the solutions and genselect
        total_gen_tokens = result["total_solution_generated_tokens"] + result["genselect_num_generated_tokens"]

        # TODO: Decide what count of generated tokens do we want to report - the total or the best solution?
        # Current implementation returns the total number of generated tokens
        result["num_generated_tokens"] = total_gen_tokens

        # Add the tokens for the best solution
        result['num_best_solution_generated_tokens'] = best_solution["output_dict"].get("num_generated_tokens", 0)

        return result

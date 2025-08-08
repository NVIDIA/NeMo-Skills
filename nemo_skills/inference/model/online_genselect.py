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
from nemo_skills.utils import get_logger_name, nested_dataclass

from .base import BaseModel

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class OnlineGenSelectConfig:
    max_num_solutions: int = 8
    genselect_prompt_config: str = "generic/genselect"
    genselect_temperature: float = 0.6
    genselect_max_tokens: int = 2048
    comparison_key: str = "generation"
    genselect_regex: str = r"Judg[e]?ment: (\d+)"


class OnlineGenSelectWrapper:
    """
    Wrapper that generates multiple completions for a datapoint and uses GenSelect
    to choose the best one.
    """

    def __init__(self, model: BaseModel, cfg: OnlineGenSelectConfig):
        self.model = model
        self.cfg = cfg

        # Load GenSelect prompt
        self.genselect_prompt = get_prompt(self.cfg.genselect_prompt_config)

    def _extract_judgment(self, generation: str, max_idx: int) -> Optional[int]:
        """Extract the judgment index from GenSelect generation."""
        judgment = None

        try:
            matches = re.findall(self.cfg.genselect_regex, generation)
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
            formatted_solutions.append(f"Solution {i}: {solution['generation']}")
        return "\n\n".join(formatted_solutions)

    async def _run_genselect(self, problem: str, solutions: List[Dict]) -> int:
        """Run GenSelect to choose the best solution."""
        num_solutions = len(solutions)
        max_idx = num_solutions - 1

        solutions_text = self._format_solutions_for_genselect(solutions)

        # Create GenSelect prompt
        genselect_input = {
            'problem': problem,
            'solutions': solutions_text,
            'num_solutions': num_solutions,
            'max_idx': max_idx,
        }
        genselect_prompt = self.genselect_prompt.fill(genselect_input)

        # Generate GenSelect judgment
        genselect_result = await self.model.generate_async(
            prompt=genselect_prompt,
            tokens_to_generate=self.config.genselect_max_tokens,
            temperature=self.config.genselect_temperature,
            remove_stop_phrases=True,
        )

        judgment = self._extract_judgment(genselect_result['generation'], max_idx)

        if judgment is None:
            LOG.warning("GenSelect failed to produce valid judgment, falling back to random selection")
            judgment = random.randint(0, max_idx)

        return judgment

    async def generate_async(
        self,
        prompt: Union[str, List],
        random_seed: int = 0,
        **kwargs,
    ) -> Dict:
        """
        Generate multiple solutions and use GenSelect to choose the best one.
        """

        # Generate multiple solutions
        solutions = []
        generation_kwargs = {
            'prompt': prompt,
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'stop_phrases': stop_phrases,
            'timeout': timeout,
            'remove_stop_phrases': remove_stop_phrases,
            'reasoning_effort': reasoning_effort,
            'tools': tools,
            'include_response': include_response,
            'extra_body': extra_body,
        }

        # Generate solutions with different seeds for diversity
        for i in range(self.cfg.max_num_solutions):
            solution_kwargs = generation_kwargs.copy()
            solution_kwargs['random_seed'] = random_seed + i

            solution_result = await self.model.generate_async(**solution_kwargs)

            # Extract answer from the generation
            predicted_answer = self._extract_answer(solution_result['generation'])

            solutions.append(
                {
                    'generation': solution_result['generation'],
                    self.cfg.answer_key: predicted_answer,
                    'solution_index': i,
                }
            )

        # Run GenSelect to choose the best solution
        best_index = await self._run_genselect(problem_text, solutions)
        best_solution = solutions[best_index]

        # Return the best solution in the expected format
        result = {
            'generation': best_solution['generation'],
            'genselect_chosen_index': best_index,
            'genselect_num_solutions': self.cfg.max_num_solutions,
        }

        # Add answer if extracted
        if self.cfg.answer_key in best_solution:
            result[self.cfg.answer_key] = best_solution[self.cfg.answer_key]

        # Add generation metadata from the chosen solution if available
        if 'num_generated_tokens' in best_solution:
            result['num_generated_tokens'] = best_solution['num_generated_tokens']

        return result

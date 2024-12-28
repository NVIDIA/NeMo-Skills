# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import sys
from dataclasses import asdict, field
from pathlib import Path
from typing import Any

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.generate import InferenceConfig
from nemo_skills.inference.server.code_execution_model import get_code_execution_model, get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)

# TODO: should we move slightly confusing input/output dir and rs to the pipeline wrapper?


@nested_dataclass(kw_only=True)
class LlmMathJudgeConfig:
    """Top-level parameters for the script"""

    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset
    output_file: str | None = None  # Where to save the generations if `input_file` is provided
    # Can specify an input directory, where the file will be inferred output.jsonl if no seed
    # is provided, and output-rs{{seed}}.jsonl. This pattern is used to match the output files from
    # the `generate` pipeline
    input_dir: str | None = None
    # Where to save the generations (with the identical file name) if `input_dir` is provided
    output_dir: str | None = None
    # Used to identify the input file if `input_dir` is provided. If `random_seed` is not provided,
    # the input will be assumed to be from 'greedy' generation
    random_seed: str | None = None
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str = "judge/math"
    examples_type: str | None = None  # to be able to customize few-shot examples
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    batch_size: int = 128
    generation_key: str = "judgement"

    skip_filled: bool = False  # skip already filled judgements

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # set to True if code execution needs to be supported
    code_execution: bool = False

    def __post_init__(self):
        if self.random_seed.strip() == 'None':
            self.random_seed = None
        if self.input_file is None and self.input_dir is not None:
            seed = f'-rs{self.random_seed}' if self.random_seed is not None else ''
            self.input_file = Path(self.input_dir) / f"output{seed}.jsonl"
            self.output_file = Path(self.output_dir) / f"output{seed}.jsonl"
        elif self.input_file is not None and self.input_dir is None:
            if self.output_file is None:
                raise ValueError("Output file should be provided if providing `input_file`")
        else:
            raise ValueError("`input_file` and `input_dir` cannot be provided at the same time")

        if self.server.server_type != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server.server_type == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_llm_math_judge_config", node=LlmMathJudgeConfig)


def prefill_judgement(data_point: dict) -> str | None:
    """Will automatically fill judgement if there is an exact match or the answer is None."""
    if data_point['predicted_answer'] is None:
        return "Reasoning: No answer was provided.\nJudgement: No"

    if str(data_point['predicted_answer']).strip() == str(data_point['expected_answer']).strip():
        return "Reasoning: The two answers are identical.\nJudgement: Yes"

    return None


@hydra.main(version_base=None, config_name='base_llm_math_judge_config', config_path='.')
def llm_math_judge(cfg: LlmMathJudgeConfig):
    cfg = LlmMathJudgeConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)
    if cfg.code_execution:
        sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
        llm = get_code_execution_model(**cfg.server, sandbox=sandbox)
    else:
        llm = get_model(**cfg.server)

    # if using code execution, we need some extra parameters for generate call
    if cfg.code_execution:
        extra_generate_params = prompt.get_code_execution_args()
    else:
        extra_generate_params = {}

    # making sure output dir exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.input_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")
    # additionally, skipping whatever is pre-filled, assuming offset didn't change
    data = data[starting_idx:]

    if len(data) == 0:  # we might not have any examples if skip_filled=True
        return

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    if "predicted_answer" not in data[0]:
        data[0]["predicted_answer"] = extract_answer(data[0]["generation"])

    LOG.info("Prompt used: %s", prompt)
    LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", data[0], prompt.fill(data[0]))

    data_points = []
    judgements = []
    prefilled_indices = []

    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in enumerate(tqdm(data, initial=starting_idx, total=len(data) + starting_idx)):
            if "predicted_answer" not in data_point:
                data_point["predicted_answer"] = extract_answer(data_point["generation"])
            if data_point["expected_answer"] is None:
                raise ValueError(f"Expected answer is required for judgement, found None at line {idx}")
            judgement = prefill_judgement(data_point)
            if judgement is None:
                data_points.append(data_point)
            else:
                judgements.append({cfg.generation_key: judgement, **data_point})
                prefilled_indices.append(idx)

            if len(data_points) == cfg.batch_size or idx == len(data) - 1:
                prompts = [prompt.fill(dp) for dp in data_points]
                stop_phrases = prompt.stop_phrases

                if len(prompts) > 0:
                    outputs = llm.generate(
                        prompts=prompts,
                        stop_phrases=stop_phrases,
                        **asdict(cfg.inference),
                        **extra_generate_params,
                    )

                prefilled_idx = 0
                generated_idx = 0
                for cur_idx in range(idx - len(data_points) - len(judgements) + 1, idx + 1):
                    if cur_idx in prefilled_indices:
                        fout.write(json.dumps(judgements[prefilled_idx]) + "\n")
                        prefilled_idx += 1
                    else:
                        output = outputs[generated_idx]
                        output[cfg.generation_key] = output.pop("generation")
                        data_points[generated_idx].pop(cfg.generation_key, None)
                        output.update(data_points[generated_idx])
                        fout.write(json.dumps(output) + "\n")
                        generated_idx += 1

                data_points.clear()
                judgements.clear()
                prefilled_indices.clear()


HELP_MESSAGE = get_help_message(
    LlmMathJudgeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        llm_math_judge()

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

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.generate import InferenceConfig
from nemo_skills.inference.server.code_execution_model import (
    ErrorRecoveryConfig,
    get_code_execution_model,
    get_model,
    server_params,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_fields_docstring, get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class LlmMathJudgeConfig:
    """Top-level parameters for the script"""

    input_files: Any  # will update them with judgements
    # Inference server configuration {server_params} {error_recovery_params}
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
        if isinstance(self.input_files, str):
            self.input_files = self.input_files.split(" ")

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


# TODO: should we change this to write to different files, so that it's easy to do things like majority voting?
@hydra.main(version_base=None, config_name='base_llm_math_judge_config', config_path='.')
def llm_math_judge(cfg: LlmMathJudgeConfig):
    cfg = LlmMathJudgeConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)
    if cfg.code_execution:
        sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
        llm = get_code_execution_model(**cfg.server, sandbox=sandbox)
    else:
        llm = get_model(**cfg.server)

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    LOG.info("Prompt used: %s", prompt)

    # if using code execution, we need some extra parameters for generate call
    if cfg.code_execution:
        extra_generate_params = {
            "code_begin": prompt.config.template.code_begin,
            "code_end": prompt.config.template.code_end,
            "code_output_begin": prompt.config.template.code_output_begin,
            "code_output_end": prompt.config.template.code_output_end,
        }
    else:
        extra_generate_params = {}

    # assuming everything fits in memory for simplicity
    all_files = unroll_files(cfg.input_files)
    if not all_files:
        raise ValueError(f"No files found for the input pattern: {cfg.input_files}")
    for jsonl_file in all_files:
        LOG.info("Processing file: %s", jsonl_file)
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]

        LOG.info("Example prompt:\nData dictionary: %s\nPrompt string: %s", data[0], prompt.fill(data[0]))

        if cfg.dry_run:
            return

        if cfg.skip_filled and all(cfg.generation_key in data_point for data_point in data):
            continue

        data_points = []
        judgements = []
        prefilled_indices = []

        output_file = jsonl_file + '-judgement'
        starting_idx = 0
        if cfg.skip_filled:
            try:
                with open(output_file, "rt", encoding="utf-8") as fin:
                    starting_idx = len(fin.readlines())
            except FileNotFoundError:
                LOG.warning(f"File `{output_file}` not found, starting from scratch")
        data = data[starting_idx:]

        # saving to a tmp file to avoid corrupting original generation in case something goes wrong
        with open(output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
            for idx, data_point in enumerate(tqdm(data, initial=starting_idx, total=len(data) + starting_idx)):
                data_point.pop(cfg.generation_key, None)
                judgement = prefill_judgement(data_point)
                if judgement is None:
                    data_points.append(data_point)
                else:
                    judgements.append({cfg.generation_key: judgement, **data_point})
                    prefilled_indices.append(idx)

                if len(data_points) == cfg.batch_size or idx == len(data) - 1:
                    prompts = [prompt.fill(dp) for dp in data_points]
                    stop_phrases = prompt.stop_phrases

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
                            output.update(data_points[generated_idx])
                            fout.write(json.dumps(output) + "\n")
                            generated_idx += 1

                    data_points.clear()
                    judgements.clear()
                    prefilled_indices.clear()

        # replacing original file with the judgement file
        Path(output_file).replace(jsonl_file)


error_recovery_params = '\n' + get_fields_docstring(
    ErrorRecoveryConfig,
    prefix='server.error_recovery.',
    level=2,
)


HELP_MESSAGE = get_help_message(
    LlmMathJudgeConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
    error_recovery_params=error_recovery_params,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        llm_math_judge()

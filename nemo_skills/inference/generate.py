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

import importlib
import json
import logging
import sys
from copy import deepcopy
from dataclasses import asdict, field
from pathlib import Path

import hydra
from omegaconf import open_dict
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import get_sandbox, sandbox_params
from nemo_skills.inference.server.code_execution_model import (
    ErrorRecoveryConfig,
    get_code_execution_model,
    get_model,
    server_params,
)
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_fields_docstring, get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class InferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int = 0
    top_p: float = 0.95
    random_seed: int = 0
    tokens_to_generate: int = 2048
    repetition_penalty: float = 1.0


@nested_dataclass(kw_only=True)
class GenerateSolutionsConfig:
    """LLM generation parameters."""

    output_file: str  # Where to save the generations
    # Inference server configuration {server_params} {error_recovery_params}
    server: dict = field(default_factory=dict)
    # Sandbox configuration {sandbox_params}
    sandbox: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str | None = None  # we will fetch it from dataset dir if not provided
    examples_type: str | None = None  # to be able to customize few-shot examples
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    # Can specify one of the existing datasets.
    dataset: str | None = None
    split: str | None = None  # Can be train, validation, test or train_full (train + validation)
    input_file: str | None = None  # Can directly specify an input file, if using a custom dataset

    batch_size: int = 128
    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # if > 0, will skip this many samples from the beginning of the data file.
    # Useful if need to run multiple slurm jobs on the same data file
    offset: int = 0

    generation_key: str = "generation"
    # if specified, we will have a loop over that key in the data file and
    # treat each element as a new turn of conversation
    # E.g. if multi_turn_key="turns" and a line in your data file has
    # turns: ['Hey how are you?', 'And where do you live?']
    # the generations will also be a list with the first entry corresponding to prompt
    # with the first question, second entry to both first question, first answer and second question
    # and so on
    multi_turn_key: str | None = None

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    # set to True if code execution needs to be supported
    code_execution: bool = False

    def __post_init__(self):
        if self.input_file is not None:
            if self.dataset is not None or self.split is not None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided, but not both")
        else:
            if self.dataset is None or self.split is None:
                raise ValueError("Either `input_file` or `dataset` and `split` should be provided")
            self.input_file = Path(__file__).parents[1] / "dataset" / self.dataset / f"{self.split}.jsonl"

        if self.dataset is None and self.prompt_config is None:
            raise ValueError("If `dataset` is not provided, `prompt_config` is required")

        if self.server["server_type"] == "trtllm" and self.prompt_template is None:
            # TODO: fix that
            raise ValueError("Prompt template is required for trtllm servers")

        if self.server["server_type"] == "nemo" and self.prompt_template is not None:
            LOG.warning(
                "NeMo implementation of openai chat completions api doesn't support batching and thus is very slow. "
                "Until this is fixed, we highly recommend that you provide prompt template explicitly."
            )

        if self.server["server_type"] == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_generation_config", node=GenerateSolutionsConfig)


@hydra.main(version_base=None, config_name='base_generation_config')
def generate(cfg: GenerateSolutionsConfig):
    cfg = GenerateSolutionsConfig(_init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)

    if cfg.prompt_template is None and cfg.server["server_type"] != "openai":
        # TODO: handle this explicitly in model.py clients
        # switching to OpenAI client always if prompt template is not provided
        with open_dict(cfg.server):
            cfg.server["server_type"] = "openai"
            cfg.server["model"] = "model"
        if cfg.code_execution:
            raise ValueError("Code execution is not supported for OpenAI server")

    if cfg.code_execution:
        sandbox = get_sandbox(**cfg.sandbox) if cfg.sandbox is not None else None
        llm = get_code_execution_model(**cfg.server, sandbox=sandbox)
    else:
        llm = get_model(**cfg.server)

    # making sure output dir exists
    Path(cfg.output_file).absolute().parent.mkdir(parents=True, exist_ok=True)

    # we currently assume the dataset is small enough to be loaded into memory
    data = []
    with open(cfg.input_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            data.append(json.loads(line))

    # skipping based on the offset first
    data = data[cfg.offset :]

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")

    # additionally, skipping whatever is pre-filled, assuming offset didn't change
    data = data[starting_idx:]
    if cfg.prompt_config is None:
        # fetching from the default for corresponding dataset
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{cfg.dataset}")
        cfg.prompt_config = dataset_module.PROMPT_CONFIG

    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    LOG.info("Prompt used: %s", prompt)

    # need to account for anything that's prefilled
    if 0 <= cfg.max_samples <= starting_idx:
        cfg.max_samples = 0

    if starting_idx < cfg.max_samples:
        cfg.max_samples -= starting_idx

    if cfg.max_samples < 0 or cfg.max_samples > len(data):
        cfg.max_samples = len(data)

    if len(data) == 0:  # we might not have any examples if skip_filled=True
        return

    if cfg.multi_turn_key is None:
        LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", data[0], prompt.fill(data[0]))
    else:
        first_sample = deepcopy(data[0])
        first_sample[cfg.multi_turn_key] = first_sample[cfg.multi_turn_key][:1]
        LOG.info(
            "Example prompt (first turn only):\nData dictionary: %s\nPrompt: %s",
            first_sample,
            prompt.fill(first_sample, multi_turn_key=cfg.multi_turn_key),
        )

    if cfg.dry_run:
        return

    # if using code execution, we need some extra parameters for generate call
    if cfg.code_execution:
        extra_generate_params = {
            "code_begin": prompt.config.template.code_begin,
            "code_end": prompt.config.template.code_end,
            "code_output_begin": prompt.config.template.code_output_begin,
            "code_output_end": prompt.config.template.code_output_end,
            "code_output_format": prompt.config.template.code_output_format,
        }
    else:
        extra_generate_params = {}

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        data_points = []
        for idx, data_point in tqdm(enumerate(data), initial=starting_idx, total=len(data) + starting_idx):
            if idx >= cfg.max_samples:
                break
            data_point.pop(cfg.generation_key, None)
            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == cfg.max_samples - 1:
                if cfg.multi_turn_key is None:
                    outputs = llm.generate(
                        prompts=[prompt.fill(dp) for dp in data_points],
                        stop_phrases=prompt.stop_phrases,
                        **asdict(cfg.inference),
                        **extra_generate_params,
                    )
                else:
                    # TODO: this will not be efficient if different elements have different number of turns
                    # (effective batch size gets smaller). Need to rewrite it to ensure batch size is filled
                    # no matter the turns. Also even the below implementation can probably be simplified
                    turn_data_points = deepcopy(data_points)
                    dp_indices = list(range(len(turn_data_points)))
                    cur_turn = 1
                    outputs = [{"generation": []} for _ in range(len(data_points))]
                    while dp_indices:
                        # updating the turns to only have data up-to the current turn
                        # and adding any generated assistant messages
                        for dp_index in dp_indices:
                            turn_data_points[dp_index][cfg.multi_turn_key] = data_points[dp_index][cfg.multi_turn_key][
                                :cur_turn
                            ]
                            for turn_idx in range(cur_turn - 1):
                                turn_data_points[dp_index][cfg.multi_turn_key][turn_idx]['assistant'] = outputs[
                                    dp_index
                                ]["generation"][turn_idx]
                        # getting a new set of generations
                        turn_outputs = llm.generate(
                            prompts=[
                                prompt.fill(turn_data_points[dp_index], multi_turn_key=cfg.multi_turn_key)
                                for dp_index in dp_indices
                            ],
                            stop_phrases=prompt.stop_phrases,
                            **asdict(cfg.inference),
                            **extra_generate_params,
                        )
                        # adding assistant answers to the generations
                        for pos_index, dp_index in enumerate(dp_indices):
                            outputs[dp_index]["generation"].append(turn_outputs[pos_index]["generation"])

                        # removing any indices that got through all turns
                        dp_indices = []
                        for dp_index, (output, dp) in enumerate(zip(outputs, data_points)):
                            if len(output["generation"]) < len(dp[cfg.multi_turn_key]):
                                dp_indices.append(dp_index)
                        cur_turn += 1

                for output, original_data_point in zip(outputs, data_points):
                    # to make it easier to follow up with evaluation and limit accidental errors, we are adding
                    # all of the ground-truth data to the output file alongside the generated solutions
                    output[cfg.generation_key] = output.pop("generation")
                    output.update(original_data_point)

                    fout.write(json.dumps(output) + "\n")
                data_points = []


error_recovery_params = '\n' + get_fields_docstring(
    ErrorRecoveryConfig,
    prefix='server.error_recovery.',
    level=2,
)


HELP_MESSAGE = get_help_message(
    GenerateSolutionsConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
    error_recovery_params=error_recovery_params,
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()

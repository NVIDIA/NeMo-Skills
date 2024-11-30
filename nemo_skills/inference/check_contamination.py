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

import hydra
from tqdm import tqdm

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import InferenceConfig
from nemo_skills.inference.server.code_execution_model import get_model, server_params
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__file__)


@nested_dataclass(kw_only=True)
class CheckContaminationConfig:
    """Top-level parameters for the script"""

    input_file: str  # an output of the retrieve_similar.py script
    output_file: str  # where to save the generations
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)
    # Prompt configuration - path to yaml files
    prompt_template: str | None = None  # not required for OpenAI server
    prompt_config: str = "judge/check-contamination"
    examples_type: str | None = None  # to be able to customize few-shot examples
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters

    batch_size: int = 128
    generation_key: str = "contaminated"
    retrieve_key: str = "problem"  # will be used to fill in prompt with retrieve_key1 and retrieve_key2

    skip_filled: bool = False  # skip already filled generations

    # ask both with retrieve_key1 / retrieve_key2 and retrieve_key2 / retrieve_key1 and fill True if any is True
    check_both_ways: bool = False

    # can add this flag to just print the first prompt instead of running generation
    # useful to double check that your data can be loaded and prompt has what you expect
    dry_run: bool = False

    def __post_init__(self):
        if self.server.server_type != "openai" and self.prompt_template is None:
            raise ValueError("Prompt template is required for non-OpenAI servers")

        if self.server.server_type == "openai" and self.prompt_template is not None:
            raise ValueError("Prompt template is not supported for OpenAI server")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_check_contamination_config", node=CheckContaminationConfig)


# TODO: try to unify common parts in 3 similar generation scripts
@hydra.main(version_base=None, config_name='base_check_contamination_config', config_path='.')
def check_contamination(cfg: CheckContaminationConfig):
    cfg = CheckContaminationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    llm = get_model(**cfg.server)
    prompt = get_prompt(cfg.prompt_config, cfg.prompt_template, examples_type=cfg.examples_type)
    LOG.info("Prompt used: %s", prompt)

    # assuming everything fits in memory for simplicity
    with open(cfg.input_file, 'rt', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    first_element = {
        f'{cfg.retrieve_key}1': data[0][cfg.retrieve_key],
        f'{cfg.retrieve_key}2': data[0]['similar_items'][0],
    }
    LOG.info(
        "Example prompt:\nData dictionary: %s\nPrompt: %s",
        first_element,
        prompt.fill(first_element),
    )

    if cfg.dry_run:
        return

    data_points = []

    # we need to make top_k (* 2 if cfg.check_both_ways) generations for each data point
    # correcting to not exceed the requesting batch size
    top_k = len(data[0]['similar_items'])
    cfg.batch_size = max(1, cfg.batch_size // top_k // (2 if cfg.check_both_ways else 1))

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")
    data = data[starting_idx:]
    total = 0
    num_contaminated = 0

    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in enumerate(tqdm(data, initial=starting_idx, total=len(data) + starting_idx)):
            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == len(data) - 1:
                # constructing the data for llm calls
                all_data = []
                for original_data_point in data_points:
                    for similar_item in original_data_point['similar_items']:
                        all_data.append(
                            {
                                f'{cfg.retrieve_key}1': original_data_point[cfg.retrieve_key],
                                f'{cfg.retrieve_key}2': similar_item,
                            }
                        )

                        if cfg.check_both_ways:
                            all_data.append(
                                {
                                    f'{cfg.retrieve_key}2': original_data_point[cfg.retrieve_key],
                                    f'{cfg.retrieve_key}1': similar_item,
                                }
                            )

                prompts = [prompt.fill(dp) for dp in all_data]
                stop_phrases = prompt.stop_phrases

                outputs = llm.generate(prompts=prompts, stop_phrases=stop_phrases, **asdict(cfg.inference))
                output_idx = 0
                for original_data_point in data_points:
                    all_generations = []
                    elem = {}
                    contaminated = False
                    for output in outputs[output_idx : output_idx + top_k * (2 if cfg.check_both_ways else 1)]:
                        all_generations.append(output['generation'])
                        if output['generation'].strip() == "True":
                            contaminated = True
                        output_idx += 1
                    elem[cfg.generation_key] = contaminated
                    if contaminated:
                        num_contaminated += 1
                    total += 1
                    elem["all_generations"] = all_generations
                    original_data_point.pop(cfg.generation_key, None)
                    original_data_point.pop("all_generations", None)
                    elem.update(original_data_point)
                    fout.write(json.dumps(elem) + '\n')
                data_points = []
    if total > 0:
        LOG.info("Contamination portion: %.2f%% (%d/%d)", 100 * num_contaminated / total, num_contaminated, total)


HELP_MESSAGE = get_help_message(
    CheckContaminationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        check_contamination()

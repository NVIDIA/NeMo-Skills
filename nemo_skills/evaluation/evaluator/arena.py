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

import json
import logging
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from nemo_skills.inference.server.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


JUDGE_MODEL = 'gpt-4-1106-preview'
JUDGE_SERVER = 'openai'


@nested_dataclass(kw_only=True)
class LlmEvaluatorConfig:
    batch_size: int = 100  # lower if running into rate limits
    tokens_to_generate: int = 4096  # will auto-lower to max possible for NGC models
    use_batch_api: bool = True  # only supported for OpenAI models!
    base_url: str = "https://api.openai.com/v1"
    judge_model: str = JUDGE_MODEL
    # defaults to True to avoid regenerating judgements unless necessary
    skip_filled: bool = True


# TODO: this needs to be moved into a separate job as we might need to host the server
def eval_arena(cfg):
    eval_config = LlmEvaluatorConfig(**cfg.eval_config)
    assert eval_config.batch_size % 2 == 0  # required due to how everything is implemented, can fix later

    if eval_config.use_batch_api and eval_config.base_url != "https://api.openai.com/v1":
        raise ValueError("Batch API is only supported for OpenAI models!")

    llm = get_model(
        server_type='openai',
        base_url=eval_config.base_url,
        model=eval_config.judge_model,
    )
    prompt = get_prompt('judge/arena')

    # assuming everything fits in memory for simplicity
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]

        if eval_config.skip_filled and all(
            'judgement-gen-base' in data_point and 'judgement-base-gen' in data_point for data_point in data
        ):
            continue

        data_points = []

        if eval_config.use_batch_api:
            for data_point in data:
                # adding required fields for judgement prompt
                to_add = data_point.copy()
                to_add['answer_1'] = data_point['generation']
                to_add['answer_2'] = data_point['baseline_answer']
                data_points.append(to_add)
                # reversing the answers
                to_add = data_point.copy()
                to_add['answer_2'] = data_point['generation']
                to_add['answer_1'] = data_point['baseline_answer']
                data_points.append(to_add)

            request_metadata = llm.batch_generate(
                prompts=[prompt.fill(data_point) for data_point in data_points],
                tokens_to_generate=eval_config.tokens_to_generate,
            )
            # saving the request id to be able to retrieve results when they are ready
            with open(jsonl_file + '-batch-request-id', 'wt', encoding='utf-8') as fout:
                fout.write(json.dumps({'request_id': request_metadata.id}))
            LOG.info('Submitted batch evaluation request to OpenAI. Please wait for the results to be ready.')
            LOG.info('The current status and final results can be accessed through summarize_results.py')
            LOG.info('Request metadata: %s', str(request_metadata))
        else:
            output_file = jsonl_file + '-judgement'
            starting_idx = 0
            if eval_config.skip_filled:
                try:
                    with open(output_file, "rt", encoding="utf-8") as fin:
                        starting_idx = len(fin.readlines())
                except FileNotFoundError:
                    LOG.warning(f"File `{output_file}` not found, starting from scratch")
            data = data[starting_idx:]

            # saving to a tmp file to avoid corrupting original generation in case something goes wrong
            with open(output_file, "at" if eval_config.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
                for data_idx, data_point in enumerate(
                    tqdm(data, initial=starting_idx, total=len(data) + starting_idx)
                ):
                    # adding required fields for judgement prompt
                    to_add = data_point.copy()
                    to_add['answer_1'] = data_point['generation']
                    to_add['answer_2'] = data_point['baseline_answer']
                    to_add['judgement_mode'] = 'gen-base'
                    data_points.append(to_add)
                    # reversing the answers
                    to_add = data_point.copy()
                    to_add['answer_2'] = data_point['generation']
                    to_add['answer_1'] = data_point['baseline_answer']
                    to_add['judgement_mode'] = 'base-gen'
                    data_points.append(to_add)

                    if len(data_points) == eval_config.batch_size or data_idx == len(data) - 1:
                        outputs = llm.generate(
                            prompts=[prompt.fill(data_point) for data_point in data_points],
                            tokens_to_generate=eval_config.tokens_to_generate,
                        )
                        to_write = {}
                        for idx, (output, original_data_point) in enumerate(zip(outputs, data_points)):
                            to_write[f'judgement-{original_data_point["judgement_mode"]}'] = output['generation']
                            if idx % 2 != 0:
                                fout.write(json.dumps(to_write) + "\n")
                                to_write = {}
                        data_points = []

            # fusing back into original file
            with open(jsonl_file, 'wt', encoding='utf-8') as fout, open(output_file, 'rt', encoding='utf-8') as fin:
                for data_point, judgement_line in zip(data, fin):
                    data_point.update(json.loads(judgement_line))
                    fout.write(json.dumps(data_point) + "\n")

            # removing judgement file
            Path(output_file).unlink()


def eval_mtbench(cfg):
    eval_config = LlmEvaluatorConfig(**cfg.eval_config)
    assert eval_config.batch_size % 2 == 0  # required due to how everything is implemented, can fix later

    if eval_config.use_batch_api and eval_config.base_url != "https://api.openai.com/v1":
        raise ValueError("Batch API is only supported for OpenAI models!")

    llm = get_model(
        server_type='openai',
        base_url=eval_config.base_url,
        model=eval_config.judge_model,
    )
    prompt_turn1 = get_prompt('judge/mt-bench/turn1')
    prompt_turn2 = get_prompt('judge/mt-bench/turn2')
    prompt_turn1_with_ref = get_prompt('judge/mt-bench/turn1_with_ref')
    prompt_turn2_with_ref = get_prompt('judge/mt-bench/turn2_with_ref')

    # assuming everything fits in memory for simplicity
    for jsonl_file in unroll_files(cfg.input_files):
        with open(jsonl_file, 'rt', encoding='utf-8') as fin:
            data = [json.loads(line) for line in fin]

        if eval_config.skip_filled and all(
            'judgement-turn1' in data_point and 'judgement-turn2' in data_point for data_point in data
        ):
            continue

        filled_prompts = []

        if eval_config.use_batch_api:
            for data_point in data:
                # adding required fields for judgement prompt turn1
                to_add = deepcopy(data_point)
                to_add['question_1'] = data_point['turns'][0]['question']
                to_add['answer_1'] = data_point['generation'][0]
                if 'ref_answer_1' in data_point:
                    to_add['ref_answer_1'] = data_point['ref_answer_1']
                    filled_prompts.append(prompt_turn1_with_ref.fill(to_add))
                else:
                    filled_prompts.append(prompt_turn1.fill(to_add))
                # turn2 - no need to copy since we are only adding information
                to_add['question_2'] = data_point['turns'][1]['question']
                to_add['answer_2'] = data_point['generation'][1]
                if 'ref_answer_2' in data_point:
                    to_add['ref_answer_2'] = data_point['ref_answer_2']
                    filled_prompts.append(prompt_turn2_with_ref.fill(to_add))
                else:
                    filled_prompts.append(prompt_turn2.fill(to_add))

            request_metadata = llm.batch_generate(
                prompts=filled_prompts,
                tokens_to_generate=eval_config.tokens_to_generate,
            )
            # saving the request id to be able to retrieve results when they are ready
            with open(jsonl_file + '-batch-request-id', 'wt', encoding='utf-8') as fout:
                fout.write(json.dumps({'request_id': request_metadata.id}))
            LOG.info('Submitted batch evaluation request to OpenAI. Please wait for the results to be ready.')
            LOG.info('The current status and final results can be accessed through summarize_results.py')
            LOG.info('Request metadata: %s', str(request_metadata))
        else:
            output_file = jsonl_file + '-judgement'
            starting_idx = 0
            if eval_config.skip_filled:
                try:
                    with open(output_file, "rt", encoding="utf-8") as fin:
                        starting_idx = len(fin.readlines())
                except FileNotFoundError:
                    LOG.warning(f"File `{output_file}` not found, starting from scratch")
            data = data[starting_idx:]

            # saving to a tmp file to avoid corrupting original generation in case something goes wrong
            with open(output_file, "at" if eval_config.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
                for data_idx, data_point in enumerate(
                    tqdm(data, initial=starting_idx, total=len(data) + starting_idx)
                ):
                    # adding required fields for judgement prompt turn1
                    to_add = deepcopy(data_point)
                    to_add['question_1'] = data_point['turns'][0]['question']
                    to_add['answer_1'] = data_point['generation'][0]
                    if 'ref_answer_1' in data_point:
                        to_add['ref_answer_1'] = data_point['ref_answer_1']
                        filled_prompts.append(prompt_turn1_with_ref.fill(to_add))
                    else:
                        filled_prompts.append(prompt_turn1.fill(to_add))
                    # turn2 - no need to copy since we are only adding information
                    to_add['question_2'] = data_point['turns'][1]['question']
                    to_add['answer_2'] = data_point['generation'][1]
                    if 'ref_answer_2' in data_point:
                        to_add['ref_answer_2'] = data_point['ref_answer_2']
                        filled_prompts.append(prompt_turn2_with_ref.fill(to_add))
                    else:
                        filled_prompts.append(prompt_turn2.fill(to_add))

                    if len(filled_prompts) == eval_config.batch_size or data_idx == len(data) - 1:
                        outputs = llm.generate(
                            prompts=filled_prompts,
                            tokens_to_generate=eval_config.tokens_to_generate,
                        )
                        to_write = {}
                        for idx, output in enumerate(outputs):
                            turn = 'turn1' if idx % 2 == 0 else 'turn2'
                            to_write[f'judgement-{turn}'] = output['generation']
                            if idx % 2 != 0:
                                fout.write(json.dumps(to_write) + "\n")
                                to_write = {}
                        filled_prompts = []

            # fusing back into original file
            with open(jsonl_file, 'wt', encoding='utf-8') as fout, open(output_file, 'rt', encoding='utf-8') as fin:
                for data_point, judgement_line in zip(data, fin):
                    data_point.update(json.loads(judgement_line))
                    fout.write(json.dumps(data_point) + "\n")

            # removing judgement file
            Path(output_file).unlink()

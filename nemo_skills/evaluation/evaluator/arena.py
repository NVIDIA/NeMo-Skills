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

# adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/show_result.py

import inspect
import json
import logging
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from nemo_skills.inference.model import get_model
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, nested_dataclass, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


JUDGE_MODEL = 'gpt-4-1106-preview'
JUDGE_SERVER = 'openai'


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx) // 2 :] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if "baseline" in models.index:
        elo_scores += 1000 - elo_scores[models["baseline"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    kwargs = {}
    if "baseline" in inspect.signature(func_compute_elo).parameters:
        kwargs["baseline"] = "baseline"
    for _ in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), **kwargs))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.NAN for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(df, column):
    to_dict = df[["model", column]].set_index("model").to_dict()[column]
    win_rate_table = predict_win_rate(to_dict)
    return win_rate_table["baseline"].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgment(scores, WEIGHT=3):
    arena_hard_battles = pd.DataFrame()
    num_invalid = 0

    for score in scores:
        # game 1
        output = {"model_a": "candidate", "model_b": 'baseline'}

        assert len(score) == 2
        cur_score = score[0]

        weight = 1
        if cur_score == "A=B":
            output["winner"] = "tie"
        elif cur_score == "A>B":
            output["winner"] = "model_a"
        elif cur_score == "A>>B":
            output["winner"] = "model_a"
            weight = WEIGHT
        elif cur_score == "B>A":
            output["winner"] = "model_b"
        elif cur_score == "B>>A":
            output["winner"] = "model_b"
            weight = WEIGHT
        else:
            num_invalid += 1
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

        # game 2
        output = {"model_a": "candidate", "model_b": 'baseline'}

        cur_score = score[1]

        weight = 1
        if cur_score == "A=B":
            output["winner"] = "tie"
        elif cur_score == "A>B":
            output["winner"] = "model_b"
        elif cur_score == "A>>B":
            output["winner"] = "model_b"
            weight = WEIGHT
        elif cur_score == "B>A":
            output["winner"] = "model_a"
        elif cur_score == "B>>A":
            output["winner"] = "model_a"
            weight = WEIGHT
        else:
            num_invalid += 1
            weight = 0

        if weight:
            arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
    return arena_hard_battles, num_invalid


def get_aggregate_score(scores, weight=3):
    battles, num_invalid = get_battles_from_judgment(scores, weight)
    bootstrap_online_elo = compute_mle_elo(battles)

    np.random.seed(42)
    num_rounds = 100
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, num_rounds)

    stats = pd.DataFrame()
    stats["results"] = None
    stats["results"] = stats['results'].astype('object')

    for i, model in enumerate(bootstrap_online_elo.index):
        assert model in bootstrap_elo_lu.columns

        stats.at[i, "model"] = model
        stats.at[i, "score"] = bootstrap_online_elo[model]
        stats.at[i, "lower"] = np.percentile(bootstrap_elo_lu[model], 2.5)
        stats.at[i, "upper"] = np.percentile(bootstrap_elo_lu[model], 97.5)
        stats.at[i, "results"] = bootstrap_elo_lu[model].tolist()

    stats.sort_values(by="model", inplace=True)
    stats["score"] = get_win_rate_column(stats, "score").tolist()
    stats["lower"] = get_win_rate_column(stats, "lower").tolist()
    stats["upper"] = get_win_rate_column(stats, "upper").tolist()

    candidate_stats = stats[stats['model'] == 'candidate']
    interval = (
        round((candidate_stats['lower'] - candidate_stats['score']).iloc[0], 2),
        round((candidate_stats['upper'] - candidate_stats['score']).iloc[0], 2),
    )
    metrics = {
        'score': candidate_stats['score'].iloc[0],
        '95_CI': interval,
        'invalid_scores': num_invalid,
    }
    return metrics


@nested_dataclass(kw_only=True)
class LlmEvaluatorConfig:
    batch_size: int = 100  # lower if running into rate limits
    tokens_to_generate: int = 4096  # will auto-lower to max possible for NGC models
    use_batch_api: bool = True  # only supported for OpenAI models!
    base_url: str = "https://api.openai.com/v1"
    judge_model: str = JUDGE_MODEL
    # defaults to True to avoid regenerating judgements unless necessary
    skip_filled: bool = True


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

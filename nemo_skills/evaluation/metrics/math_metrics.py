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
from collections import Counter, defaultdict
from pathlib import Path

from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.inference.server.model import get_model
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


class MathMetrics(BaseMetrics):
    def setup(self, input_files):
        # checking if judgements are ready and fusing them with predictions
        # might get permission errors when running locally, since original file
        # is generated inside docker. Is there any way around that?
        for jsonl_file in unroll_files(input_files):
            if Path(jsonl_file + '-batch-request-id').exists():
                with open(jsonl_file + '-batch-request-id', 'rt', encoding='utf-8') as fin:
                    request_id = json.load(fin)['request_id']

                llm = get_model(server_type=JUDGE_SERVER, model=JUDGE_MODEL)
                metadata, outputs = llm.get_batch_results(request_id)

                if outputs is None:
                    raise RuntimeError(f"Judgements are not ready yet! Current status: {metadata}")

                with open(jsonl_file, 'rt', encoding='utf-8') as fin:
                    predictions = [json.loads(line) for line in fin]

                with open(jsonl_file, 'wt', encoding='utf-8') as fout:
                    for prediction, output in zip(predictions, outputs):
                        prediction['judgement'] = output['generation']
                        fout.write(json.dumps(prediction) + '\n')

                Path(jsonl_file + '-batch-request-id').unlink()

    def __init__(self):
        self.reset()

    def update_comb_metric(self, perf_dict, current_correct_sympy, current_correct_judge, no_answer):
        perf_dict["correct_sympy"] += int(current_correct_sympy)
        perf_dict["correct_judge"] += int(current_correct_judge)
        perf_dict["no_answer"] += int(no_answer)
        if self.has_sympy and self.has_judge:
            perf_dict["both_correct"] += int(current_correct_sympy and current_correct_judge)
            perf_dict["any_correct"] += int(current_correct_sympy or current_correct_judge)

    def update_comb_metric_averaged(self, perf_dict, current_correct_sympy, current_correct_judge, no_answer):
        perf_dict["correct_sympy"] += current_correct_sympy
        perf_dict["correct_judge"] += current_correct_judge
        perf_dict["no_answer"] += no_answer
        if self.has_sympy and self.has_judge:
            perf_dict["both_correct"] += current_correct_sympy and current_correct_judge
            perf_dict["any_correct"] += current_correct_sympy or current_correct_judge

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        # TODO: rename is_correct since it's only for sympy now?
        if 'is_correct' in predictions[0]:
            self.has_sympy = True
        if 'judgement' in predictions[0]:
            self.has_judge = True
        if 'reward_model_score' in predictions[0]:
            self.has_reward = True

        # Local vars for tracking prediction correctness
        current_correct_sympy, current_correct_judge, no_answer = False, False, False

        if len(predictions) == 1:
            self.num_decoding = 1
            # Single decoding
            if self.has_sympy:
                current_correct_sympy = predictions[0]['is_correct']
            if self.has_judge:
                current_correct_judge = is_correct_judgement(predictions[0]['judgement'])

            no_answer = predictions[0]['predicted_answer'] is None
            self.update_comb_metric(
                self.agg_mode_dict["greedy"], current_correct_sympy, current_correct_judge, no_answer
            )

            # Log any discrepancy between the two judgements
            if self.has_sympy and self.has_judge:
                if current_correct_sympy != current_correct_judge:
                    LOG.debug(
                        "Discrepancy between symbolic (%s) and LLM checkers (%s).\n"
                        "Question: %s\nPredicted answer: %s\nExpected answer: %s\nLLM reasoning: %s\n",
                        bool(current_correct_sympy),
                        bool(current_correct_judge),
                        predictions[0]['problem'],
                        predictions[0]['predicted_answer'],
                        predictions[0]['expected_answer'],
                        predictions[0]['judgement'],
                    )
        else:
            # Multiple decodings - pass/majority
            # getting metrics for all k up to len(predictions). Starting from last to make sure it's printed
            for k in range(len(predictions), 0, -1):
                # Initialize local vars for tracking prediction correctness
                current_correct_sympy, current_correct_judge, no_answer = False, False, False
                valid_answers = [
                    elem['predicted_answer'] for elem in predictions[:k] if elem['predicted_answer'] is not None
                ]
                if not len(valid_answers):
                    # Consider the answer to be incorrect if no valid answer among predictions
                    self.update_comb_metric(
                        self.agg_mode_dict[f"pass@{k}"],
                        current_correct_sympy,
                        current_correct_judge,
                        no_answer,
                    )
                    self.update_comb_metric(
                        self.agg_mode_dict[f"majority@{k}"],
                        current_correct_sympy,
                        current_correct_judge,
                        no_answer,
                    )
                    self.update_comb_metric_averaged(
                        self.agg_mode_dict[f"pass@1[{k}]"],
                        current_correct_sympy,
                        current_correct_judge,
                        no_answer,
                    )
                    if self.has_reward:
                        self.update_comb_metric(
                            self.agg_mode_dict[f"rm_best@{k}"],
                            current_correct_sympy,
                            current_correct_judge,
                            no_answer,
                        )
                        self.update_comb_metric(
                            self.agg_mode_dict[f"rm_majority@{k}"],
                            current_correct_sympy,
                            current_correct_judge,
                            no_answer,
                        )

                    return

                # Majority@K
                # TODO: currently majority does not take into account equivalent answers written in a different way
                # Reinitialize local vars for tracking prediction correctness
                current_correct_sympy, current_correct_judge, no_answer = False, False, False

                def get_majority_result(predictions, result_extractor):
                    valid_answers_and_results = [
                        (elem['predicted_answer'], result_extractor(elem))
                        for elem in predictions
                        if elem['predicted_answer'] is not None
                    ]

                    majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                    return majority_result[1], False

                if self.has_sympy:
                    current_correct_sympy, no_answer = get_majority_result(
                        predictions[:k], lambda elem: elem['is_correct']
                    )

                if self.has_judge:
                    current_correct_judge, no_answer = get_majority_result(
                        predictions[:k], lambda elem: is_correct_judgement(elem['judgement'])
                    )

                self.update_comb_metric(
                    self.agg_mode_dict[f"majority@{k}"],
                    current_correct_sympy,
                    current_correct_judge,
                    no_answer,
                )

                # Reward Models
                if self.has_reward:
                    # Reinitialize local vars for tracking prediction correctness
                    current_correct_sympy, current_correct_judge, no_answer = False, False, False

                    def get_reward_best_result(predictions, result_extractor):
                        valid_answers_and_results = [
                            (elem['predicted_answer'], result_extractor(elem), elem['reward_model_score'])
                            for elem in predictions
                            if elem['predicted_answer'] is not None
                        ]

                        # Answer is the top-scoring reward
                        current_correct = sorted(valid_answers_and_results, key=lambda x: x[2], reverse=True)[0][1]
                        return current_correct, False

                    if self.has_sympy:
                        current_correct_sympy, no_answer = get_reward_best_result(
                            predictions[:k], lambda elem: elem['is_correct']
                        )

                    if self.has_judge:
                        current_correct_judge, no_answer = get_reward_best_result(
                            predictions[:k], lambda elem: is_correct_judgement(elem['judgement'])
                        )

                    self.update_comb_metric(
                        self.agg_mode_dict[f"rm_best@{k}"],
                        current_correct_sympy,
                        current_correct_judge,
                        no_answer,
                    )

                    # Reinitialize local vars for tracking prediction correctness
                    current_correct_sympy, current_correct_judge, no_answer = False, False, False

                    def get_majority_reward_result(predictions, result_extractor):
                        valid_answers_and_results = [
                            (elem['predicted_answer'], result_extractor(elem), elem['reward_model_score'])
                            for elem in predictions
                            if elem['predicted_answer'] is not None
                        ]

                        answer_to_score_dict = defaultdict(float)
                        answer_to_correctness_dict = {}
                        for predicted_answer, is_correct, reward_score in valid_answers_and_results:
                            answer_to_score_dict[predicted_answer] += reward_score
                            answer_to_correctness_dict[predicted_answer] = is_correct

                        top_cum_reward_answer = sorted(
                            list(answer_to_score_dict.items()), key=lambda x: x[1], reverse=True
                        )[0][0]
                        current_correct = answer_to_correctness_dict[top_cum_reward_answer]
                        return current_correct, False

                    if self.has_sympy:
                        current_correct_sympy, no_answer = get_majority_reward_result(
                            predictions[:k], lambda elem: elem['is_correct']
                        )

                    if self.has_judge:
                        current_correct_judge, no_answer = get_majority_reward_result(
                            predictions[:k], lambda elem: is_correct_judgement(elem['judgement'])
                        )

                    self.update_comb_metric(
                        self.agg_mode_dict[f"rm_majority@{k}"],
                        current_correct_sympy,
                        current_correct_judge,
                        no_answer,
                    )

                # Pass@K
                # Reinitialize local vars for tracking prediction correctness
                current_correct_sympy, current_correct_judge, no_answer = False, False, False
                if self.has_sympy:
                    current_correct_sympy = any([elem['is_correct'] for elem in predictions[:k]])
                if self.has_judge:
                    current_correct_judge = any([is_correct_judgement(elem['judgement']) for elem in predictions[:k]])
                if all([elem['predicted_answer'] is None for elem in predictions[:k]]):
                    no_answer = True

                self.update_comb_metric(
                    self.agg_mode_dict[f"pass@{k}"], current_correct_sympy, current_correct_judge, no_answer
                )

                # Pass@1[k] - mean of pass@1 across all generations
                current_correct_sympy, current_correct_judge, no_answer = False, False, False
                if self.has_sympy:
                    current_correct_sympy = sum([elem['is_correct'] for elem in predictions[:k]]) / k
                if self.has_judge:
                    current_correct_judge = (
                        sum([is_correct_judgement(elem['judgement']) for elem in predictions[:k]]) / k
                    )
                if all([elem['predicted_answer'] is None for elem in predictions[:k]]):
                    no_answer = True

                self.update_comb_metric_averaged(
                    self.agg_mode_dict[f"pass@1[{k}]"], current_correct_sympy, current_correct_judge, no_answer
                )

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {"num_entries": self.total}
            if self.has_sympy:
                metrics_dict[agg_mode]["symbolic_correct"] = (agg_metric_dict["correct_sympy"] / self.total) * 100.0
            if self.has_judge:
                metrics_dict[agg_mode]["judge_correct"] = (agg_metric_dict["correct_judge"] / self.total) * 100.0
            if self.has_sympy and self.has_judge:
                metrics_dict[agg_mode]["both_correct"] = (agg_metric_dict["both_correct"] / self.total) * 100.0
                metrics_dict[agg_mode]["any_correct"] = (agg_metric_dict["any_correct"] / self.total) * 100.0

            metrics_dict[agg_mode]["no_answer"] = (agg_metric_dict["no_answer"] / self.total) * 100.0

        return metrics_dict

    def reset(self):
        self.has_sympy = False
        self.has_judge = False
        self.has_reward = False
        self.total = 0
        self.agg_mode_dict = defaultdict(lambda: defaultdict(int))

    def max_aggregations_to_print(self):
        """We will log all majority/rm/pass/pass@1[k] up to k, but only report the kth one."""
        # majority + pass + 2xRM + pass@1[k]
        return 1 + 1 + 2 * self.has_reward + 1

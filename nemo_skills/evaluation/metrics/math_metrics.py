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
from collections import defaultdict
from pathlib import Path

from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
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

    def _get_correctness_dict(self, prediction: dict) -> dict[bool]:
        correctness_dict = {}
        if 'is_correct' in prediction:
            has_sympy = True
            correctness_dict["symbolic_correct"] = prediction['is_correct']
        if 'judgement' in prediction:
            has_judge = True
            correctness_dict["judge_correct"] = is_correct_judgement(prediction['judgement'])
        if has_sympy and has_judge:
            correctness_dict["both_correct"] = (
                correctness_dict["symbolic_correct"] and correctness_dict["judge_correct"]
            )
            correctness_dict["any_correct"] = correctness_dict["symbolic_correct"] or correctness_dict["judge_correct"]

        return correctness_dict

    def _compute_reward_at_k(self, predictions: list[dict]):
        agg_mode_dict = agg_mode_dict or self.agg_mode_dict

        correctness_dicts = [self._get_correctness_dict(pred) for pred in predictions]

        for k in range(1, len(predictions) + 1):
            for check_correctness_method in correctness_dicts[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (elem['predicted_answer'], correctness_dict[check_correctness_method], elem['reward_model_score'])
                    for elem, correctness_dict in zip(predictions[:k], correctness_dicts[:k])
                    if elem['predicted_answer'] is not None
                ]

                # If no valid answers, it's incorrect
                if not valid_answers_and_results:
                    is_correct = False
                    agg_mode_dict[f"rm_best@{k}"]["no_answer"] += 1
                    agg_mode_dict[f"rm_majority@{k}"]["no_answer"] += 1
                else:
                    is_correct_best = sorted(valid_answers_and_results, key=lambda x: x[2], reverse=True)[0][1]
                    agg_mode_dict[f"rm_best@{k}"][check_correctness_method] += is_correct_best

                    answer_to_score_dict = defaultdict(float)
                    answer_to_correctness_dict = {}
                    for predicted_answer, is_correct, reward_score in valid_answers_and_results:
                        answer_to_score_dict[predicted_answer] += reward_score
                        answer_to_correctness_dict[predicted_answer] = is_correct

                    top_cum_reward_answer = sorted(
                        list(answer_to_score_dict.items()), key=lambda x: x[1], reverse=True
                    )[0][0]
                    is_correct_majority = answer_to_correctness_dict[top_cum_reward_answer]
                    agg_mode_dict[f"rm_majority@{k}"][check_correctness_method] += is_correct_majority

    def _update_metrics_for_pass(
        self,
        agg_mode_dict: dict,
        k: int,
        predictions: list[dict],
    ):
        super()._update_metrics_for_pass(
            agg_mode_dict,
            k,
            predictions,
        )
        no_answer_list = [pred['predicted_answer'] is None for pred in predictions[:k]]
        agg_mode_dict[f"pass@{k}"]["no_answer"] += all(no_answer_list)
        agg_mode_dict[f"pass@1[{k}]"]["no_answer"] += sum(no_answer_list) / k

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        predicted_answers = [pred['predicted_answer'] for pred in predictions]
        self._compute_pass_at_k(predictions=predictions)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        if 'reward_model_score' in predictions[0]:
            self._compute_reward_at_k(predictions=predictions)

        # Log discrepancies between the two judgements
        for prediction in predictions:
            correctness_dict = self._get_correctness_dict(prediction)
            if "symbolic_correct" not in correctness_dict or "judge_correct" not in correctness_dict:
                continue
            if correctness_dict["symbolic_correct"] != correctness_dict["judge_correct"]:
                LOG.debug(
                    "Discrepancy between symbolic (%s) and LLM checkers (%s).\n"
                    "Question: %s\nPredicted answer: %s\nExpected answer: %s\nLLM reasoning: %s\n",
                    correctness_dict["symbolic_correct"],
                    correctness_dict["judge_correct"],
                    prediction['problem'],
                    prediction['predicted_answer'],
                    prediction['expected_answer'],
                    prediction['judgement'],
                )

    def aggregations_to_print(self):
        """We will log all majority/rm/pass/pass@1[k] up to k, but only report the kth one."""
        return [
            f'pass@1[{self.max_k}]',
            f'majority@{self.max_k}',
            f'pass@{self.max_k}',
            f'rm_best@{self.max_k}',
            f'rm_majority@{self.max_k}',
        ]

    def metrics_to_print(self):
        return {
            'num_entries': as_int,
            'avg_tokens': as_int,
            'judge_correct': as_percentage,
            'symbolic_correct': as_percentage,
            'no_answer': as_percentage,
        }

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

import abc
import json
import logging
import re
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from nemo_skills.utils import unroll_files

LOG = logging.getLogger(__file__)


class BaseEval(abc.ABC):
    @abc.abstractmethod
    def fill_up_missing(self):
        pass

    @abc.abstractmethod
    def is_incomplete(self, elem):
        pass

    @abc.abstractmethod
    def update(self, predictions, aggregation_mode):
        pass

    @abc.abstractmethod
    def get_metrics(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def setup(self, prediction_jsonl_files):
        pass


class MathEval(BaseEval):
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'predicted_answer': None, 'is_correct': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'predicted_answer' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "majority", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            if all([elem['predicted_answer'] is None for elem in predictions]):
                self.total_no_answer += 1
        elif aggregation_mode == "majority":
            # TODO: currently majority does not take into account equivalent answers written in a different way
            valid_answers_and_results = [
                (elem['predicted_answer'], elem['is_correct'])
                for elem in predictions
                if elem['predicted_answer'] is not None
            ]
            if len(valid_answers_and_results) == 0:
                self.total_no_answer += 1
            else:
                majority_result = Counter(valid_answers_and_results).most_common(1)[0][0]
                self.total_correct += majority_result[1]
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_no_answer += predictions[0]['predicted_answer'] is None
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "correct_answer": self.total_correct / self.total * 100.0,
            "wrong_answer": (self.total - self.total_correct - self.total_no_answer) / self.total * 100.0,
            "no_answer": self.total_no_answer / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_no_answer = 0
        self.total = 0


class CodeEval(BaseEval):
    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {'is_correct': False, 'is_correct-plus': False}

    def is_incomplete(self, elem):
        return 'is_correct' not in elem or 'is_correct-plus' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        if aggregation_mode == "best":
            self.total_correct += any([elem['is_correct'] for elem in predictions])
            self.total_correct_plus += any([elem['is_correct-plus'] for elem in predictions])
        elif aggregation_mode == "first":
            self.total_correct += predictions[0]['is_correct']
            self.total_correct_plus += predictions[0]['is_correct-plus']
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        return {
            "num_entries": self.total,
            "passing_base_tests": self.total_correct / self.total * 100.0,
            "passing_plus_tests": self.total_correct_plus / self.total * 100.0,
        }

    def reset(self):
        self.total_correct = 0
        self.total_correct_plus = 0
        self.total = 0


class IFEval(BaseEval):
    # loosely adapted from
    # https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_main.py

    required_keys = ['follow_instruction_list', 'instruction_id_list']

    def __init__(self):
        self.reset()

    def fill_up_missing(self):
        return {
            'loose_eval': {key: [] for key in self.required_keys},
            'strict_eval': {key: [] for key in self.required_keys},
        }

    def is_incomplete(self, elem):
        incomplete = 'loose_eval' not in elem or 'strict_eval' not in elem
        if incomplete:
            return True

        if any([key not in elem['loose_eval'] for key in self.required_keys]):
            return True

        if any([key not in elem['strict_eval'] for key in self.required_keys]):
            return True

        return False

    def _update_single_stat(self, stats_dict, elems):
        """Will update using the pass@k strategy (just pass a single-element list to get greedy)."""
        # has to be the same across all elements as they are solutions for the same question
        instruction_id_list = elems[0]['instruction_id_list']
        # computing "pass@k" score
        follow_instruction_list = elems[0]['follow_instruction_list']
        for elem in elems:
            follow_instruction_list = [
                follow_instruction_list[i] or elem['follow_instruction_list'][i]
                for i in range(len(follow_instruction_list))
            ]

        stats_dict['prompt']['total'] += 1
        if all(follow_instruction_list):
            stats_dict['prompt']['correct'] += 1

        stats_dict['instruction']['total'] += len(instruction_id_list)
        stats_dict['instruction']['correct'] += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            instruction_id = instruction_id.split(":")[0]
            stats_dict['tier0']['total'][instruction_id] += 1
            if followed_or_not:
                stats_dict['tier0']['correct'][instruction_id] += 1

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        if aggregation_mode == "best":
            self._update_single_stat(self.strict_stats, [pred['strict_eval'] for pred in predictions])
            self._update_single_stat(self.loose_stats, [pred['loose_eval'] for pred in predictions])
        elif aggregation_mode == "first":
            self._update_single_stat(self.strict_stats, [predictions[0]['strict_eval']])
            self._update_single_stat(self.loose_stats, [predictions[0]['loose_eval']])
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        prompt_total = self.strict_stats['prompt']['total']
        inst_total = self.strict_stats['instruction']['total']
        return {
            "num_prompts": prompt_total,
            "num_instructions": inst_total,
            "prompt_strict_accuracy": self.strict_stats['prompt']['correct'] / prompt_total * 100.0,
            "instruction_strict_accuracy": self.strict_stats['instruction']['correct'] / inst_total * 100.0,
            "prompt_loose_accuracy": self.loose_stats['prompt']['correct'] / prompt_total * 100.0,
            "instruction_loose_accuracy": self.loose_stats['instruction']['correct'] / inst_total * 100.0,
        }

    def reset(self):
        # the original code also has a deeper breakdown into tier1 scores,
        # but that's probably too much for us to track at this stage
        self.strict_stats = {
            "prompt": {"total": 0, "correct": 0},
            "instruction": {"total": 0, "correct": 0},
            "tier0": {"total": defaultdict(int), "correct": defaultdict(int)},
        }
        self.loose_stats = {
            "prompt": {"total": 0, "correct": 0},
            "instruction": {"total": 0, "correct": 0},
            "tier0": {"total": defaultdict(int), "correct": defaultdict(int)},
        }


class ArenaEval(BaseEval):
    def __init__(self):
        self.reset()

    def setup(self, prediction_jsonl_files):
        # checking if judgements are ready and fusing them with predictions
        # might get permission errors when running locally, since original file
        # is generated inside docker. Is there any way around that?
        for jsonl_file in unroll_files(prediction_jsonl_files):
            if Path(jsonl_file + '-batch-request-id').exists():
                with open(jsonl_file + '-batch-request-id', 'rt', encoding='utf-8') as fin:
                    request_id = json.load(fin)['request_id']
                from nemo_skills.inference.server.model import get_model

                llm = get_model(server_type='openai', model='gpt-4-1106-preview')
                metadata, outputs = llm.get_batch_results(request_id)

                if outputs is None:
                    raise RuntimeError(f"Judgements are not ready yet! Current status: {metadata}")

                with open(jsonl_file, 'rt', encoding='utf-8') as fin:
                    predictions = [json.loads(line) for line in fin]

                with open(jsonl_file, 'wt', encoding='utf-8') as fout:
                    for idx, output in enumerate(outputs):
                        if idx % 2 == 0:
                            prediction = predictions[idx // 2]
                            prediction['judgement-gen-base'] = output['generation']
                        else:
                            prediction['judgement-base-gen'] = output['generation']
                            fout.write(json.dumps(prediction) + '\n')

                Path(jsonl_file + '-batch-request-id').unlink()

    def _get_judge_score(self, judgment):
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        pattern = re.compile('\[\[([AB<>=]+)\]\]')
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def fill_up_missing(self):
        return {'judgement-gen-base': '', 'judgement-base-gen': '', 'generation': ''}

    def is_incomplete(self, elem):
        return 'judgement-gen-base' not in elem or 'judgement-base-gen' not in elem or 'generation' not in elem

    def update(self, predictions, aggregation_mode):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
            aggregation_mode (str): "best", "first", etc. Might vary by benchmark.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        self.scores.append([])
        if aggregation_mode == "best":
            judge_scores = [self._get_judge_score(elem['judgement-gen-base']) for elem in predictions]
            # adding the best score out of all the generations
            possible_scores = ['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']
            for possible_score in possible_scores:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += len(predictions[best_id]['generation'])
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score

            judge_scores = [self._get_judge_score(elem['judgement-base-gen']) for elem in predictions]
            # second score is grading swapped answers, so we iterate from the end
            for possible_score in possible_scores[::-1]:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += len(predictions[best_id]['generation'])
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score
        elif aggregation_mode == "first":
            self.lengths += len(predictions[0]['generation'])
            self.scores[-1] = [
                self._get_judge_score(predictions[0]['judgement-gen-base']),
                self._get_judge_score(predictions[0]['judgement-base-gen']),
            ]
        else:
            raise ValueError(f"Unsupported mode {aggregation_mode}")

    def get_metrics(self):
        # run the score aggregation using arena-hard logic
        # currently needs sklearn, which is not ideal, but let's just error-out if it's not installed
        # it's also not going to work on clusters unless there is python 3.10 and all packages are installed
        # so currently need to be done inside the container or with some custom setup
        try:
            from nemo_skills.evaluation.arena_utils import get_aggregate_score
        except ImportError:
            raise ImportError(
                "Please install scikit-learn to be able to bootstrap battle results and calculate accurate elo scores"
            )

        metrics = {'num_entries': self.total}
        metrics.update(get_aggregate_score(self.scores))
        metrics['avg_response_length'] = self.lengths / self.total
        return metrics

    def reset(self):
        self.scores = []  # list of lists
        self.lengths = 0
        self.total = 0


def read_predictions(predictions, evaluator, allow_incomplete=False):
    data = []
    for prediction in predictions:
        if not prediction:  # could have missing predictions
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        prediction_dict = json.loads(prediction)
        if not prediction_dict:
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        if evaluator.is_incomplete(prediction_dict):
            if not allow_incomplete:
                raise RuntimeError("Some data is missing!")
            data.append(evaluator.fill_up_missing())
            continue
        data.append(prediction_dict)

    return data


def compute_metrics(
    prediction_jsonl_files,
    evaluator,
    allow_incomplete=False,
    max_samples=-1,
    aggregation_mode='first',
):
    evaluator.reset()
    evaluator.setup(prediction_jsonl_files)

    file_handles = [open(file, "rt", encoding="utf-8") for file in unroll_files(prediction_jsonl_files)]
    for idx, predictions in enumerate(zip_longest(*file_handles)):
        if idx == max_samples:
            break
        data = read_predictions(predictions, evaluator, allow_incomplete)
        evaluator.update(data, aggregation_mode)

    for file_handle in file_handles:
        file_handle.close()

    return evaluator.get_metrics()

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
import re
from collections import defaultdict
from pathlib import Path

from nemo_skills.evaluation.arena_utils import get_aggregate_score
from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.inference.server.model import get_model
from nemo_skills.utils import unroll_files


class ArenaMetrics(BaseMetrics):

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
        pattern = re.compile(r'\[\[([AB<>=]+)\]\]')
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def get_prediction_results(self, prediction):
        return {
            'lengths': len(prediction['generation']),
            'scores': [
                [
                    self._get_judge_score(prediction['judgement-gen-base']),
                    self._get_judge_score(prediction['judgement-base-gen']),
                ]
            ],
        }

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)

        prediction_results = [self.get_prediction_results(pred) for pred in predictions]

        if len(predictions) == 1:
            # Single prediction
            self.get_pass_at_k(self.agg_mode_dict, prediction_results=prediction_results)
        else:
            k = len(predictions)
            self.agg_mode_dict[f"pass@{k}"]['scores'].append([])
            possible_scores = ['A>>B', 'A>B', 'A=B', 'B>A', 'B>>A']

            for possible_score in possible_scores:
                for i in range(2):
                    judge_scores = [elem['scores'][i] for elem in prediction_results]
                    if any([score == possible_score for score in judge_scores]):
                        self.agg_mode_dict[f"pass@{k}"]['scores'][-1].append(possible_score)
                        best_id = judge_scores.index(possible_score)
                        self.agg_mode_dict[f"pass@{k}"]['lengths'] += prediction_results[best_id]['lengths']
                        break
                else:
                    self.agg_mode_dict[f"pass@{k}"]['scores'][-1].append(None)

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode, agg_metric_dict in self.agg_mode_dict.items():
            metrics_dict[agg_mode] = {'num_entries': self.total}
            metrics_dict[agg_mode].update(get_aggregate_score(agg_metric_dict['scores']))
            metrics_dict[agg_mode]['avg_response_length'] = agg_metric_dict['lengths'] / self.total
        return metrics_dict

    def reset(self):
        super().reset()
        self.agg_mode_dict = defaultdict(lambda: {'scores': [], 'lengths': 0})

    def aggregations_to_print(self):
        """We will log all pass up to k, but only report the kth one."""
        return [f'pass@{self.max_k}']

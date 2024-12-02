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
from pathlib import Path

from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.inference.server.model import get_model
from nemo_skills.utils import unroll_files


class ArenaMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

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
        pattern = re.compile('\[\[([AB<>=]+)\]\]')
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1
        self.scores.append([])
        if len(predictions) > 1:
            self.agg_mode = f"pass@{len(predictions)}"

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
        else:
            # Single prediction
            self.agg_mode = "greedy"

            self.lengths += len(predictions[0]['generation'])
            self.scores[-1] = [
                self._get_judge_score(predictions[0]['judgement-gen-base']),
                self._get_judge_score(predictions[0]['judgement-base-gen']),
            ]

    def get_metrics(self):
        from nemo_skills.evaluation.arena_utils import get_aggregate_score

        metrics = {'num_entries': self.total}
        metrics.update(get_aggregate_score(self.scores))
        metrics['avg_response_length'] = self.lengths / self.total
        return {self.agg_mode: metrics}

    def reset(self):
        self.scores = []  # list of lists
        self.lengths = 0
        self.total = 0
        # Set automatically
        self.agg_mode = "greedy"

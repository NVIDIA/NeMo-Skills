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

from nemo_skills.evaluation.constants import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics
from nemo_skills.inference.server.model import get_model
from nemo_skills.utils import unroll_files


class MtBenchMetrics(BaseMetrics):
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

                llm = get_model(server_type=JUDGE_MODEL, model=JUDGE_SERVER)
                metadata, outputs = llm.get_batch_results(request_id)

                if outputs is None:
                    raise RuntimeError(f"Judgements are not ready yet! Current status: {metadata}")

                with open(jsonl_file, 'rt', encoding='utf-8') as fin:
                    predictions = [json.loads(line) for line in fin]

                with open(jsonl_file, 'wt', encoding='utf-8') as fout:
                    for idx, output in enumerate(outputs):
                        if idx % 2 == 0:
                            prediction = predictions[idx // 2]
                            prediction['judgement-turn1'] = output['generation']
                        else:
                            prediction['judgement-turn2'] = output['generation']
                            fout.write(json.dumps(prediction) + '\n')

                Path(jsonl_file + '-batch-request-id').unlink()

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        self.total += 1

        if len(predictions) > 1:
            # TODO: might all have missing judgement?
            # If multiple predictions, set it to "best" aggregation mode
            self.agg_mode = "best"

            rating1 = max(
                int(re.search(r'Rating: \[\[(\d+)\]\]', elem['judgement-turn1']).group(1))
                for elem in predictions
                if re.search(r'Rating: \[\[(\d+)\]\]', elem['judgement-turn1'])
            )
            rating2 = max(
                int(re.search(r'Rating: \[\[(\d+)\]\]', elem['judgement-turn2']).group(1))
                for elem in predictions
                if re.search(r'Rating: \[\[(\d+)\]\]', elem['judgement-turn2'])
            )
            category = predictions[0]['category']
            self.scores[category].append((rating1, rating2))
        else:
            # If single prediction, set it to greedy aggregation mode
            self.agg_mode = "greedy"

            rating1_match = re.search(r'Rating: \[\[(\d+)\]\]', predictions[0]['judgement-turn1'])
            rating1 = int(rating1_match.group(1)) if rating1_match else None
            rating2_match = re.search(r'Rating: \[\[(\d+)\]\]', predictions[0]['judgement-turn2'])
            rating2 = int(rating2_match.group(1)) if rating2_match else None
            category = predictions[0]['category']
            self.scores[category].append((rating1, rating2))

    def get_metrics(self):
        metrics = {'num_entries': self.total}

        # Calculate average scores across all categories for each turn
        all_ratings1 = [r1 for scores in self.scores.values() for r1, _ in scores if r1 is not None]
        all_ratings2 = [r2 for scores in self.scores.values() for _, r2 in scores if r2 is not None]

        all_ratings = all_ratings1 + all_ratings2
        if all_ratings:
            metrics['average'] = sum(all_ratings) / len(all_ratings)

        if all_ratings1:
            metrics['average_turn1'] = sum(all_ratings1) / len(all_ratings1)
        if all_ratings2:
            metrics['average_turn2'] = sum(all_ratings2) / len(all_ratings2)

        none_count_turn1 = 0
        none_count_turn2 = 0
        for category, scores in self.scores.items():
            if not scores:
                continue
            ratings1 = [r1 for r1, _ in scores if r1 is not None]
            ratings2 = [r2 for _, r2 in scores if r2 is not None]
            none_count_turn1 += sum(1 for r1, _ in scores if r1 is None)
            none_count_turn2 += sum(1 for _, r2 in scores if r2 is None)
            metrics[f'{category}_turn1'] = sum(ratings1) / len(ratings1)
            metrics[f'{category}_turn2'] = sum(ratings2) / len(ratings2)
        metrics["missing_rating_turn1"] = none_count_turn1
        metrics["missing_rating_turn2"] = none_count_turn2
        print("Please see metrics.json for MT-bench per-category breakdown")
        return {self.agg_mode: metrics}

    def reset(self):
        self.scores = defaultdict(list)
        self.total = 0
        self.agg_mode = "greedy"

    def max_metrics_to_print(self):
        """We are only printing the averages, but all other metrics can still be found in metrics.json"""
        return 4

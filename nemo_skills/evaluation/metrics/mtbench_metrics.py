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

from nemo_skills.evaluation.evaluator.arena import JUDGE_MODEL, JUDGE_SERVER
from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int
from nemo_skills.inference.model import get_model
from nemo_skills.utils import unroll_files


class MtBenchMetrics(BaseMetrics):
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
        super().update(predictions)

        self.agg_mode = f"pass@{len(predictions)}"
        if len(predictions) > 1:
            # TODO: might all have missing judgement?
            # If multiple predictions, set it to "best" aggregation mode

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
            rating1_match = re.search(r'Rating: \[\[(\d+)\]\]', predictions[0]['judgement-turn1'])
            rating1 = int(rating1_match.group(1)) if rating1_match else None
            rating2_match = re.search(r'Rating: \[\[(\d+)\]\]', predictions[0]['judgement-turn2'])
            rating2 = int(rating2_match.group(1)) if rating2_match else None
            category = predictions[0]['category']
            self.scores[category].append((rating1, rating2))

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction['judgement-turn1'] = 'Rating: [[1]]'
        prediction['judgement-turn2'] = 'Rating: [[1]]'
        return prediction

    def get_metrics(self):
        metrics = {'num_entries': self.total}
        if self.avg_tokens > 0:
            metrics['avg_tokens'] = int(self.avg_tokens / self.total)
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
        metrics_dict = {self.agg_mode: metrics}
        self.update_common_metrics(metrics_dict[self.agg_mode])
        return metrics_dict

    def reset(self):
        super().reset()
        self.scores = defaultdict(list)
        self.agg_mode = "pass@1"

    def metrics_to_print(self):
        """We are only printing the averages, but all other metrics can still be found in metrics.json"""
        return {
            'num_entries': as_int,
            'avg_tokens': as_int,
            'gen_seconds': as_int,
            'average': as_float,
            'average_turn1': as_float,
            'average_turn2': as_float,
        }

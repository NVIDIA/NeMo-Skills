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

from nemo_skills.evaluation.metrics.base import BaseMetrics


# Base class for metrics computation
class LoftMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def update(self, predictions):
        
        self.total += 1
        # find current stored metric
        if 'recall_at_1' in predictions[0]:
            self.one_sample_metric = predictions[0]['recall_at_1']
        elif 'mrecall_at_2' in predictions[0]:
            self.one_sample_metric = predictions[0]['mrecall_at_2']
        elif 'mrecall_at_3' in predictions[0]:
            self.one_sample_metric = predictions[0]['mrecall_at_3']
        elif 'mrecall_at_5' in predictions[0]:    
            self.one_sample_metric = predictions[0]['mrecall_at_5']
        else:
            raise ValueError(f"Unsupported metric: {predictions[0]}")
            
        self.recall_at_k += predictions[0][f'recall_at_{self.k}']

    def get_metrics(self):
        metrics = {"num_entries": self.total}
        metrics[f"macro_recall_at_{self.k}"] = self.one_sample_metric / self.total * 100.
        
        # metrics["null_error"] = self.timeout_error / self.total * 100.0
        return {self.agg_mode: metrics}

    def reset(self):
        self.total = 0 
        self.one_sample_metric = 0
        self.agg_mode = "greedy"
        
            
    # def setup(self, input_files):
    #     pass

    # def max_metrics_to_print(self):
    #     """No limit by default."""
    #     return None

    # def max_aggregations_to_print(self):
    #     """No limit by default."""
    #     return None
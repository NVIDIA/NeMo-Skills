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


class BFCLMetrics(BaseMetrics):
    """Metrics for BFCL (Berkeley Function Calling Leaderboard) evaluation.
    
    This class handles metrics computation for function calling tasks,
    including AST evaluation, execution results, and relevance detection.
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract BFCL scores from prediction results.
        
        Args:
            prediction: Dictionary containing BFCL evaluation results
            
        Returns:
            Dictionary with score keys and their values
        """
        scores = {}
        
        # Extract scores from different test categories
        for key, value in prediction.items():
            if key.startswith('bfcl_') and key.endswith('_result'):
                category = key.replace('bfcl_', '').replace('_result', '')
                
                if isinstance(value, dict):
                    # Extract individual metric scores
                    for metric_name, metric_value in value.items():
                        if isinstance(metric_value, (int, float, bool)):
                            scores[f'{category}_{metric_name}'] = metric_value
                
                # Also add the raw result for debugging
                scores[f'{category}_raw'] = value
        
        # Standard correctness score for pass@k computation
        # Use overall accuracy or fall back to is_correct
        if 'is_correct' in prediction:
            scores['is_correct'] = prediction['is_correct']
        elif any('overall_accuracy' in str(v) for v in prediction.values() if isinstance(v, dict)):
            # Extract overall accuracy from any category that has it
            for value in prediction.values():
                if isinstance(value, dict) and 'overall_accuracy' in value:
                    scores['is_correct'] = value['overall_accuracy'] > 0
                    break
        else:
            scores['is_correct'] = False
            
        return scores
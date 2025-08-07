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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class AALCRMetrics(BaseMetrics):
    """Metrics for AA-LCR (Artificial Analysis Long Context Reading) dataset.
    
    This dataset uses an LLM-based equality checker (Officially, the non-reasoning Qwen3 235B A22B 2507)
    to evaluate whether candidate answers are consistent with official answers.
    """

    def __init__(self):
        super().__init__()
        # Track metrics by document category for detailed analysis
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)  # Track input token statistics

    def reset(self):
        super().reset()
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)

    @staticmethod
    def is_aalcr_correct(judgement: str) -> bool:
        """Check if AA-LCR judgement indicates correct answer.
        
        AA-LCR uses 'CORRECT' or 'INCORRECT' format instead of 'Judgement: Yes/No'.
        """
        if judgement is None:
            return False
        judgement = judgement.strip().upper()
        return judgement == 'CORRECT' or judgement.startswith('CORRECT')

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness scores for a prediction using LLM-based equality checker."""
        correctness_dict = {}
        
        # Primary evaluation method: LLM-based equality checker
        if 'judgement' in prediction:
            correctness_dict["judge_correct"] = self.is_aalcr_correct(prediction['judgement'])
        
        return correctness_dict

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a prediction that evaluates as incorrect."""
        prediction = prediction.copy()
        prediction['judgement'] = 'INCORRECT'
        prediction['predicted_answer'] = None
        return prediction

    def _update_category_metrics(self, prediction: dict, score_dict: dict):
        """Update per-category metrics if document_category is available."""
        category = prediction.get('document_category', 'unknown')
        self.category_totals[category] += 1
        
        for score_method, is_correct in score_dict.items():
            if is_correct:
                self.category_metrics[category][score_method] += 1
    
    def _update_token_stats(self, prediction: dict):
        """Track input token statistics by category."""
        category = prediction.get('document_category', 'unknown')
        input_tokens = prediction.get('input_tokens')
        if input_tokens is not None:
            self.token_stats[category].append(int(input_tokens))

    def update(self, predictions):
        """Update the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'judgement' from LLM equality checker.
        """
        super().update(predictions)
        
        # Update category metrics and token stats using the first prediction 
        # (they should all have the same expected answer and metadata)
        if predictions:
            score_dict = self._get_score_dict(predictions[0])
            self._update_category_metrics(predictions[0], score_dict)
            self._update_token_stats(predictions[0])
        
        # Compute standard pass@k and majority@k metrics
        predicted_answers = [pred.get('predicted_answer') for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get all computed metrics including per-category breakdown and token statistics."""
        metrics_dict = super().get_metrics()
        
        # Add per-category metrics
        if self.category_totals:
            category_results = {}
            for category, total in self.category_totals.items():
                category_results[category] = {}
                for score_method, correct_count in self.category_metrics[category].items():
                    category_results[category][score_method] = 100.0 * correct_count / total
                category_results[category]['total_samples'] = total
                
                # Add token statistics
                if category in self.token_stats and self.token_stats[category]:
                    tokens = self.token_stats[category]
                    category_results[category]['avg_input_tokens'] = int(sum(tokens) / len(tokens))
                    category_results[category]['max_input_tokens'] = max(tokens)
                    category_results[category]['min_input_tokens'] = min(tokens)
            
            # Add category breakdown to the main evaluation mode
            for eval_mode in metrics_dict:
                if eval_mode == f'pass@1[avg-of-{self.max_k}]':  # Only add to the main evaluation mode
                    metrics_dict[eval_mode]['category_breakdown'] = category_results
        
        return metrics_dict

    def evaluations_to_print(self):
        """Return which evaluations should be printed in the summary."""
        if self.max_k > 1:
            return [f'pass@1[avg-of-{self.max_k}]', f'majority@{self.max_k}', f'pass@{self.max_k}']
        else:
            return ['pass@1']
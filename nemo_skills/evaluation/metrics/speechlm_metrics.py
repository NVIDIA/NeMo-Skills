# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SpeechLMMetrics(BaseMetrics):
    """Metrics class for speech/audio language model evaluation tasks like MMAU-Pro."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k
    
    def _extract_judge_result(self, judgement_text: str) -> bool:
        """Extract judge result from judgement text."""
        import re
        
        if re.search(r'\byes\b', judgement_text, re.IGNORECASE):
            return True
        elif re.search(r'\bno\b', judgement_text, re.IGNORECASE):
            return False
        else:
            return False

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction."""
        score_dict = {}
        
        category = prediction.get("category", "unknown")
        
        if "judgement" in prediction and category == "open":
            judge_result = self._extract_judge_result(prediction["judgement"])
            score_dict["judge_correct"] = judge_result

        if category == "open" and "judge_correct" in score_dict:
            score_dict["correct"] = score_dict["judge_correct"]
        elif "is_correct" in prediction:
            score_dict["correct"] = prediction["is_correct"]
        else:
            score_dict["correct"] = False

        return score_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Return a sample marked as incorrect for all metrics."""
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["judge_correct"] = False
        if not prediction.get("generation", "").strip():
            prediction["generation"] = None
        return prediction

    def update(self, predictions):
        """Update metrics with new predictions."""
        super().update(predictions)

        predicted_answers = [
            pred.get("generation", None).strip() or None for pred in predictions
        ]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get computed metrics."""
        metrics_dict = super().get_metrics()
        
        for agg_mode, agg_metrics in metrics_dict.items():
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0
            
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            elif "judge_correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["judge_correct"]
        
        return metrics_dict

    def evaluations_to_print(self):
        """Specify which evaluation modes to print."""
        evals = [f"pass@{self.max_k}"]
        if self.max_k > 1:
            evals.extend([f"majority@{self.max_k}", f"pass@1[avg-of-{self.max_k}]"])
        return evals

    def metrics_to_print(self):
        """Specify which metrics to print in GPQA-style summary."""
        base_metrics = {
            "num_entries": as_int,
            "avg_tokens": as_int, 
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }
        
        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage
            
        return base_metrics
    

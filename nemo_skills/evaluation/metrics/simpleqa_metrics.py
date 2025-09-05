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

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage

CORRECT_LABEL = "A"
INCORRECT_LABEL = "B"
NOT_ATTEMPTED_LABEL = "C"


def is_correct_judgement_label_matching(judgement: str, correct_label: str) -> bool:
    """Check if judgement label matches the correct label.
    For example, if the correct label is "A", then "A" or "a" will be considered correct.

    For reference, SimpleQA judge returns: A: CORRECT, B: INCORRECT, C: NOT_ATTEMPTED
    """
    if not judgement:
        return False
    judgement = judgement.strip()
    if judgement == correct_label or judgement[0] == correct_label:
        return True
    return False


class SimpleQAMetrics(BaseMetrics):
    """Metrics for SimpleQA dataset which evaluates factual accuracy through judgement."""

    def __init__(self, compute_no_answer: bool = True, answer_key: str = "predicted_answer"):
        super().__init__(compute_no_answer=compute_no_answer)
        self.answer_key = answer_key

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """
        Returns correctness scores based on judgement for SimpleQA.

        SimpleQA uses judge-based evaluation where answers are rated as:
        - "A": CORRECT
        - "B": INCORRECT
        - "C": NOT_ATTEMPTED
        """
        correctness_dict = {}

        if "judgement" in prediction:
            correctness_dict["judge_correct"] = is_correct_judgement_label_matching(
                prediction["judgement"], CORRECT_LABEL
            )
            correctness_dict["judge_not_attempted"] = is_correct_judgement_label_matching(
                prediction["judgement"], NOT_ATTEMPTED_LABEL
            )

        return correctness_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Create a sample that evaluates as incorrect for filtering purposes."""
        prediction = prediction.copy()
        prediction["judgement"] = INCORRECT_LABEL
        prediction[self.answer_key] = None
        return prediction

    def update(self, predictions):
        """Update evaluation results with current predictions.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'judgement' field with judge evaluation.
        """
        super().update(predictions)
        predicted_answers = [pred[self.answer_key] for pred in predictions]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def evaluations_to_print(self):
        """Return evaluation metrics to be printed in final results."""
        return [
            f"pass@1[avg-of-{self.max_k}]",
            f"majority@{self.max_k}",
            f"pass@{self.max_k}",
        ]

    def metrics_to_print(self):
        """Return metrics to be displayed with their formatting."""
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "judge_correct": as_percentage,
            "judge_not_attempted": as_percentage,
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print

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


def compute_metrics(results):
    """Compute MultiChallenge metrics from evaluation results.

    Args:
        results: List of evaluation results, each containing:
            - question_id: Unique identifier
            - axis: Evaluation axis (e.g., REFINEMENT, EXPLICIT IF, COHERENCE, RECOLLECTION)
            - is_correct: Boolean indicating if any attempt passed
            - evaluations: List of evaluation results for each attempt

    Returns:
        Dictionary containing:
            - overall_score: Overall accuracy across all axes
            - axis_scores: Dictionary mapping axis names to their accuracy scores
            - total_questions: Total number of questions evaluated
            - correct_questions: Number of correctly answered questions
    """
    # Group results by axis
    axis_counts = defaultdict(lambda: {'passed': 0, 'total': 0, 'questions': set()})

    for result in results:
        question_id = result.get('question_id')
        axis = result.get('axis', 'UNKNOWN')
        is_correct = result.get('is_correct', False)

        # Only count each question once per axis
        if question_id not in axis_counts[axis]['questions']:
            axis_counts[axis]['total'] += 1
            axis_counts[axis]['questions'].add(question_id)

            if is_correct:
                axis_counts[axis]['passed'] += 1

    # Calculate scores for each axis
    axis_scores = {}
    for axis, counts in axis_counts.items():
        if counts['total'] > 0:
            axis_scores[axis] = (counts['passed'] / counts['total']) * 100
        else:
            axis_scores[axis] = 0.0

    # Calculate overall score
    if axis_scores:
        overall_score = sum(axis_scores.values()) / len(axis_scores)
    else:
        overall_score = 0.0

    # Calculate total statistics
    total_questions = sum(counts['total'] for counts in axis_counts.values())
    correct_questions = sum(counts['passed'] for counts in axis_counts.values())

    return {
        'overall_score': overall_score,
        'axis_scores': axis_scores,
        'total_questions': total_questions,
        'correct_questions': correct_questions,
        'accuracy': (correct_questions / total_questions * 100) if total_questions > 0 else 0.0,
    }


def format_metrics_report(metrics):
    """Format metrics into a human-readable report.

    Args:
        metrics: Dictionary of metrics from compute_metrics()

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("MultiChallenge Evaluation Results")
    report.append("=" * 60)
    report.append(f"\nOverall Score: {metrics['overall_score']:.2f}%")
    report.append(f"Total Questions: {metrics['total_questions']}")
    report.append(f"Correct Questions: {metrics['correct_questions']}")
    report.append(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    report.append("\n" + "-" * 60)
    report.append("Axis Scores:")
    report.append("-" * 60)

    for axis, score in sorted(metrics['axis_scores'].items()):
        report.append(f"{axis:30s}: {score:6.2f}%")

    report.append("=" * 60)

    return "\n".join(report)

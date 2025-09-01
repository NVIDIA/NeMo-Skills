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

import statistics
from unittest.mock import patch

from nemo_skills.evaluation.metrics.base import BaseMetrics


class BenchmarkStdMetrics(BaseMetrics):
    """Test implementation for benchmark std dev functionality."""

    def _get_score_dict(self, prediction):
        """Return correctness score."""
        return {"correct": float(prediction.get("is_correct", False))}


def test_add_benchmark_run_std_single_run():
    """Test _add_benchmark_run_std with k=1 returns zero."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [[1.0]]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 1, sample_list)

    assert "pass@1[std-across-1-runs]" in metrics_dict
    assert metrics_dict["pass@1[std-across-1-runs]"]["correct"] == 0.0


def test_add_benchmark_run_std_with_variance():
    """Test _add_benchmark_run_std calculates variance correctly."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [
        [1.0, 0.0, 1.0, 0.0],  # Sample 1
        [1.0, 1.0, 0.0, 0.0],  # Sample 2
        [0.0, 1.0, 1.0, 1.0],  # Sample 3
    ]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 4, sample_list)

    # Expected: Run 0=[1,1,0]→66.67%, Run 1=[0,1,1]→66.67%, Run 2=[1,0,1]→66.67%, Run 3=[0,0,1]→33.33%
    # Precise calculation: [66.666...%, 66.666...%, 66.666...%, 33.333...%]
    expected_run_averages = [66.66666666666666, 66.66666666666666, 66.66666666666666, 33.33333333333333]
    expected_std = statistics.stdev(expected_run_averages)
    actual_std = metrics_dict["pass@1[std-across-4-runs]"]["correct"]
    assert abs(actual_std - expected_std) < 1e-10  # Use precise comparison


def test_add_benchmark_run_std_identical_runs():
    """Test _add_benchmark_run_std with identical run averages produces zero std dev."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [
        [1.0, 0.0],  # Sample 1: attempts [1, 0]
        [0.0, 1.0],  # Sample 2: attempts [0, 1]
    ]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 2, sample_list)

    # Run 0: [1, 0] → 50%, Run 1: [0, 1] → 50%
    # stdev([50, 50]) = 0
    assert metrics_dict["pass@1[std-across-2-runs]"]["correct"] == 0.0


def test_add_average_sample_std_single_attempt():
    """Test _add_average_sample_std with k=1 returns zero."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [[1.0], [0.0]]

    metrics._add_average_sample_std(metrics_dict, "correct", 1, sample_list)

    assert "pass@1[avg-sample-std-of-1]" in metrics_dict
    assert metrics_dict["pass@1[avg-sample-std-of-1]"]["correct"] == 0.0


def test_add_average_sample_std_with_variance():
    """Test _add_average_sample_std calculates average correctly."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [
        [1.0, 0.0, 1.0],  # Sample 1: high variance
        [0.0, 0.0, 0.0],  # Sample 2: no variance
    ]

    metrics._add_average_sample_std(metrics_dict, "correct", 3, sample_list)

    # Expected: (stdev([100,0,100]) + stdev([0,0,0])) / 2 = (57.735... + 0) / 2
    sample_1_std = statistics.stdev([100.0, 0.0, 100.0])
    sample_2_std = 0.0  # All zeros have zero std dev
    expected_avg = (sample_1_std + sample_2_std) / 2
    actual_avg = metrics_dict["pass@1[avg-sample-std-of-3]"]["correct"]
    assert abs(actual_avg - expected_avg) < 1e-10


def test_add_average_sample_std_uniform_samples():
    """Test _add_average_sample_std with uniform samples returns zero."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [
        [1.0, 1.0, 1.0],  # Sample 1: all correct
        [0.0, 0.0, 0.0],  # Sample 2: all incorrect
    ]

    metrics._add_average_sample_std(metrics_dict, "correct", 3, sample_list)

    # Sample 1: stdev([100, 100, 100]) = 0, Sample 2: stdev([0, 0, 0]) = 0
    # Average: (0 + 0) / 2 = 0
    assert metrics_dict["pass@1[avg-sample-std-of-3]"]["correct"] == 0.0


@patch.object(BenchmarkStdMetrics, "_add_benchmark_run_std")
@patch.object(BenchmarkStdMetrics, "_add_average_sample_std")
def test_add_benchmark_std_metrics_orchestration(mock_avg_sample, mock_benchmark_run):
    """Test _add_benchmark_std_metrics calls both sub-methods correctly."""
    metrics = BenchmarkStdMetrics()
    metrics.all_scores = {"correct": {2: [[1.0, 0.0], [0.0, 1.0]], 3: [[1.0, 0.0, 1.0]]}}
    metrics_dict = {}

    metrics._add_benchmark_std_metrics(metrics_dict)

    # Should call both methods for each k value
    assert mock_benchmark_run.call_count == 2
    assert mock_avg_sample.call_count == 2

    # Verify correct arguments
    mock_benchmark_run.assert_any_call(metrics_dict, "correct", 2, [[1.0, 0.0], [0.0, 1.0]])
    mock_benchmark_run.assert_any_call(metrics_dict, "correct", 3, [[1.0, 0.0, 1.0]])


@patch.object(BenchmarkStdMetrics, "_add_benchmark_run_std")
@patch.object(BenchmarkStdMetrics, "_add_average_sample_std")
def test_add_benchmark_std_metrics_multiple_score_methods(mock_avg_sample, mock_benchmark_run):
    """Test _add_benchmark_std_metrics handles multiple score methods."""
    metrics = BenchmarkStdMetrics()
    metrics.all_scores = {"correct": {2: [[1.0, 0.0]]}, "partial": {2: [[0.5, 0.0]]}}
    metrics_dict = {}

    metrics._add_benchmark_std_metrics(metrics_dict)

    # Should call for both score methods
    assert mock_benchmark_run.call_count == 2
    assert mock_avg_sample.call_count == 2

    mock_benchmark_run.assert_any_call(metrics_dict, "correct", 2, [[1.0, 0.0]])
    mock_benchmark_run.assert_any_call(metrics_dict, "partial", 2, [[0.5, 0.0]])


def test_add_benchmark_std_metrics_skips_empty_lists():
    """Test _add_benchmark_std_metrics skips empty sample lists."""
    metrics = BenchmarkStdMetrics()
    metrics.all_scores = {
        "correct": {
            2: [[1.0, 0.0]],  # Non-empty
            3: [],  # Empty - should be skipped
        }
    }
    metrics_dict = {}

    # Should not raise any errors
    metrics._add_benchmark_std_metrics(metrics_dict)

    # Should only process k=2, skip k=3
    assert "pass@1[std-across-2-runs]" in metrics_dict
    assert "pass@1[std-across-3-runs]" not in metrics_dict


def test_add_average_sample_std_handles_short_samples():
    """Test _add_average_sample_std handles samples with fewer scores than k."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [
        [1.0, 0.0, 1.0],  # Sample 1: full 3 scores
        [1.0],  # Sample 2: only 1 score (< k=3)
    ]

    metrics._add_average_sample_std(metrics_dict, "correct", 3, sample_list)

    # Sample 1: stdev([100, 0, 100]) ≈ 57.735..., Sample 2: only 1 score → std dev = 0
    # Average: (57.735... + 0) / 2 ≈ 28.867...
    sample_1_std = statistics.stdev([100.0, 0.0, 100.0])
    sample_2_std = 0.0  # Single score has zero std dev
    expected_avg = (sample_1_std + sample_2_std) / 2
    actual_avg = metrics_dict["pass@1[avg-sample-std-of-3]"]["correct"]
    assert abs(actual_avg - expected_avg) < 1e-10


def test_benchmark_run_std_creates_nested_dict():
    """Test that _add_benchmark_run_std creates proper nested dictionary structure."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [[1.0, 0.0]]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 2, sample_list)

    assert "pass@1[std-across-2-runs]" in metrics_dict
    assert isinstance(metrics_dict["pass@1[std-across-2-runs]"], dict)
    assert "correct" in metrics_dict["pass@1[std-across-2-runs]"]
    assert isinstance(metrics_dict["pass@1[std-across-2-runs]"]["correct"], float)


def test_average_sample_std_creates_nested_dict():
    """Test that _add_average_sample_std creates proper nested dictionary structure."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}
    sample_list = [[1.0, 0.0]]

    metrics._add_average_sample_std(metrics_dict, "correct", 2, sample_list)

    assert "pass@1[avg-sample-std-of-2]" in metrics_dict
    assert isinstance(metrics_dict["pass@1[avg-sample-std-of-2]"], dict)
    assert "correct" in metrics_dict["pass@1[avg-sample-std-of-2]"]
    assert isinstance(metrics_dict["pass@1[avg-sample-std-of-2]"]["correct"], float)


def test_compute_pass_at_k_score_collection():
    """Test _compute_pass_at_k collects scores correctly."""
    metrics = BenchmarkStdMetrics()

    predictions = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]

    metrics._compute_pass_at_k(predictions)

    # Verify scores were collected for each k
    assert "correct" in metrics.all_scores
    assert 1 in metrics.all_scores["correct"]
    assert 2 in metrics.all_scores["correct"]
    assert 3 in metrics.all_scores["correct"]

    # Verify actual score values
    assert metrics.all_scores["correct"][1] == [[1.0]]
    assert metrics.all_scores["correct"][2] == [[1.0, 0.0]]
    assert metrics.all_scores["correct"][3] == [[1.0, 0.0, 1.0]]


@patch.object(BenchmarkStdMetrics, "_get_score_dict")
def test_compute_pass_at_k_multiple_score_methods(mock_get_score):
    """Test _compute_pass_at_k handles multiple score methods."""
    mock_get_score.return_value = {"correct": 1.0, "partial": 0.5}

    metrics = BenchmarkStdMetrics()
    predictions = [{"dummy": "data"}]

    metrics._compute_pass_at_k(predictions)

    # Should collect scores for both methods
    assert "correct" in metrics.all_scores
    assert "partial" in metrics.all_scores
    assert metrics.all_scores["correct"][1] == [[1.0]]
    assert metrics.all_scores["partial"][1] == [[0.5]]


def test_compute_pass_at_k_score_accumulation():
    """Test _compute_pass_at_k accumulates scores across calls."""
    metrics = BenchmarkStdMetrics()

    # First call
    predictions1 = [{"is_correct": True}, {"is_correct": False}]
    metrics._compute_pass_at_k(predictions1)

    # Second call
    predictions2 = [{"is_correct": False}, {"is_correct": True}]
    metrics._compute_pass_at_k(predictions2)

    # Should have accumulated both samples
    assert len(metrics.all_scores["correct"][2]) == 2
    assert metrics.all_scores["correct"][2] == [[1.0, 0.0], [0.0, 1.0]]


def test_compute_pass_at_k_non_binary_scores():
    """Test _compute_pass_at_k handles non-binary scores correctly."""

    class PartialMetrics(BaseMetrics):
        def _get_score_dict(self, prediction):
            return {"partial": prediction.get("score", 0.0)}

    partial_metrics = PartialMetrics()
    predictions = [{"score": 0.7}, {"score": 0.3}]

    partial_metrics._compute_pass_at_k(predictions)

    # Should collect the actual scores, not binary transformations
    assert partial_metrics.all_scores["partial"][1] == [[0.7]]
    assert partial_metrics.all_scores["partial"][2] == [[0.7, 0.3]]


def test_end_to_end_integration():
    """Test the complete flow from _compute_pass_at_k to get_metrics."""
    metrics = BenchmarkStdMetrics()

    # Simulate 3 samples, each with 3 attempts
    sample1_preds = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]
    sample2_preds = [{"is_correct": False}, {"is_correct": True}, {"is_correct": False}]
    sample3_preds = [{"is_correct": True}, {"is_correct": True}, {"is_correct": True}]

    # Process each sample
    metrics.update(sample1_preds)
    metrics._compute_pass_at_k(sample1_preds)

    metrics.update(sample2_preds)
    metrics._compute_pass_at_k(sample2_preds)

    metrics.update(sample3_preds)
    metrics._compute_pass_at_k(sample3_preds)

    # Get final metrics
    result_metrics = metrics.get_metrics()

    # Verify std dev metrics exist
    assert "pass@1[std-across-3-runs]" in result_metrics
    assert "pass@1[avg-sample-std-of-3]" in result_metrics
    assert "correct" in result_metrics["pass@1[std-across-3-runs]"]
    assert "correct" in result_metrics["pass@1[avg-sample-std-of-3]"]

    # Verify they are valid numbers
    benchmark_std = result_metrics["pass@1[std-across-3-runs]"]["correct"]
    sample_std = result_metrics["pass@1[avg-sample-std-of-3]"]["correct"]

    assert isinstance(benchmark_std, float) and benchmark_std >= 0.0
    assert isinstance(sample_std, float) and sample_std >= 0.0

    # Verify mathematical correctness
    # Expected benchmark runs: [1,0,1], [0,1,1], [1,0,1] → 66.67%, 66.67%, 66.67% → std = 0
    assert benchmark_std == 0.0

    # Expected sample stds: stdev([100,0,100])≈57.73, stdev([0,100,0])≈57.73, stdev([100,100,100])=0
    # Average: (57.73 + 57.73 + 0) / 3 ≈ 38.49
    sample_1_std = statistics.stdev([100.0, 0.0, 100.0])
    sample_2_std = statistics.stdev([0.0, 100.0, 0.0])
    sample_3_std = 0.0  # All identical values
    expected_sample_std = (sample_1_std + sample_2_std + sample_3_std) / 3
    assert abs(sample_std - expected_sample_std) < 1e-10


def test_single_prediction_edge_case():
    """Test behavior with single prediction."""
    metrics = BenchmarkStdMetrics()

    predictions = [{"is_correct": True}]
    metrics._compute_pass_at_k(predictions)

    # Should collect score for k=1 only
    assert "correct" in metrics.all_scores
    assert 1 in metrics.all_scores["correct"]
    assert metrics.all_scores["correct"][1] == [[1.0]]

    # Process through get_metrics
    metrics.update(predictions)
    result_metrics = metrics.get_metrics()

    # Should have std dev metrics with zero values
    assert result_metrics["pass@1[std-across-1-runs]"]["correct"] == 0.0
    assert result_metrics["pass@1[avg-sample-std-of-1]"]["correct"] == 0.0


def test_percentage_conversion_precision():
    """Test that percentage conversion is handled correctly."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}

    # Use fractional scores to test percentage conversion
    sample_list = [
        [0.75, 0.25],  # Sample 1: 75%, 25%
        [0.5, 0.5],  # Sample 2: 50%, 50%
    ]

    metrics._add_average_sample_std(metrics_dict, "correct", 2, sample_list)

    # Expected: Sample 1 std = stdev([75, 25]) = 35.36, Sample 2 std = stdev([50, 50]) = 0
    # Average: (35.36 + 0) / 2 = 17.68
    sample_1_std = statistics.stdev([75.0, 25.0])
    sample_2_std = 0.0
    expected_avg = (sample_1_std + sample_2_std) / 2

    actual_avg = metrics_dict["pass@1[avg-sample-std-of-2]"]["correct"]
    assert abs(actual_avg - expected_avg) < 1e-10


def test_large_k_value():
    """Test behavior with larger k values."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}

    # Create 5 samples with 6 attempts each
    sample_list = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # Alternating pattern
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # Opposite alternating
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # Two groups
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],  # Opposite two groups
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Front-loaded
    ]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 6, sample_list)

    # Calculate expected benchmark runs
    runs = []
    for attempt_idx in range(6):
        run = [sample[attempt_idx] for sample in sample_list]
        run_avg = sum(run) / len(run) * 100.0
        runs.append(run_avg)

    expected_std = statistics.stdev(runs)
    actual_std = metrics_dict["pass@1[std-across-6-runs]"]["correct"]

    assert abs(actual_std - expected_std) < 1e-10
    assert actual_std > 0  # Should have actual variance


def test_near_zero_but_not_zero_variance():
    """Test very small but non-zero variance cases."""
    metrics = BenchmarkStdMetrics()
    metrics_dict = {}

    # Samples with very small differences
    sample_list = [
        [1.0, 1.0, 1.0],  # Perfect scores
        [0.99, 0.99, 0.99],  # Nearly perfect
        [0.98, 0.99, 1.0],  # Small variation
    ]

    metrics._add_benchmark_run_std(metrics_dict, "correct", 3, sample_list)

    # Calculate expected
    run_0 = [1.0, 0.99, 0.98]  # 99%, 99%, 98% → avg = 98.67%
    run_1 = [1.0, 0.99, 0.99]  # 100%, 99%, 99% → avg = 99.33%
    run_2 = [1.0, 0.99, 1.0]  # 100%, 99%, 100% → avg = 99.67%

    run_averages = [sum(run_0) / 3 * 100, sum(run_1) / 3 * 100, sum(run_2) / 3 * 100]
    expected_std = statistics.stdev(run_averages)
    actual_std = metrics_dict["pass@1[std-across-3-runs]"]["correct"]

    assert abs(actual_std - expected_std) < 1e-10
    assert actual_std > 0  # Should be small but non-zero

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

from unittest.mock import patch

import pytest

from nemo_skills.evaluation.metrics.base import BaseMetrics


class BenchmarkStdMetrics(BaseMetrics):
    """Test implementation for benchmark std dev functionality."""

    def _get_score_dict(self, prediction):
        """Return correctness score."""
        return {"correct": float(prediction.get("is_correct", False))}


class TestBenchmarkAndSampleStd:
    """Test both _add_benchmark_run_std and _add_average_sample_std functionality."""

    @pytest.mark.parametrize(
        "k,sample_list,expected_benchmark_std,expected_sample_std,_description",
        [
            (1, [[1.0]], 0.0, 0.0, "k=1 always returns 0 for both"),
            (1, [[0.0]], 0.0, 0.0, "k=1 with different single value"),
            (1, [[1.0], [0.0], [1.0]], 0.0, 0.0, "k=1 with multiple samples"),
            (1, [[0.5, 0.8, 0.2]], 0.0, 0.0, "k=1 with fractional values"),
            (
                2,
                [[1.0, 0.0], [0.0, 1.0]],
                0.0,
                70.71067811865476,
                "Identical run averages (50%, 50%), different sample stds",
            ),
            (2, [[1.0, 0.0], [1.0]], 70.71067811865476, 35.35533905932738, "Different benchmark averages (50%, 100%)"),
            (
                3,
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                0.0,
                0.0,
                "Different benchmark averages (100%, 0%), but only 2 samples so k=3 uses only first 2 elements of each sample",
            ),
            (
                3,
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                28.867513459481287,
                28.867513459481287,
                "Mixed variance: benchmark (66.67%, 0%), sample stds",
            ),
            (
                3,
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                0.0,
                38.49001794597505,
                "Same benchmark averages (66.67%), different sample stds",
            ),
            (
                3,
                [[1.0, 1.0, 0.0], [0.5, 0.5, 0.5], [0.0, 1.0, 0.0]],
                33.33333333333334,
                38.49001794597505,
                "Mixed fractional scores",
            ),
            (
                4,
                [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
                16.666666666666664,
                55.15668461264172,
                "Original case",
            ),
            (
                5,
                [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0]],
                0.0,
                54.772255750516614,
                "k=5, identical averages (60%)",
            ),
        ],
    )
    def test_both_std_calculations(self, k, sample_list, expected_benchmark_std, expected_sample_std, _description):
        """Test both benchmark run std and average sample std calculations."""
        metrics = BenchmarkStdMetrics()

        metrics_dict_benchmark = {}
        metrics._add_benchmark_run_std(metrics_dict_benchmark, "correct", k, sample_list)
        actual_benchmark_std = metrics_dict_benchmark[f"pass@1[std-across-{k}-runs]"]["correct"]
        assert abs(actual_benchmark_std - expected_benchmark_std) < 1e-10, (
            f"Benchmark std: expected {expected_benchmark_std}, got {actual_benchmark_std}"
        )

        metrics_dict_sample = {}
        metrics._add_average_sample_std(metrics_dict_sample, "correct", k, sample_list)
        actual_sample_std = metrics_dict_sample[f"pass@1[avg-sample-std-of-{k}]"]["correct"]
        assert abs(actual_sample_std - expected_sample_std) < 1e-10, (
            f"Sample std: expected {expected_sample_std}, got {actual_sample_std}"
        )


class TestStdMetricsOrchestration:
    """Test _add_benchmark_std_metrics orchestration."""

    def test_orchestration_calls_both_methods(self):
        """Test _add_benchmark_std_metrics calls both sub-methods correctly."""
        with (
            patch.object(BenchmarkStdMetrics, "_add_benchmark_run_std") as mock_benchmark_run,
            patch.object(BenchmarkStdMetrics, "_add_average_sample_std") as mock_avg_sample,
        ):
            metrics = BenchmarkStdMetrics()
            metrics.all_scores = {"correct": {2: [[1.0, 0.0], [0.0, 1.0]], 3: [[1.0, 0.0, 1.0]]}}
            metrics_dict = {}
            metrics._add_benchmark_std_metrics(metrics_dict)
            assert mock_benchmark_run.call_count == 2
            assert mock_avg_sample.call_count == 2
            mock_benchmark_run.assert_any_call(metrics_dict, "correct", 2, [[1.0, 0.0], [0.0, 1.0]])
            mock_benchmark_run.assert_any_call(metrics_dict, "correct", 3, [[1.0, 0.0, 1.0]])

    def test_skips_empty_lists(self):
        """Test _add_benchmark_std_metrics skips empty sample lists."""
        metrics = BenchmarkStdMetrics()
        metrics.all_scores = {"correct": {2: [[1.0, 0.0]], 3: []}}
        metrics_dict = {}
        metrics._add_benchmark_std_metrics(metrics_dict)
        assert "pass@1[std-across-2-runs]" in metrics_dict
        assert "pass@1[std-across-3-runs]" not in metrics_dict

    def test_handles_multiple_score_methods(self):
        """Test _add_benchmark_std_metrics handles multiple score methods."""
        metrics = BenchmarkStdMetrics()
        metrics.all_scores = {"correct": {2: [[1.0, 0.0]]}, "partial": {2: [[0.5, 0.0]]}}
        metrics_dict = {}
        metrics._add_benchmark_std_metrics(metrics_dict)
        assert "correct" in metrics_dict["pass@1[std-across-2-runs]"]
        assert "partial" in metrics_dict["pass@1[std-across-2-runs]"]
        assert "correct" in metrics_dict["pass@1[avg-sample-std-of-2]"]
        assert "partial" in metrics_dict["pass@1[avg-sample-std-of-2]"]


class TestPassAtKIntegration:
    """Test integration with _compute_pass_at_k."""

    def test_score_collection(self):
        """Test _compute_pass_at_k collects scores correctly."""
        metrics = BenchmarkStdMetrics()
        predictions = [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}]
        metrics._compute_pass_at_k(predictions)
        assert "correct" in metrics.all_scores
        assert metrics.all_scores["correct"][1] == [[1.0]]
        assert metrics.all_scores["correct"][2] == [[1.0, 0.0]]
        assert metrics.all_scores["correct"][3] == [[1.0, 0.0, 1.0]]

    def test_score_accumulation(self):
        """Test _compute_pass_at_k accumulates scores across calls."""
        metrics = BenchmarkStdMetrics()
        metrics._compute_pass_at_k([{"is_correct": True}, {"is_correct": False}])
        metrics._compute_pass_at_k([{"is_correct": False}, {"is_correct": True}])
        assert len(metrics.all_scores["correct"][2]) == 2
        assert metrics.all_scores["correct"][2] == [[1.0, 0.0], [0.0, 1.0]]

    def test_multiple_score_methods(self):
        """Test _compute_pass_at_k handles multiple score methods."""

        class MultiScoreMetrics(BaseMetrics):
            def _get_score_dict(self, prediction):
                return {"correct": 1.0, "partial": 0.5}

        metrics = MultiScoreMetrics()
        predictions = [{"dummy": "data"}]
        metrics._compute_pass_at_k(predictions)
        assert "correct" in metrics.all_scores
        assert "partial" in metrics.all_scores
        assert metrics.all_scores["correct"][1] == [[1.0]]
        assert metrics.all_scores["partial"][1] == [[0.5]]


class TestEvaluationsToPrint:
    """Test evaluations_to_print functionality."""

    @pytest.mark.parametrize(
        "max_k,expected_std_metrics",
        [
            (1, []),
            (2, ["pass@1[std-across-2-runs]", "pass@1[avg-sample-std-of-2]"]),
            (3, ["pass@1[std-across-3-runs]", "pass@1[avg-sample-std-of-3]"]),
            (4, ["pass@1[std-across-4-runs]", "pass@1[avg-sample-std-of-4]"]),
            (5, ["pass@1[std-across-5-runs]", "pass@1[avg-sample-std-of-5]"]),
            (10, ["pass@1[std-across-10-runs]", "pass@1[avg-sample-std-of-10]"]),
            (100, ["pass@1[std-across-100-runs]", "pass@1[avg-sample-std-of-100]"]),
        ],
    )
    def test_evaluations_to_print(self, max_k, expected_std_metrics):
        """Test evaluations_to_print includes std dev metrics when k > 1."""
        metrics = BenchmarkStdMetrics()
        metrics.max_k = max_k
        evaluations = metrics.evaluations_to_print()
        assert f"pass@1[avg-of-{max_k}]" in evaluations
        assert f"majority@{max_k}" in evaluations
        assert f"pass@{max_k}" in evaluations
        for std_metric in expected_std_metrics:
            assert std_metric in evaluations
        if max_k == 1:
            assert not any("std-across" in str(e) or "avg-sample-std" in str(e) for e in evaluations)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for std dev calculations."""

    @pytest.mark.parametrize(
        "k,sample_list,_description",
        [
            (1, [[]], "k=1 with empty sample"),
            (1, [[0.1, 0.9, 0.5, 0.7, 0.3]], "k=1 with fractional values"),
            (1, [[1.0] * 1000], "k=1 with extremely long sample"),
        ],
    )
    def test_edge_cases_no_crash(self, k, sample_list, _description):
        """Test that edge cases don't crash and produce reasonable results."""
        metrics = BenchmarkStdMetrics()
        metrics_dict = {}
        metrics._add_benchmark_run_std(metrics_dict, "correct", k, sample_list)
        metrics._add_average_sample_std(metrics_dict, "correct", k, sample_list)
        if f"pass@1[std-across-{k}-runs]" in metrics_dict:
            std_val = metrics_dict[f"pass@1[std-across-{k}-runs]"]["correct"]
            assert isinstance(std_val, (int, float)) and std_val >= 0
        if f"pass@1[avg-sample-std-of-{k}]" in metrics_dict:
            avg_val = metrics_dict[f"pass@1[avg-sample-std-of-{k}]"]["correct"]
            assert isinstance(avg_val, (int, float)) and avg_val >= 0


class TestInheritanceCompatibility:
    """Test that all metrics classes inherit std dev functionality correctly."""

    @pytest.mark.parametrize(
        "metrics_class",
        [
            "BFCLMetrics",
            "EvalPlusMetrics",
            "LiveCodeBenchMetrics",
            "SweBenchMetrics",
            "SciCodeMetrics",
            "Lean4Metrics",
            "RulerMetrics",
            "MRCRMetrics",
        ],
    )
    def test_metrics_classes_inherit_std_dev(self, metrics_class):
        """Test that all metrics classes inherit std dev functionality from BaseMetrics."""
        if metrics_class == "BFCLMetrics":
            from nemo_skills.evaluation.metrics.bfcl_metrics import BFCLMetrics

            cls = BFCLMetrics
        elif metrics_class == "EvalPlusMetrics":
            from nemo_skills.evaluation.metrics.code_metrics import EvalPlusMetrics

            cls = EvalPlusMetrics
        elif metrics_class == "LiveCodeBenchMetrics":
            from nemo_skills.evaluation.metrics.code_metrics import LiveCodeBenchMetrics

            cls = LiveCodeBenchMetrics
        elif metrics_class == "SweBenchMetrics":
            from nemo_skills.evaluation.metrics.code_metrics import SweBenchMetrics

            cls = SweBenchMetrics
        elif metrics_class == "SciCodeMetrics":
            from nemo_skills.evaluation.metrics.code_metrics import SciCodeMetrics

            cls = SciCodeMetrics
        elif metrics_class == "Lean4Metrics":
            from nemo_skills.evaluation.metrics.lean4_metrics import Lean4Metrics

            cls = Lean4Metrics
        elif metrics_class == "RulerMetrics":
            from nemo_skills.evaluation.metrics.ruler_metrics import RulerMetrics

            cls = RulerMetrics
        elif metrics_class == "MRCRMetrics":
            from nemo_skills.evaluation.metrics.mrcr_metrics import MRCRMetrics

            cls = MRCRMetrics
        else:
            raise ValueError(f"Unknown metrics class: {metrics_class}")

        instance = cls()

        instance.max_k = 1
        evaluations_k1 = instance.evaluations_to_print()
        assert not any("std-across" in str(e) or "avg-sample-std" in str(e) for e in evaluations_k1)

        instance.max_k = 3
        evaluations_k3 = instance.evaluations_to_print()
        assert "pass@1[std-across-3-runs]" in evaluations_k3
        assert "pass@1[avg-sample-std-of-3]" in evaluations_k3


class TestScoreTypesCompatibility:
    """Test std dev calculations work with different score types."""

    def test_mixed_score_types(self):
        """Test std dev calculations work correctly with bool, int, and float scores."""

        class MixedScoreMetrics(BaseMetrics):
            def _get_score_dict(self, prediction):
                return {
                    "binary_score": prediction.get("binary_score", True),
                    "integer_score": prediction.get("integer_score", 5),
                    "float_score": prediction.get("float_score", 0.75),
                }

        metrics = MixedScoreMetrics()

        sample1 = [
            {"binary_score": True, "integer_score": 10, "float_score": 0.9},
            {"binary_score": False, "integer_score": 3, "float_score": 0.4},
            {"binary_score": True, "integer_score": 8, "float_score": 0.8},
        ]
        sample2 = [
            {"binary_score": False, "integer_score": 2, "float_score": 0.2},
            {"binary_score": True, "integer_score": 7, "float_score": 0.7},
            {"binary_score": False, "integer_score": 1, "float_score": 0.1},
        ]

        for preds in [sample1, sample2]:
            metrics.update(preds)
            metrics._compute_pass_at_k(preds)
        result_metrics = metrics.get_metrics()
        assert "pass@1[std-across-3-runs]" in result_metrics
        assert "pass@1[avg-sample-std-of-3]" in result_metrics
        for score_method in ["binary_score", "integer_score", "float_score"]:
            assert score_method in result_metrics["pass@1[std-across-3-runs]"]
            assert score_method in result_metrics["pass@1[avg-sample-std-of-3]"]
            benchmark_std = result_metrics["pass@1[std-across-3-runs]"][score_method]
            sample_std = result_metrics["pass@1[avg-sample-std-of-3]"][score_method]
            assert isinstance(benchmark_std, (int, float)) and benchmark_std >= 0
            assert isinstance(sample_std, (int, float)) and sample_std >= 0


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""

    def test_complete_flow(self):
        """Test the complete flow from _compute_pass_at_k to get_metrics."""
        metrics = BenchmarkStdMetrics()
        # Simulate 3 samples, each with 3 attempts
        samples = [
            [{"is_correct": True}, {"is_correct": False}, {"is_correct": True}],  # [1,0,1]
            [{"is_correct": False}, {"is_correct": True}, {"is_correct": False}],  # [0,1,0]
            [{"is_correct": True}, {"is_correct": True}, {"is_correct": True}],  # [1,1,1]
        ]
        for sample_preds in samples:
            metrics.update(sample_preds)
            metrics._compute_pass_at_k(sample_preds)
        result_metrics = metrics.get_metrics()
        assert "pass@1[std-across-3-runs]" in result_metrics
        assert "pass@1[avg-sample-std-of-3]" in result_metrics
        assert "correct" in result_metrics["pass@1[std-across-3-runs]"]
        assert "correct" in result_metrics["pass@1[avg-sample-std-of-3]"]
        benchmark_std = result_metrics["pass@1[std-across-3-runs]"]["correct"]
        sample_std = result_metrics["pass@1[avg-sample-std-of-3]"]["correct"]
        assert isinstance(benchmark_std, float) and benchmark_std >= 0.0
        assert isinstance(sample_std, float) and sample_std >= 0.0
        # Expected benchmark runs: [1,0,1], [0,1,1], [1,0,1] → all 66.67% → std = 0
        assert benchmark_std == 0.0
        # Expected sample stds: stdev([100,0,100])≈57.73, stdev([0,100,0])≈57.73, stdev([100,100,100])=0
        # Average: (57.73 + 57.73 + 0) / 3 ≈ 38.49
        expected_sample_std = 38.49001794597505
        assert abs(sample_std - expected_sample_std) < 1e-10

    @pytest.mark.parametrize(
        "real_metrics_class,test_data",
        [
            (
                "EvalPlusMetrics",
                [
                    [{"is_correct": True, "is_correct-plus": False}, {"is_correct": False, "is_correct-plus": False}],
                    [{"is_correct": True, "is_correct-plus": True}, {"is_correct": True, "is_correct-plus": False}],
                ],
            ),
            (
                "EvalPlusMetrics",
                [
                    [{"is_correct": True, "is_correct-plus": True}],
                    [{"is_correct": False, "is_correct-plus": False}],
                    [{"is_correct": True, "is_correct-plus": False}],
                    [{"is_correct": True, "is_correct-plus": True}],
                ],
            ),
            (
                "MathMetrics",
                [
                    [{"symbolic_correct": True, "judgement": "Judgement: Yes", "predicted_answer": "42"}],
                    [{"symbolic_correct": False, "judgement": "Judgement: No", "predicted_answer": "43"}],
                ],
            ),
            (
                "MathMetrics",
                [
                    [{"symbolic_correct": True, "judgement": "Judgement: Yes", "predicted_answer": "1"}],
                    [{"symbolic_correct": True, "judgement": "Judgement: Yes", "predicted_answer": "2"}],
                    [{"symbolic_correct": False, "judgement": "Judgement: No", "predicted_answer": "3"}],
                ],
            ),
            (
                "MathMetrics",
                [
                    [{"symbolic_correct": False, "judgement": "Judgement: No", "predicted_answer": "wrong"}] * 5,
                    [{"symbolic_correct": True, "judgement": "Judgement: Yes", "predicted_answer": "right"}] * 5,
                ],
            ),
        ],
    )
    def test_real_world_integration(self, real_metrics_class, test_data):
        """Test std dev calculations work end-to-end with real metrics classes."""
        if real_metrics_class == "EvalPlusMetrics":
            from nemo_skills.evaluation.metrics.code_metrics import EvalPlusMetrics

            metrics = EvalPlusMetrics()
            expected_keys = ["passing_base_tests", "passing_plus_tests"]
        elif real_metrics_class == "MathMetrics":
            from nemo_skills.evaluation.metrics.math_metrics import MathMetrics

            metrics = MathMetrics()
            expected_keys = ["symbolic_correct", "judge_correct"]
        else:
            raise ValueError(f"Unknown real_metrics_class: {real_metrics_class}")
        for preds in test_data:
            metrics.update(preds)
        result = metrics.get_metrics()
        std_across_key = f"pass@1[std-across-{len(test_data[0])}-runs]"

        assert std_across_key in result
        assert any(key in result[std_across_key] for key in expected_keys)

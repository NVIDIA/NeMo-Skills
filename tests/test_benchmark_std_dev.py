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


import pytest

from nemo_skills.evaluation.metrics.base import BaseMetrics


class BenchmarkStdMetrics(BaseMetrics):
    """Test implementation for benchmark std dev functionality."""

    def _get_score_dict(self, prediction):
        """Return correctness score."""
        return {"correct": float(prediction.get("is_correct", False))}


class TestBenchmarkAndSampleStd:
    """Test std dev metrics integration as columns."""

    @pytest.mark.parametrize(
        "k,sample_list,expected_benchmark_std,expected_sample_std,expected_benchmark_std_err,expected_sample_std_err,_description",
        [
            (1, [[1.0]], 0.0, 0.0, 0.0, 0.0, "k=1 always returns 0 for both"),
            (
                2,
                [[1.0, 0.0], [0.0, 1.0]],
                0.0,
                0.7071067811865476,
                0.0,
                0.5000,
                "Identical run averages (50%, 50%), different sample stds",
            ),
            (
                2,
                [[1.0, 0.0], [1.0, 0.0]],
                0.7071067811865476,
                0.7071067811865476,
                0.5000,
                0.5000,
                "Different benchmark averages (50%, 100%)",
            ),
            (
                3,
                [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                0.28867513459481287,
                0.28867513459481287,
                0.16666666666666666,
                0.20412414523193154,
                "Mixed variance: benchmark (66.67%, 0%), sample stds",
            ),
            (
                3,
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                0.0,
                0.38490017945975047,
                0.0,
                0.22222222222222221,
                "Same benchmark averages (66.67%), different sample stds",
            ),
            (
                4,
                [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
                0.16666666666666666,
                0.5515668461264172,
                0.08333333333333333,
                0.31844726708715987,
                "Original case",
            ),
        ],
    )
    def test_std_columns_added_to_evaluation_modes(
        self,
        k,
        sample_list,
        expected_benchmark_std,
        expected_sample_std,
        expected_benchmark_std_err,
        expected_sample_std_err,
        _description,
    ):
        """Test that std dev and std err metrics are added as columns to existing evaluation modes."""
        metrics = BenchmarkStdMetrics()
        metrics.max_k = k
        metrics.all_scores = {"correct": {k: sample_list}}

        # Create evaluation modes that would normally be created by _compute_pass_at_k
        metrics_dict = {
            f"pass@1[avg-of-{k}]": {"num_entries": len(sample_list), "correct": 50.0},
            f"majority@{k}": {"num_entries": len(sample_list), "correct": 60.0},
            f"pass@{k}": {"num_entries": len(sample_list), "correct": 80.0},
        }

        if k == 1:
            # k=1 should not add std or std err columns
            metrics._add_benchmark_std_metrics(metrics_dict)
            for eval_mode in metrics_dict:
                assert "correct_std_across_runs" not in metrics_dict[eval_mode]
                assert "correct_avg_sample_std" not in metrics_dict[eval_mode]
                assert "correct_std_err_across_runs" not in metrics_dict[eval_mode]
                assert "correct_avg_sample_std_err" not in metrics_dict[eval_mode]
        else:
            # k > 1 should add std and std err columns
            metrics._add_benchmark_std_metrics(metrics_dict)

            # Check that only pass@1[avg-of-k] has std and std err columns
            for eval_mode in [f"pass@1[avg-of-{k}]"]:
                assert "correct_std_across_runs" in metrics_dict[eval_mode]
                assert "correct_avg_sample_std" in metrics_dict[eval_mode]
                assert "correct_std_err_across_runs" in metrics_dict[eval_mode]
                assert "correct_avg_sample_std_err" in metrics_dict[eval_mode]

                actual_benchmark_std = metrics_dict[eval_mode]["correct_std_across_runs"]
                actual_sample_std = metrics_dict[eval_mode]["correct_avg_sample_std"]
                actual_benchmark_std_err = metrics_dict[eval_mode]["correct_std_err_across_runs"]
                actual_sample_std_err = metrics_dict[eval_mode]["correct_avg_sample_std_err"]

                assert abs(actual_benchmark_std - expected_benchmark_std) < 1e-10, (
                    f"{eval_mode} benchmark std: expected {expected_benchmark_std}, got {actual_benchmark_std}"
                )
                assert abs(actual_sample_std - expected_sample_std) < 1e-10, (
                    f"{eval_mode} sample std: expected {expected_sample_std}, got {actual_sample_std}"
                )
                assert abs(actual_benchmark_std_err - expected_benchmark_std_err) < 1e-10, (
                    f"{eval_mode} benchmark std err: expected {expected_benchmark_std_err}, got {actual_benchmark_std_err}"
                )
                assert abs(actual_sample_std_err - expected_sample_std_err) < 1e-10, (
                    f"{eval_mode} sample std err: expected {expected_sample_std_err}, got {actual_sample_std_err}"
                )

            for eval_mode in [f"majority@{k}", f"pass@{k}"]:
                assert "correct_std_across_runs" not in metrics_dict[eval_mode]
                assert "correct_avg_sample_std" not in metrics_dict[eval_mode]
                assert "correct_std_err_across_runs" not in metrics_dict[eval_mode]
                assert "correct_avg_sample_std_err" not in metrics_dict[eval_mode]


class TestStdMetricsOrchestration:
    """Test _add_benchmark_std_metrics orchestration."""

    def test_processes_all_k_values(self):
        """Test _add_benchmark_std_metrics processes all k values with correct sample counts."""
        metrics = BenchmarkStdMetrics()
        metrics.all_scores = {"correct": {2: [[1.0, 0.0], [0.0, 1.0]], 3: [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]}}

        metrics_dict = {
            "pass@1[avg-of-2]": {"correct": 50.0},
            "pass@2": {"correct": 75.0},
            "pass@1[avg-of-3]": {"correct": 67.0},
        }

        metrics._add_benchmark_std_metrics(metrics_dict)

        assert "correct_std_across_runs" in metrics_dict["pass@1[avg-of-2]"]
        assert "correct_avg_sample_std" in metrics_dict["pass@1[avg-of-2]"]
        assert "correct_std_err_across_runs" in metrics_dict["pass@1[avg-of-2]"]
        assert "correct_avg_sample_std_err" in metrics_dict["pass@1[avg-of-2]"]
        assert "correct_std_across_runs" in metrics_dict["pass@1[avg-of-3]"]
        assert "correct_avg_sample_std" in metrics_dict["pass@1[avg-of-3]"]
        assert "correct_std_err_across_runs" in metrics_dict["pass@1[avg-of-3]"]
        assert "correct_avg_sample_std_err" in metrics_dict["pass@1[avg-of-3]"]
        assert "correct_std_across_runs" not in metrics_dict["pass@2"]
        assert "correct_avg_sample_std" not in metrics_dict["pass@2"]
        assert "correct_std_err_across_runs" not in metrics_dict["pass@2"]
        assert "correct_avg_sample_std_err" not in metrics_dict["pass@2"]

    def test_handles_multiple_score_methods(self):
        """Test _add_benchmark_std_metrics handles multiple score methods."""
        metrics = BenchmarkStdMetrics()
        metrics.all_scores = {"correct": {2: [[1.0, 0.0], [0.0, 1.0]]}, "partial": {2: [[0.5, 0.0], [0.0, 0.5]]}}

        metrics_dict = {
            "pass@1[avg-of-2]": {"correct": 50.0, "partial": 25.0},
            "pass@2": {"correct": 100.0, "partial": 50.0},
        }

        metrics._add_benchmark_std_metrics(metrics_dict)

        for eval_mode in ["pass@1[avg-of-2]"]:
            assert "correct_std_across_runs" in metrics_dict[eval_mode]
            assert "correct_avg_sample_std" in metrics_dict[eval_mode]
            assert "correct_std_err_across_runs" in metrics_dict[eval_mode]
            assert "correct_avg_sample_std_err" in metrics_dict[eval_mode]
            assert "partial_std_across_runs" in metrics_dict[eval_mode]
            assert "partial_avg_sample_std" in metrics_dict[eval_mode]
            assert "partial_std_err_across_runs" in metrics_dict[eval_mode]
            assert "partial_avg_sample_std_err" in metrics_dict[eval_mode]

        for eval_mode in ["pass@2"]:
            assert "correct_std_across_runs" not in metrics_dict[eval_mode]
            assert "correct_avg_sample_std" not in metrics_dict[eval_mode]
            assert "correct_std_err_across_runs" not in metrics_dict[eval_mode]
            assert "correct_avg_sample_std_err" not in metrics_dict[eval_mode]
            assert "partial_std_across_runs" not in metrics_dict[eval_mode]
            assert "partial_avg_sample_std" not in metrics_dict[eval_mode]
            assert "partial_std_err_across_runs" not in metrics_dict[eval_mode]
            assert "partial_avg_sample_std_err" not in metrics_dict[eval_mode]


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
        "max_k",
        [1, 2, 3, 4, 5, 10, 100],
    )
    def test_evaluations_to_print_no_std_modes(self, max_k):
        """Test evaluations_to_print no longer includes separate std evaluation modes."""
        metrics = BenchmarkStdMetrics()
        metrics.max_k = max_k
        evaluations = metrics.evaluations_to_print()

        # Should always have basic evaluation modes
        assert f"pass@1[avg-of-{max_k}]" in evaluations
        assert f"majority@{max_k}" in evaluations
        assert f"pass@{max_k}" in evaluations


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions for std dev calculations."""

    def test_k_equals_1_no_std_columns(self):
        """Test that k=1 doesn't add std columns."""
        metrics = BenchmarkStdMetrics()
        metrics.max_k = 1
        metrics.all_scores = {"correct": {1: [[1.0]]}}

        metrics_dict = {"pass@1[avg-of-1]": {"correct": 100.0}}

        metrics._add_benchmark_std_metrics(metrics_dict)

        # Should not have std or std err columns for k=1
        assert "correct_std_across_runs" not in metrics_dict["pass@1[avg-of-1]"]
        assert "correct_avg_sample_std" not in metrics_dict["pass@1[avg-of-1]"]
        assert "correct_std_err_across_runs" not in metrics_dict["pass@1[avg-of-1]"]
        assert "correct_avg_sample_std_err" not in metrics_dict["pass@1[avg-of-1]"]


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

        # Test that evaluations_to_print works for different k values
        instance.max_k = 1
        evaluations_k1 = instance.evaluations_to_print()
        assert len(evaluations_k1) > 0

        instance.max_k = 3
        evaluations_k3 = instance.evaluations_to_print()
        assert len(evaluations_k3) > 0


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

        # Check that std columns are added to existing evaluation modes
        assert "pass@1[avg-of-3]" in result_metrics

        # Check that each score method has std and std err columns in the evaluation mode
        for score_method in ["binary_score", "integer_score", "float_score"]:
            assert f"{score_method}_std_across_runs" in result_metrics["pass@1[avg-of-3]"]
            assert f"{score_method}_avg_sample_std" in result_metrics["pass@1[avg-of-3]"]
            assert f"{score_method}_std_err_across_runs" in result_metrics["pass@1[avg-of-3]"]
            assert f"{score_method}_avg_sample_std_err" in result_metrics["pass@1[avg-of-3]"]
            benchmark_std = result_metrics["pass@1[avg-of-3]"][f"{score_method}_std_across_runs"]
            sample_std = result_metrics["pass@1[avg-of-3]"][f"{score_method}_avg_sample_std"]
            benchmark_std_err = result_metrics["pass@1[avg-of-3]"][f"{score_method}_std_err_across_runs"]
            sample_std_err = result_metrics["pass@1[avg-of-3]"][f"{score_method}_avg_sample_std_err"]
            assert isinstance(benchmark_std, (int, float)) and benchmark_std >= 0
            assert isinstance(sample_std, (int, float)) and sample_std >= 0
            assert isinstance(benchmark_std_err, (int, float)) and benchmark_std_err >= 0
            assert isinstance(sample_std_err, (int, float)) and sample_std_err >= 0


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""

    def test_complete_flow(self):
        """Test the complete flow from _compute_pass_at_k to get_metrics with std columns."""
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

        # Check that std columns are added to existing evaluation modes
        assert "pass@1[avg-of-3]" in result_metrics

        # Check that the main evaluation mode has std and std err columns
        assert "correct_std_across_runs" in result_metrics["pass@1[avg-of-3]"]
        assert "correct_avg_sample_std" in result_metrics["pass@1[avg-of-3]"]
        assert "correct_std_err_across_runs" in result_metrics["pass@1[avg-of-3]"]
        assert "correct_avg_sample_std_err" in result_metrics["pass@1[avg-of-3]"]

        benchmark_std = result_metrics["pass@1[avg-of-3]"]["correct_std_across_runs"]
        sample_std = result_metrics["pass@1[avg-of-3]"]["correct_avg_sample_std"]
        benchmark_std_err = result_metrics["pass@1[avg-of-3]"]["correct_std_err_across_runs"]
        sample_std_err = result_metrics["pass@1[avg-of-3]"]["correct_avg_sample_std_err"]

        assert isinstance(benchmark_std, float) and benchmark_std >= 0.0
        assert isinstance(sample_std, float) and sample_std >= 0.0
        assert isinstance(benchmark_std_err, float) and benchmark_std_err >= 0.0
        assert isinstance(sample_std_err, float) and sample_std_err >= 0.0

        # Test specific values
        # Expected benchmark runs: [1,0,1], [0,1,1], [1,0,1] → all 66.67% → std = 0
        assert benchmark_std == 0.0
        # Expected sample stds: stdev([1,0,1])≈0.5773, stdev([0,1,0])≈0.5773, stdev([1,1,1])=0
        # Average: (0.5773 + 0.5773 + 0) / 3 ≈ 0.3849
        expected_sample_std = 0.38490017945975047
        assert abs(sample_std - expected_sample_std) < 1e-10

        # Check other evaluation modes that might exist
        for eval_mode in result_metrics:
            if eval_mode.startswith("pass@") and "correct_std_across_runs" in result_metrics[eval_mode]:
                # All std and std err columns should be valid floats >= 0
                assert isinstance(result_metrics[eval_mode]["correct_std_across_runs"], float)
                assert result_metrics[eval_mode]["correct_std_across_runs"] >= 0.0
                assert isinstance(result_metrics[eval_mode]["correct_avg_sample_std"], float)
                assert result_metrics[eval_mode]["correct_avg_sample_std"] >= 0.0
                assert isinstance(result_metrics[eval_mode]["correct_std_err_across_runs"], float)
                assert result_metrics[eval_mode]["correct_std_err_across_runs"] >= 0.0
                assert isinstance(result_metrics[eval_mode]["correct_avg_sample_std_err"], float)
                assert result_metrics[eval_mode]["correct_avg_sample_std_err"] >= 0.0

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

        # Check that std columns are added to evaluation modes
        k = len(test_data[0])
        eval_modes_to_check = [f"pass@1[avg-of-{k}]"]

        for eval_mode in eval_modes_to_check:
            if eval_mode in result and k > 1:
                # Should have std and std err columns for k > 1
                for expected_key in expected_keys:
                    if expected_key in result[eval_mode]:  # Only check if base metric exists
                        assert f"{expected_key}_std_across_runs" in result[eval_mode]
                        assert f"{expected_key}_avg_sample_std" in result[eval_mode]
                        assert f"{expected_key}_std_err_across_runs" in result[eval_mode]
                        assert f"{expected_key}_avg_sample_std_err" in result[eval_mode]

        eval_modes_to_check = [f"majority@{k}", f"pass@{k}"]
        for eval_mode in eval_modes_to_check:
            if eval_mode in result:
                for expected_key in expected_keys:
                    if expected_key in result[eval_mode]:
                        assert f"{expected_key}_std_across_runs" not in result[eval_mode]
                        assert f"{expected_key}_avg_sample_std" not in result[eval_mode]
                        assert f"{expected_key}_std_err_across_runs" not in result[eval_mode]
                        assert f"{expected_key}_avg_sample_std_err" not in result[eval_mode]

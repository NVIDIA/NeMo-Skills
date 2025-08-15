# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path):
    """Load a JSON file from the given string path."""
    if not os.path.isfile(path):
        raise IOError("File not found: {}".format(path))
    with open(path, "r") as f:
        return json.load(f)


def fmt_pct(x, nd=2):
    """Format a number as percentage text with a % sign (no scaling)."""
    return "{0:.{1}f}%".format(x, nd)


def in_any_range(value, ranges):
    """Return True if value is inside ANY (lo, hi) inclusive range."""
    for lo, hi in ranges:
        if lo <= value <= hi:
            return True
    return False


def get_aime_symbolic_avg8(d, bench_key):
    """
    Return float(d[bench_key]['pass@1[avg-of-8]']['symbolic_correct']).
    The root JSON is expected to have benchmark keys at the top-level.
    """
    try:
        return float(d[bench_key]["pass@1[avg-of-8]"]["symbolic_correct"])
    except KeyError as e:
        raise KeyError("Missing key for {}.pass@1[avg-of-8].symbolic_correct".format(bench_key))
    except Exception as e:
        raise ValueError("Non-numeric value at {}.pass@1[avg-of-8].symbolic_correct".format(bench_key))


RANGE_CONSTRAINTS = {
    "after_training": {
        "aime24": [(20.0, 30.0)],
        "aime25": [(17.5, 27.5)],
    },
    "baseline": {
        "aime24": [(6.25, 16.25)],
        "aime25": [(8.75, 18.75)],
    },
}


def check_benchmark(benchmark, baseline_results, after_training_results):
    lines = []

    baseline_accuracy = get_aime_symbolic_avg8(baseline_results, benchmark)
    after_training_accuracy = get_aime_symbolic_avg8(after_training_results, benchmark)

    # Condition 1: strict improvement
    improvement_pass = after_training_accuracy > baseline_accuracy

    # Condition 2: after training accuracy in range (assume ranges exist)

    after_training_ranges = RANGE_CONSTRAINTS["after_training"][benchmark]
    after_training_in_range = in_any_range(after_training_accuracy, after_training_ranges)
    after_training_range_status = "OK" if after_training_in_range else "FAIL"
    after_training_range_desc = " OR ".join(
        "[{},{}]".format(fmt_pct(lo), fmt_pct(hi)) for lo, hi in after_training_ranges
    )

    # Condition 3: baseline accuracy in range (assume ranges exist)

    baseline_ranges = RANGE_CONSTRAINTS["baseline"][benchmark]
    baseline_in_range = in_any_range(baseline_accuracy, baseline_ranges)
    baseline_range_status = "OK" if baseline_in_range else "FAIL"
    baseline_range_desc = " OR ".join("[{},{}]".format(fmt_pct(lo), fmt_pct(hi)) for lo, hi in baseline_ranges)

    # Overall pass
    overall_pass = improvement_pass and after_training_in_range and baseline_in_range
    overall_status = "OK" if overall_pass else "FAIL"

    # Print lines
    lines.append("[check] --- {} ---".format(benchmark))
    lines.append(
        "[check] accuracy: after training={}, baseline={}".format(
            fmt_pct(after_training_accuracy), fmt_pct(baseline_accuracy)
        )
    )
    lines.append(
        "[check] condition 1: improvement -> after training > baseline ({} > {})  RESULT={}".format(
            fmt_pct(after_training_accuracy), fmt_pct(baseline_accuracy), "OK" if improvement_pass else "FAIL"
        )
    )
    lines.append(
        "[check] condition 2: after training accuracy in-range -> ranges {}  ACCURACY={}  RESULT={}".format(
            after_training_range_desc, fmt_pct(after_training_accuracy), after_training_range_status
        )
    )
    lines.append(
        "[check] condition 3: baseline accuracy in-range -> ranges {}  ACCURACY={}  RESULT={}".format(
            baseline_range_desc, fmt_pct(baseline_accuracy), baseline_range_status
        )
    )
    lines.append("[check] RESULT: {}".format(overall_status))

    return overall_pass, lines


def main():
    parser = argparse.ArgumentParser(
        description="Compare after-training vs baseline metrics for AIME24/25 (metric fixed to pass@1[avg-of-8].symbolic_correct)."
    )
    parser.add_argument("--workspace", required=True, help="Base workspace directory containing eval results.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser()
    baseline_metric_path = os.path.join(workspace, "evals", "baseline", "eval-results", "metrics.json")
    after_training_metric_path = os.path.join(workspace, "evals", "after-training", "eval-results", "metrics.json")

    baseline_results = load_json(baseline_metric_path)
    after_training_results = load_json(after_training_metric_path)

    benchmarks = ["aime24", "aime25"]

    print("[check] baseline file: {}".format(baseline_metric_path))
    print("[check] after training file: {}".format(after_training_metric_path))
    print("[check] metric: pass@1[avg-of-8].symbolic_correct")
    print("[check] rule: after training accuracy must be strictly greater than baseline accuracy")

    all_passed = True
    for benchmark in benchmarks:
        passed, lines = check_benchmark(benchmark, baseline_results, after_training_results)
        for ln in lines:
            print(ln)
        if not passed:
            all_passed = False

    if not all_passed:
        print("[check] FINAL RESULT: FAIL — one or more benchmarks did not meet the criteria.")
        sys.exit(1)
    else:
        print("[check] FINAL RESULT: OK — all benchmarks passed.")


if __name__ == "__main__":
    main()

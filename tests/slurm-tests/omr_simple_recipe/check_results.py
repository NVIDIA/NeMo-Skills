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

import argparse
import json
from pathlib import Path

# Hard-coded accuracy ranges for baseline and after-training results
RANGE_CONSTRAINTS = {
    "after_training": {
        "aime24": (20.0, 30.0),
        "aime25": (17.5, 27.5),
    },
    "baseline": {
        "aime24": (6.25, 16.25),
        "aime25": (8.75, 18.75),
    },
}


def load_json(path: Path):
    """Load a JSON file from the given path."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def get_aime_symbolic_avg8(d: dict, bench_key: str) -> float:
    """Extract the value of pass@1[avg-of-8].symbolic_correct and convert to float."""
    return float(d[bench_key]["pass@1[avg-of-8]"]["symbolic_correct"])


def in_range(value: float, lo: float, hi: float) -> bool:
    """Return True if value is within [lo, hi] inclusive."""
    return lo <= value <= hi


def check_benchmark(benchmark: str, baseline_results: dict, after_training_results: dict):
    """
    Validate one benchmark:
      - baseline accuracy must be within its allowed range
      - after-training accuracy must be within its allowed range
      - after-training accuracy must be strictly greater than baseline accuracy
    """
    baseline_acc = get_aime_symbolic_avg8(baseline_results, benchmark)
    after_acc = get_aime_symbolic_avg8(after_training_results, benchmark)

    lo_b, hi_b = RANGE_CONSTRAINTS["baseline"][benchmark]
    lo_a, hi_a = RANGE_CONSTRAINTS["after_training"][benchmark]

    assert in_range(baseline_acc, lo_b, hi_b), f"{benchmark}: baseline {baseline_acc}% out of range [{lo_b}%, {hi_b}%]"
    assert in_range(after_acc, lo_a, hi_a), f"{benchmark}: after_training {after_acc}% out of range [{lo_a}%, {hi_a}%]"
    assert after_acc > baseline_acc, (
        f"{benchmark}: after_training {after_acc}% not greater than baseline {baseline_acc}%"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Compare after-training vs baseline on AIME24/25 (metric: pass@1[avg-of-8].symbolic_correct)."
    )
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results.")
    args = ap.parse_args()

    for benchmark in ("aime24", "aime25"):
        common_path = Path(args.workspace) / "evals"
        baseline_results = load_json(common_path / "baseline" / "eval-results" / benchmark / "metrics.json")
        after_training_results = load_json(
            common_path / "after-training" / "eval-results" / benchmark / "metrics.json"
        )
        check_benchmark(benchmark, baseline_results, after_training_results)

    print("All checks passed.")


if __name__ == "__main__":
    main()

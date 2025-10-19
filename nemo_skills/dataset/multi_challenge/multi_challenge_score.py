#!/usr/bin/env python
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

"""
Score computation for MultiChallenge benchmark.
"""

import json
import sys
from pathlib import Path

from nemo_skills.evaluation.metrics.multi_challenge_metrics import (
    compute_metrics,
    format_metrics_report,
)


def collect_results_from_file(file_path):
    """Read and collect results from a JSONL file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def main(input_file, output_file=None):
    """Main function to compute and report scores.

    Args:
        input_file: Path to the evaluated results JSONL file
        output_file: Optional path to save the report
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Collect results
    results = collect_results_from_file(input_path)

    if not results:
        print(f"Error: No results found in {input_file}")
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(results)

    # Format report
    report = format_metrics_report(metrics)

    # Print to console
    print(report)

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nReport saved to: {output_file}")

        # Also save JSON metrics
        json_output = output_path.with_suffix('.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"JSON metrics saved to: {json_output}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multi_challenge_score.py <input_file> [output_file]")
        print("Example: python multi_challenge_score.py results.jsonl report.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    main(input_file, output_file)

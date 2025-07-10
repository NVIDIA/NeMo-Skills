# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_metric_files(metric_files: List[str]) -> Dict[str, Any]:
    """Load and combine multiple metric.json files into a single dictionary."""
    combined_metrics = {}

    for metric_file in metric_files:
        with open(metric_file, 'r') as f:
            metrics = json.load(f)
            combined_metrics.update(metrics)

    return combined_metrics


def import_score_module(score_module: str):
    """Dynamically import score module from module name or file path."""
    if score_module.endswith('.py') or '/' in score_module:
        # Treat as file path
        spec = importlib.util.spec_from_file_location("score_module", score_module)
        module = importlib.util.module_from_spec(spec)
        sys.modules["score_module"] = module
        spec.loader.exec_module(module)
        return module
    else:
        # Treat as module name
        return importlib.import_module(score_module)


def main():
    parser = argparse.ArgumentParser(description='Compute scores from metric files')
    parser.add_argument('metric_files', nargs='+', help='List of metric.json files to combine')
    parser.add_argument('--score_module', required=True, help='Score module name or path to python file')
    parser.add_argument('--save_metrics_file', required=True, help='Output file to save final metrics')

    args = parser.parse_args()

    # Load and combine metric files
    combined_metrics = load_metric_files(args.metric_files)

    # Import score module and compute score
    score_module = import_score_module(args.score_module)
    final_metrics = score_module.compute_score(combined_metrics)

    # Save final metrics
    with open(args.save_metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == '__main__':
    main()

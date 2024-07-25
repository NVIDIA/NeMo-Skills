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

# will run compute metrics on all relevant files and summarize results in a .csv file

import argparse
import glob
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))
sys.path.append(str(Path(__file__).absolute().parents[0]))

from compute_metrics import EVALUATOR_MAP, compute_metrics

from nemo_skills.evaluation.metrics import MathEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_folder',
        help="Path to the folder with results. Needs to contain <benchmark> folders inside.",
    )
    parser.add_argument(
        '--benchmarks',
        nargs="+",
        default=[],
        help="Specify benchmarks to run. If not specified, all benchmarks in the results_folder will be used.",
    )
    args = parser.parse_args()

    # running compute_metrics.py to get greedy, majority and pass @k results for all benchmarks available
    benchmarks = glob.glob(f'{args.results_folder}/*')

    if args.benchmarks:
        for benchmark in benchmarks.copy():
            if Path(benchmark).name not in args.benchmarks:
                benchmarks.remove(benchmark)

    current_folder = Path(__file__).absolute().parent
    results = defaultdict(dict)
    for benchmark_path in benchmarks:
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue
        try:
            evaluator = EVALUATOR_MAP.get(benchmark, MathEval)()
            if benchmark in ['human-eval', 'mbpp']:
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        prediction_jsonl_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        evaluator=evaluator,
                    )
                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        prediction_jsonl_files=sampling_outputs,
                        evaluator=evaluator,
                        aggregation_mode="best",
                    )
            else:
                results[benchmark] = {}
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        prediction_jsonl_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        evaluator=evaluator,
                    )

                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'majority@{len(sampling_outputs)}'] = compute_metrics(
                        prediction_jsonl_files=sampling_outputs,
                        evaluator=evaluator,
                        aggregation_mode="majority",
                    )
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        prediction_jsonl_files=sampling_outputs,
                        evaluator=evaluator,
                        aggregation_mode="best",
                    )
        except:
            print(f"Error running compute_metrics.py for {benchmark}: {e}")

    lines_to_write = []
    for benchmark, benchmark_results in results.items():
        max_widths = {}
        max_widths['evaluation_mode'] = len('evaluation_mode')
        for eval_mode, metrics in benchmark_results.items():
            for metric_key, metric_value in metrics.items():
                max_widths[metric_key] = max(
                    max_widths.get(metric_key, len(metric_key)),
                    len(f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)),
                )
            max_widths['evaluation_mode'] = max(max_widths['evaluation_mode'], len(eval_mode))

        total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
        print(f' {benchmark} '.center(total_width, '-'))
        headers = ['evaluation_mode'] + list(list(benchmark_results.values())[0].keys())
        print(' | '.join([f'{header:<{max_widths[header]}}' for header in headers]))

        for eval_mode, metrics in benchmark_results.items():
            values = [f'{eval_mode:<{max_widths["evaluation_mode"]}}']
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.2f}"
                values.append(f'{str(metric_value):<{max_widths[metric_key]}}')
            print(' | '.join(values))

        print('\n')

    # summarizing results in a .json file
    results = dict(results)
    with open(f'{args.results_folder}/results.json', 'wt', encoding="utf-8") as fout:
        json.dump(results, fout, indent=4)
    print(f"Summarized results are available in {args.results_folder}/results.json")

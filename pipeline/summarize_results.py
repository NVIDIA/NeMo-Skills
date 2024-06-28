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
import logging
import subprocess
import sys
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))
sys.path.append(str(Path(__file__).absolute().parents[0]))

from compute_metrics import compute_metrics

from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
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
    results = {}
    for benchmark_path in benchmarks:
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue
        try:
            if benchmark in ['human-eval', 'mbpp']:
                for suffix in ["", "-plus"]:
                    results[benchmark + suffix] = {}
                    if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                        correct_answer, wrong_answer, no_answer, total = compute_metrics(
                            prediction_jsonl_files=[f"{benchmark_path}/output-greedy.jsonl"],
                            is_correct_key=f'is_correct{suffix}',
                            eval_type="code",
                        )
                        results[benchmark + suffix]['greedy'] = {
                            "num_entries": total,
                            "correct_answer": correct_answer,
                            "wrong_answer": wrong_answer,
                            "no_answer": no_answer,
                        }
                    sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                    if len(sampling_outputs) > 0:
                        correct_answer, wrong_answer, no_answer, total = compute_metrics(
                            prediction_jsonl_files=sampling_outputs,
                            aggregation_mode="best",
                            is_correct_key=f'is_correct{suffix}',
                            eval_type="code",
                        )
                        results[benchmark + suffix][f'pass@{len(sampling_outputs)}'] = {
                            "num_entries": total,
                            "correct_answer": correct_answer,
                            "wrong_answer": wrong_answer,
                            "no_answer": no_answer,
                        }
            else:
                results[benchmark] = {}
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    correct_answer, wrong_answer, no_answer, total = compute_metrics(
                        prediction_jsonl_files=[f"{benchmark_path}/output-greedy.jsonl"],
                    )
                    results[benchmark]['greedy'] = {
                        "num_entries": total,
                        "correct_answer": correct_answer,
                        "wrong_answer": wrong_answer,
                        "no_answer": no_answer,
                    }

                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    correct_answer, wrong_answer, no_answer, total = compute_metrics(
                        prediction_jsonl_files=sampling_outputs,
                        aggregation_mode="majority",
                    )
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = {
                        "num_entries": total,
                        "correct_answer": correct_answer,
                        "wrong_answer": wrong_answer,
                        "no_answer": no_answer,
                    }
                    correct_answer, wrong_answer, no_answer, total = compute_metrics(
                        prediction_jsonl_files=sampling_outputs,
                        aggregation_mode="best",
                    )
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = {
                        "num_entries": total,
                        "correct_answer": correct_answer,
                        "wrong_answer": wrong_answer,
                        "no_answer": no_answer,
                    }

        except subprocess.CalledProcessError as e:
            LOG.error(f"Error running compute_metrics.py for {benchmark}: {e}")

    # summarizing results in a .csv file
    lines_to_write = []
    to_write = 'benchmark,decoding,num_entries,correct_answer,wrong_answer,no_answer'
    LOG.info(to_write)
    lines_to_write.append(to_write + '\n')
    for benchmark, benchmark_results in results.items():
        for decoding, decoding_results in benchmark_results.items():
            to_write = (
                f'{benchmark},{decoding},{decoding_results["num_entries"]},'
                f'{decoding_results["correct_answer"]:.2f},'
                f'{decoding_results["wrong_answer"]:.2f},'
                f'{decoding_results["no_answer"]:.2f}'
            )
            LOG.info(to_write)
            lines_to_write.append(to_write + '\n')

    with open(f'{args.results_folder}/results.csv', 'wt', encoding="utf-8") as fout:
        for line_to_write in lines_to_write:
            fout.write(line_to_write + '\n')
    LOG.info(f"Summarized results are available in {args.results_folder}/results.csv")

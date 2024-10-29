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

import glob
import importlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import typer

from nemo_skills.evaluation.metrics import MathMetrics
from nemo_skills.pipeline import (
    check_if_mounted,
    cluster_download,
    cluster_upload,
    get_cluster_config,
    get_tunnel,
    get_unmounted_path,
)
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.compute_metrics import compute_metrics
from nemo_skills.utils import setup_logging


@app.command()
@typer_unpacker
def summarize_results(
    results_dir: str = typer.Argument(
        ...,
        help="Path to the dir with results. Needs to contain <benchmark> dirs inside. "
        "If cluster is specified, will fetch the results from there.",
    ),
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument. "
        "If not specified, will assume the results are in the local filesystem.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    benchmarks: Optional[str] = typer.Option(
        None,
        help="Specify benchmarks to run (comma separated). "
        "If not specified, all benchmarks in the results_dir will be used.",
    ),
    remote_tar_dir: str = typer.Option(None, help="Directory where remote tar files are created on clusters"),
    debug: bool = typer.Option(False, help="Print debug information"),
    max_samples: int = typer.Option(-1, help="Limit metric computation only to first `max_samples`"),
):
    """Summarize results of an evaluation job."""
    setup_logging(disable_hydra_logs=False, log_level=logging.INFO if not debug else logging.DEBUG)

    if " " in str(benchmarks):
        raise ValueError("benchmarks should be separated with commas")

    cluster = cluster or os.environ.get("NEMO_SKILLS_CONFIG")

    # copying results from the cluster if necessary
    upload_path = None
    if cluster is not None:
        cluster_config = get_cluster_config(cluster, config_dir)
        check_if_mounted(cluster_config, results_dir)
    if cluster == "local":
        results_dir = get_unmounted_path(cluster_config, results_dir)
    elif cluster is not None:
        upload_path = results_dir
        tunnel = get_tunnel(cluster_config)
        temp_dir = tempfile.mkdtemp()
        print(f"Copying results from {results_dir} on cluster {cluster} to {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
        cluster_download(
            tunnel,
            get_unmounted_path(cluster_config, results_dir),
            temp_dir,
            remote_tar_dir=get_unmounted_path(cluster_config, remote_tar_dir),
        )
        results_dir = Path(temp_dir) / Path(results_dir).name

    # running compute_metrics.py to get greedy, majority and pass @k results for all benchmarks available
    # Check if there is an eval-results dir inside the results_dir
    eval_results_dir = Path(results_dir) / 'eval-results'
    if eval_results_dir.exists() and eval_results_dir.is_dir():
        results_dir = eval_results_dir
    benchmarks_paths = glob.glob(f'{results_dir}/*')

    if benchmarks:
        benchmarks_paths = [b for b in benchmarks_paths if Path(b).name in benchmarks.split(",")]

    results = defaultdict(dict)
    for benchmark_path in benchmarks_paths:
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue
        try:
            benchmark_module = importlib.import_module(f"nemo_skills.dataset.{benchmark}")
            metrics_calculator = benchmark_module.METRICS_CLASS()
            results[benchmark] = {}
            # TODO: we should just return all available aggregations from compute_metrics directly
            if not isinstance(metrics_calculator, MathMetrics):
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        metrics_calculator=metrics_calculator,
                        max_samples=max_samples,
                    )
                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="best",
                        max_samples=max_samples,
                    )
            else:
                if Path(f'{benchmark_path}/output-greedy.jsonl').exists():
                    results[benchmark]['greedy'] = compute_metrics(
                        input_files=[f"{benchmark_path}/output-greedy.jsonl"],
                        metrics_calculator=metrics_calculator,
                        max_samples=max_samples,
                    )

                sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
                if len(sampling_outputs) > 0:
                    results[benchmark][f'majority@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="majority",
                        max_samples=max_samples,
                    )
                    results[benchmark][f'pass@{len(sampling_outputs)}'] = compute_metrics(
                        input_files=sampling_outputs,
                        metrics_calculator=metrics_calculator,
                        aggregation_mode="best",
                        max_samples=max_samples,
                    )
        except Exception as e:
            print(f"Error running compute_metrics.py for {benchmark}: {e}")

    for benchmark, benchmark_results in results.items():
        if not benchmark_results:
            continue
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

    with open(Path(results_dir) / 'metrics.json', 'wt', encoding='utf-8') as fout:
        json.dump(results, fout, indent=2)

    if upload_path is not None:
        cluster_upload(
            tunnel,
            Path(results_dir) / 'metrics.json',
            Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json',
        )
        print("Metrics are saved to", str(Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json'))
        tunnel.cleanup()
    else:
        print("Metrics are saved to", str(Path(results_dir) / 'metrics.json'))


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

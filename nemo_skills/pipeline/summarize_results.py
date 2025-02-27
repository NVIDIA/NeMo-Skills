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
import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.pipeline import (
    check_if_mounted,
    cluster_download,
    cluster_upload,
    get_cluster_config,
    get_tunnel,
    get_unmounted_path,
)
from nemo_skills.pipeline.app import app, typer_unpacker
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
    extra_datasets: str = typer.Option(
        None,
        help="Path to a custom dataset folder that will be searched in addition to the main one. "
        "Can also specify through NEMO_SKILLS_EXTRA_DATASETS.",
    ),
    metric_type: Optional[str] = typer.Option(
        None,
        help="Specify metric type to use a specific metric calculator.",
    ),
    verbose: bool = typer.Option(True, help="Print download/upload progress"),
    wandb_name: Optional[str] = typer.Option(None, help="Name of the wandb experiment to sync these results to"),
    wandb_group: Optional[str] = typer.Option(None, help="Name of the wandb group to sync these results to"),
    wandb_project: Optional[str] = typer.Option('nemo-skills', help="Name of the wandb project"),
):
    """Summarize results of an evaluation job."""
    setup_logging(disable_hydra_logs=False, log_level=logging.WARNING if not debug else logging.DEBUG)

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
            verbose=verbose,
        )
        results_dir = Path(temp_dir) / Path(results_dir).name

    # running compute_metrics.py to get greedy, majority and pass @k results for all benchmarks available

    # Check for all possible directory structures
    # 1. {results_dir}/eval-results/{benchmark}/output*jsonl
    # 2. {results_dir}/{benchmark}/output*jsonl
    # 3. {results_dir}/output*jsonl

    # List to store all valid benchmarks results paths
    benchmarks_paths = []

    # Check for Option 3 - Root directory corresponds to a benchmark
    if Path(results_dir).is_dir() and len(glob.glob(f'{results_dir}/output*jsonl')) > 0:
        benchmarks_paths = [results_dir]
    else:
        cand_results_dir = Path(results_dir) / 'eval-results'
        # Check for Option 1
        if cand_results_dir.exists() and cand_results_dir.is_dir():
            results_dir = cand_results_dir
        else:
            # Assume by default it's Option 2.
            # Verify if it indeed has this structure: {results_dir}/{benchmark}/output*jsonl
            if len(glob.glob(f'{results_dir}/*/output*jsonl')) == 0:
                raise ValueError(
                    f"The results directory {results_dir} does not contain any valid eval-results or output*jsonl files."
                )

        benchmarks_paths = [
            cand_path
            for cand_path in glob.glob(f'{results_dir}/*')
            if '-logs' not in os.path.basename(cand_path) and Path(cand_path).is_dir()
        ]

    if benchmarks:
        # Filter benchmarks_paths to only include the specified benchmarks
        benchmarks_paths = [b for b in benchmarks_paths if Path(b).name in benchmarks.split(",")]

    if benchmarks_paths:
        # Ascertain that the benchmarks_paths are valid
        for benchmark_path in benchmarks_paths:
            # Valid benchmark_path should contain output*jsonl files
            if len(glob.glob(f'{benchmark_path}/output*jsonl')) == 0:
                raise ValueError(f"The benchmark directory {benchmark_path} lacks output*jsonl files.")
    else:
        print(f"No benchmarks found in {results_dir}")
        return

    results = defaultdict(lambda: defaultdict(dict))
    max_metrics_to_print = {}
    max_aggregations_to_print = {}
    for benchmark_path in benchmarks_paths:
        benchmark = str(Path(benchmark_path).name)
        if not Path(benchmark_path).is_dir():
            continue
        try:
            if metric_type is not None:
                metrics_calculator = ComputeMetrics(benchmark, metric_type=metric_type, max_samples=max_samples)
            else:
                metrics_calculator = ComputeMetrics(benchmark, extra_datasets=extra_datasets, max_samples=max_samples)

            metrics = {}
            # TODO: this is hacky, basically just assuming that if there is a greedy prediction, we need to add
            #       an extra aggregation to print
            has_greedy = False

            if Path(f'{benchmark_path}/output.jsonl').exists():
                has_greedy = True
                metrics = metrics_calculator.compute_metrics(input_files=[f"{benchmark_path}/output.jsonl"])
                if len(metrics) > 1:  # has subsets
                    for subset, subset_metrics in metrics.items():
                        results[f"{benchmark}-{subset}"].update(subset_metrics)
                else:
                    results[benchmark].update(metrics['all'])

            sampling_outputs = glob.glob(f'{benchmark_path}/output-rs*.jsonl')
            if len(sampling_outputs) > 0:
                metrics = metrics_calculator.compute_metrics(input_files=sampling_outputs)
                if len(metrics) > 1:  # has subsets
                    for subset, subset_metrics in metrics.items():
                        results[f"{benchmark}-{subset}"].update(subset_metrics)
                else:
                    results[benchmark].update(metrics['all'])

            if len(metrics) > 1:
                for subset, subset_metrics in metrics.items():
                    max_metrics_to_print[f"{benchmark}-{subset}"] = metrics_calculator.max_metrics_to_print()
                    max_aggregations_to_print[f"{benchmark}-{subset}"] = metrics_calculator.max_aggregations_to_print()
                    if max_aggregations_to_print[f"{benchmark}-{subset}"] is not None:
                        max_aggregations_to_print[f"{benchmark}-{subset}"] += has_greedy
            else:
                max_metrics_to_print[benchmark] = metrics_calculator.max_metrics_to_print()
                max_aggregations_to_print[benchmark] = metrics_calculator.max_aggregations_to_print()
                if max_aggregations_to_print[benchmark] is not None:
                    max_aggregations_to_print[benchmark] += has_greedy

        except Exception as e:
            logging.exception(f"Error computing metrics for {benchmark}: {e}")

    for benchmark, benchmark_results in results.items():
        if not benchmark_results:
            continue
        max_widths = {}
        max_widths['evaluation_mode'] = len('evaluation_mode')
        for eval_mode, metrics in list(benchmark_results.items())[: max_aggregations_to_print[benchmark]]:
            if max_metrics_to_print[benchmark] is None:
                max_metrics_to_print[benchmark] = len(metrics)
            for metric_key, metric_value in list(metrics.items())[: max_metrics_to_print[benchmark]]:
                max_widths[metric_key] = max(
                    max_widths.get(metric_key, len(metric_key)),
                    len(f"{metric_value:.2f}" if isinstance(metric_value, float) else str(metric_value)),
                )
            max_widths['evaluation_mode'] = max(max_widths['evaluation_mode'], len(eval_mode))

        total_width = sum(max_widths.values()) + (len(max_widths) - 1) * 3
        print(f' {benchmark} '.center(total_width, '-'))
        headers = ['evaluation_mode'] + list(list(benchmark_results.values())[0].keys())[
            : max_metrics_to_print[benchmark]
        ]
        print(' | '.join([f'{header:<{max_widths[header]}}' for header in headers]))

        for eval_mode, metrics in list(benchmark_results.items())[: max_aggregations_to_print[benchmark]]:
            values = [f'{eval_mode:<{max_widths["evaluation_mode"]}}']
            for metric_key, metric_value in list(metrics.items())[: max_metrics_to_print[benchmark]]:
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.2f}%"
                values.append(f'{str(metric_value):<{max_widths[metric_key]}}')
            print(' | '.join(values))

        print('\n')

    try:
        with open(Path(results_dir) / 'metrics.json', 'wt', encoding='utf-8') as fout:
            json.dump(results, fout, indent=2)
        if upload_path is not None:
            cluster_upload(
                tunnel,
                Path(results_dir) / 'metrics.json',
                Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json',
                verbose=verbose,
            )
            print("Metrics are saved to", str(Path(get_unmounted_path(cluster_config, upload_path)) / 'metrics.json'))
            tunnel.cleanup()
        else:
            print("Metrics are saved to", str(Path(results_dir) / 'metrics.json'))
    except PermissionError:
        print(f"Could not save metrics.json to {Path(results_dir) / 'metrics.json'}. Please check the permissions.")

    # syncing to wandb if asked
    if wandb_name is not None:
        import wandb

        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            id=wandb_name + (wandb_group if wandb_group else ""),
            resume="allow",
            group=wandb_group,
            settings=wandb.Settings(silent=True),
        )
        plots = {}

        for benchmark, benchmark_results in results.items():
            if not benchmark_results:
                continue

            # Store @k metrics separately for plotting
            k_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            for eval_mode, metrics in benchmark_results.items():
                # Check if this is a @k metric
                k_match = re.search(r'@(\d+)$', eval_mode)
                if k_match:
                    k = int(k_match.group(1))
                    base_name = eval_mode.rsplit('@', 1)[0]

                # Store k and corresponding values for each metric, but log everything
                for metric_key, metric_value in metrics.items():
                    if k_match and metric_key != "num_entries":
                        k_metrics[metric_key][base_name]["k"].append(k)
                        k_metrics[metric_key][base_name]["value"].append(metric_value)

                    run.log({f"{benchmark}/{eval_mode}/{metric_key}": metric_value})

            # Create combined plot per metric key (line series)
            for metric_key, eval_modes in k_metrics.items():
                metric_xs = []
                metric_ys = []
                mode_keys = []

                # Sort by k values and get all evaluation modes for this metric
                for mode_name, values in eval_modes.items():
                    k_value_pairs = sorted(zip(values["k"], values["value"]))
                    k_values, metric_values = zip(*k_value_pairs)
                    metric_xs.append(k_values)
                    metric_ys.append(metric_values)
                    mode_keys.append(mode_name)

                # a few hardcoded metrics to ignore
                to_ignore = ["no_answer", "any_correct", "both_correct"]
                if metric_key in to_ignore:
                    continue

                plot_key = f"{benchmark}/{metric_key}"
                plots[plot_key] = wandb.plot.line_series(
                    xs=metric_xs,
                    ys=metric_ys,
                    keys=mode_keys,
                    title=f"{benchmark} - {metric_key}",
                    xname="number of samples",
                )

                # Create individual plots for each evaluation mode
                for mode_name, values in eval_modes.items():
                    k_value_pairs = sorted(zip(values["k"], values["value"]))
                    k_values, metric_values = zip(*k_value_pairs)

                    plot_data = [[x, y] for x, y in zip(k_values, metric_values)]
                    table = wandb.Table(data=plot_data, columns=["k", "value"])

                    plot_key = f"{benchmark}/{metric_key}/{mode_name}"
                    plots[plot_key] = wandb.plot.line(
                        table,
                        "k",
                        "value",
                        title=f"{benchmark} - {metric_key} - {mode_name}",
                    )

        # Log all plots
        run.log({**plots})
        run.finish()
        print(
            f"Results are synced to wandb project {wandb_project} under the name {wandb_name} and group {wandb_group}"
        )


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

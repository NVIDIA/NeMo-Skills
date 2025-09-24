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
import logging
from pathlib import Path
import glob
import yaml
import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app
from nemo_skills.pipeline.eval import eval
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def robust_eval(ctx: typer.Context,
                prompt_set_config: str = typer.Option(..., help="Path to a yaml file containting list of prompts per benchmark"),
                **ns_eval_kwargs):
    """Run evaluation on multiple prompts and benchmarks to measure LLM robustness against prompt.
    Code expects a yaml file with the following structure: (example in /nemo_skills/prompt/config/robustness/prompt_set_config.yaml)
    ```yaml
    <benchmark_name>:
      - <path_to_prompt_1>
      - <path_to_prompt_2>
      ...
    <another_benchmark_name>:
        - <path_to_prompt_1>
        - <path_to_prompt_2>
        ...
    All other arguments are 'ns eval arguments'.
    robust_eval will run 'ns eval' for each prompt and benchmark combination and then summarize the results in the output_dir folder.
    """
    prompt_set_config = yaml.safe_load(open(prompt_set_config))
    benchmarks = ns_eval_kwargs['benchmarks'].split(',')
    if set(prompt_set_config.keys()) != set([b.split(':')[0] for b in benchmarks]):
        raise ValueError(f"Benchmark names ({benchmarks}) must match prompt set config({prompt_set_config.keys()})")
    main_output_dir = ns_eval_kwargs['output_dir']
    main_expname = ns_eval_kwargs['expname']
    dependent_tasks = []
    for benchmark in benchmarks:
        ns_eval_kwargs['benchmarks'] = benchmark
        benchmark = benchmark.split(':')[0]  # Remove any :N suffix for output dir naming
        LOG.info(f"Running {len(prompt_set_config[benchmark])} prompts on {benchmark}")
        for prompt in prompt_set_config[benchmark]:
            LOG.info(f"Running prompt: {prompt}")
            ctx.args = [arg for arg in ctx.args if "++prompt_config" not in arg]
            ctx.args.append(f"++prompt_config={prompt}")
            ns_eval_kwargs['output_dir'] = f"{main_output_dir}/{benchmark}/{Path(prompt).stem}"
            ns_eval_kwargs['expname'] = main_expname + f"_{benchmark}_{Path(prompt).stem}"
            eval(ctx=ctx, **ns_eval_kwargs)
            dependent_tasks.append(ns_eval_kwargs['expname'])

    cluster_config = pipeline_utils.get_cluster_config(ns_eval_kwargs['cluster'], ns_eval_kwargs.get('config_dir', None))

    _reuse_exp = ns_eval_kwargs.get('_reuse_exp', None)
    dry_run = ns_eval_kwargs.get('dry_run', False)
    with pipeline_utils.get_exp(ns_eval_kwargs['expname'], cluster_config, _reuse_exp) as exp:
        command = (
                f"python -m nemo_skills.pipeline.summarize_robustness {main_output_dir}"
            )
        summ_task = pipeline_utils.add_task(
            exp,
            cmd=command,
            task_name=f"summ_robust",
            log_dir=f"{main_output_dir}/summarize_robustness",
            container=cluster_config["containers"]["nemo-skills"],
            partition='cpu',
            cluster_config=cluster_config,
            reuse_code_exp=ns_eval_kwargs.get('reuse_code_exp', None),
            reuse_code=ns_eval_kwargs.get('reuse_code', True),
            run_after=dependent_tasks,
            installation_command=ns_eval_kwargs.get('installation_command', None),
            skip_hf_home_check=ns_eval_kwargs.get('skip_hf_home_check', False),
        )

        pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)

if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()

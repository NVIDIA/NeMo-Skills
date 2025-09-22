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

import typer

from nemo_skills.pipeline.app import app
from nemo_skills.pipeline.eval import eval
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def robust_eval(ctx: typer.Context,
                prompts_folder: str = typer.Option(..., help="Path to the folder inside /prompt/config/ containing the prompt yaml files to run."),
                **ns_eval_kwargs):
    prompts_dir = f"nemo_skills/prompt/config/{prompts_folder}"
    prompts_to_run = glob.glob(f"{prompts_dir}/*.yaml")
    if not prompts_to_run:
        raise ValueError(f"No prompt .yaml files found in the specified directory: {prompts_dir}")
    LOG.info(f"Found {len(prompts_to_run)} prompts to run in {prompts_dir}")
    main_output_dir = Path(ns_eval_kwargs['output_dir'])
    benchmark = ns_eval_kwargs['benchmarks'].split(':')[0] # check that there is only one benchmark
    for prompt in prompts_to_run:
        LOG.info(f"Running prompt: {prompt}")
        ctx.args = [arg for arg in ctx.args if "++prompt_config" not in arg]
        ctx.args.append(f"++prompt_config={prompt}")
        ns_eval_kwargs['output_dir'] = str(main_output_dir / benchmark / Path(prompt).stem[:-5])
        eval(ctx=ctx, **ns_eval_kwargs)

if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()

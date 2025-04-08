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

import os
from pathlib import Path

import typer

from nemo_skills.pipeline.app import app


@app.command()
def setup():
    """Helper command to setup cluster configs."""
    typer.echo(
        "Let's set up your cluster configs! It's called 'cluster', but you need one to run things locally as well.\n"
        "The configs are just yaml files that you can later inspect and modify.\n"
        "They are mostly there to let us know which containers to use and how to orchestrate jobs on slurm."
    )

    # Get the directory for cluster configs with default as current dir / cluster_configs
    default_dir = os.path.join(os.getcwd(), "cluster_configs")
    config_dir = typer.prompt("Where would you like to store your cluster configs?", default=default_dir)

    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Ask the user if they want a local or slurm config
    config_type = typer.prompt("What type of config would you like to create? (local/slurm)").lower()

    # Ask for the name of the config file
    default_name = f"{config_type}"
    config_name = typer.prompt(
        f"What would you like to name your {config_type} config file? "
        "You'd need to use that name as --cluster argument to ns commands.",
        default=default_name,
    )

    # # Create the config file
    # config_file = config_dir / config_name
    # config_file.touch(exist_ok=True)
    # typer.echo(f"Created {config_type} config file at {config_file}")

    typer.echo(
        f"Great, you're all done! It might be a good idea to define "
        f"NEMO_SKILLS_CONFIGS={config_dir}, so that configs are always found."
    )


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()

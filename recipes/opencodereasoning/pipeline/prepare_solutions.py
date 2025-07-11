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
from pathlib import Path

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments


def get_stage_expname(base_expname, stage_name, suffix):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def generate_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Extracts potential problems from raw text data."""
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    language = stage_config.get("language", "python")
    assert language in ["python", "cpp"], f"Unsupported language: {language}"

    generate(
        ctx=wrap_arguments(
            f"++prompt_config=/nemo_run/code/recipes/opencodereasoning/prompts/generate_{language}_soln.yaml "
            f"{stage_config.get('inline_args', '')} "
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname,
        run_after=run_after,
        **stage_config.get('stage_kwargs', {}),
    )


def filter_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Classifies extracted problems into different types (proof, mcq, binary, invalid)."""
    input_dir = stage_config["input_dir"]
    output_dir = stage_config["output_dir"]

    language = stage_config['language']
    assert language in ["python", "cpp"], f"Unsupported language: {language}"

    filter_expname = f"{expname}-filter"

    command = (
        f"cd /nemo_run/code/recipes/opencodereasoning/scripts && "
        f" python functional_helpers.py filter_code_samples "
        f"   --data_path='{input_dir}/*.json*' "
        f"   --output_dir={output_dir}/generation_filtered_temp "
        f"   --output_filename='code' "
        f"   --do_ast_check={True if language == 'python' else False} && "
        f" python functional_helpers.py filter_invalid_samples "
        f"   --data_path='{output_dir}/generation_filtered_temp/*.json*' "
        f"   --output_dir={output_dir}/generation_filtered "
        f"   --output_filename='filtered' && "
        f" python functional_helpers.py rename_files_to_json "
        f"   --data_path='{output_dir}/generation_filtered/*.json*' && "
        f" mkdir -p {output_dir}/dataset && "
        f" cat {output_dir}/generation_filtered/*.json* > {output_dir}/dataset/open_code_reasoning.jsonl && "
        f" rm -rf {output_dir}/generation_filtered_temp"
    )

    run_cmd(
        ctx=wrap_arguments(""),
        cluster=cluster,
        command=command,
        expname=filter_expname,
        log_dir=f'{output_dir}/filter-logs',
        run_after=run_after,
        **stage_config.get('stage_kwargs', {}),
    )


stages_map = {
    'generate_solutions': generate_solutions,
    'filter_solutions': filter_solutions,
}


def get_available_configs(config_dir):
    """Get available YAML configuration files from the config directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []
    yaml_files = list(config_dir.glob("*.yaml"))
    config_names = [file.stem for file in yaml_files if not file.name.startswith('template')]
    return config_names


if __name__ == '__main__':
    config_dir = Path(__file__).parents[1] / "configs" / "solution_sdg"
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description='OpenCodeReasoning-2 solution generation pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=available_configs,
        help="Will pick a corresponding config from configs folder",
    )
    parser.add_argument(
        '--stages',
        type=str,
        default=None,
        help='Comma-separated list of stages to run. If not specified, runs all stages from the config.',
    )

    args = parser.parse_args()

    config_path = config_dir / f"{args.mode}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if 'pipeline_stages' not in config or not config['pipeline_stages']:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config['pipeline_stages']

    if args.stages:
        # Stages specified via command line
        stages_to_run = args.stages.split(',')
        print(f"Running specified stages: {stages_to_run}")
    else:
        # No command line override, run all stages from config
        stages_to_run = full_stage_sequence
        print(f"Running all stages defined in config for mode '{args.mode}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage specified: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' requested but not part of the defined sequence for mode '{args.mode}' in {config_path}. "
                f"Specify one of {full_stage_sequence} or select an appropriate mode."
            )

    # --- Common parameters ---
    base_output_dir = config['base_output_dir']
    suffix = config.get('suffix', args.mode)
    cluster = config['cluster']
    expname_base = config['expname']

    # --- Run selected stages ---
    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get('stages', {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get('dependencies', None)
        dependencies = None
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep_stage, suffix) for dep_stage in dep_stages]

        print(f"Dependency for '{stage}': {dependencies}")

        stage_args = {
            'cluster': cluster,
            'expname': current_expname,
            'run_after': dependencies,
            'stage_config': stage_config,
        }

        # Call the stage function
        stage_func(**stage_args)

    print("\n--- Selected pipeline stages finished. ---")

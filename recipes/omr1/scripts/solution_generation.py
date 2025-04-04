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

import yaml

from nemo_skills.pipeline import generate, run_cmd, wrap_arguments


def generate_solutions(input_file, output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-merge-data"  # this is launched in problem_generation.py

    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=generic/math "
            f"++inference.temperature=0.7 "
            f"{extra_args} "
        ),
        cluster=cluster,
        output_dir=f"{output_dir}/generate-solutions",
        expname=f"{expname}-generate-solutions",
        run_after=run_after,
        **generate_kwargs,
    )


def fill_majority_answer(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-generate-solutions"

    cmd = (
        f'python -m nemo_skills.evaluation.aggregate_answers '
        f'    ++input_dir={output_dir}/generate-solutions '
        f'    ++output_dir={output_dir}/filled-majority '
        f'    ++input_files="output-rs*.jsonl" '
        f'    ++mode=fill '
        f'    ++fill_is_correct=False '
        f'    ++ignore_if_not_none=True '
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        partition="cpu",  # change that if not available (ignored if running locally)
        log_dir=f"{output_dir}/filled-majority/logs",
        expname=f"{expname}-fill-majority",
        run_after=run_after,
    )


def judge_answers(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-fill-majority"

    generate(
        ctx=wrap_arguments(f"++input_dir={output_dir}/filled-majority {extra_args} "),
        cluster=cluster,
        generation_type="math_judge",
        output_dir=f"{output_dir}/judged-generations",
        expname=f"{expname}-judge-answers",
        run_after=run_after,
        **generate_kwargs,
    )


def prepare_for_sft(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-judge-answers"

    cmd = (
        f"python -m nemo_skills.training.prepare_data "
        f"    ++input_files='{output_dir}/judged-generations/output-rs*.jsonl' "
        f"    ++output_path={output_dir}/sft-data.jsonl "
        f"    ++prompt_config=generic/math "  # can remove if not needed
        f"    ++prompt_template=qwen-instruct "  # can remove if not needed
        f"    ++filters.drop_multi_boxed=false "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.remove_len_outlier_solutions=false "
        f"    ++use_judgement=true "
        f"    ++contamination_file={output_dir}/contamination-labeled.jsonl "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        partition="cpu",  # change that if not available (ignored if running locally)
        log_dir=f"{output_dir}/prepare_for_sft/logs",
        expname=f"{expname}-prepare-for-sft",
        run_after=run_after,
    )


stages_map = {
    'generate_solutions': generate_solutions,
    'fill_majority_answer': fill_majority_answer,
    'judge_answers': judge_answers,
    # TODO: add summary regeneration step
    'prepare_for_sft': prepare_for_sft,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenMathReasoning-1 solution generation pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['demo', 'full'],
        help="Will pick a corresponding config from configs folder",
    )
    parser.add_argument(
        '--stages',
        type=str,
        help='Pipeline stages to run. '
        'Can specify all, or any subset of stages (comma-separated if want to run multiple)',
        default='all',
    )

    args = parser.parse_args()

    with open(f'{Path(__file__).parents[1]}/configs/{args.mode}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.stages == 'all':
        stages = list(stages_map.keys())
    elif ',' in args.stages:
        stages = args.stages.split(',')
    else:
        stages = [args.stages]

    for stage in stages:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {list(stages_map.keys())}")

    input_file = config['solution_sdg']['input_file'].format(output_dir=config['output_dir'])
    default_args = dict(
        output_dir=config['output_dir'],
        cluster=config['cluster'],
        expname=config['expname'],
        **config['solution_sdg']['generation'],
    )

    for stage in stages:
        stage_args = default_args.copy()
        if stage == 'generate_solutions':
            stage_args['input_file'] = input_file
        if stage == 'judge_answers':  # using different generation args here
            stage_args.update(config['solution_sdg']['judge'])
        stages_map[stage](**stage_args)

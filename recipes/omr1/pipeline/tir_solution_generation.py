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

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments


def generate_solutions(input_file, output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-merge-data"  # this is launched in problem_generation.py

    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=generic/math "
            f"++inference.temperature=0.7 "
            f"{extra_args} "
        ),
        cluster=cluster,
        output_dir=f"{output_dir}/generate-solutions-{suffix}",
        expname=f"{expname}-generate-solutions-{suffix}",
        run_after=run_after,
        **generate_kwargs,
    )


def fill_majority_answer(output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-generate-solutions-{suffix}"

    cmd = (
        f'python -m nemo_skills.evaluation.aggregate_answers '
        f'    ++input_dir={output_dir}/generate-solutions-{suffix} '
        f'    ++output_dir={output_dir}/filled-majority-{suffix} '
        f'    ++input_files="output-rs*.jsonl" '
        f'    ++mode=fill '
        f'    ++fill_is_correct=False '
        f'    ++ignore_if_not_none=True '
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        partition="cpu",  # change that if not available (ignored if running locally)
        log_dir=f"{output_dir}/filled-majority-{suffix}/logs",
        expname=f"{expname}-fill-majority-{suffix}",
        run_after=run_after,
    )


def judge_answers(output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-fill-majority-{suffix}"

    generate(
        ctx=wrap_arguments(f"++input_dir={output_dir}/filled-majority-{suffix} {extra_args} "),
        cluster=cluster,
        generation_type="math_judge",
        output_dir=f"{output_dir}/judged-generations-{suffix}",
        expname=f"{expname}-judge-answers-{suffix}",
        run_after=run_after,
        **generate_kwargs,
    )


def preprocess_tir_generations(output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    cmd = (
        f"python /nemo_run/code/nemo_skills/recipes/omr1/scripts/preprocess_tir_generations.py "
        f"--input_files '{output_dir}/judged-generations-{suffix}/output-rs*.jsonl' "
        f"--output_file {output_dir}/tir-preprocess-generations-{suffix}/preprocessed_output.jsonl "
        f"--code_begin '```python' "
        f"--code_end '```' "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        partition="cpu",
        log_dir=f"{output_dir}/tir-preprocess-generations-{suffix}/logs",
        exclusive=True,
        run_after=f"{expname}-judge-answers-{suffix}",
        expname=f"{expname}-tir-preprocess-generations-{suffix}",
    )


def filter_novelty_significance(output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-fill-majority-{suffix}"
    fragments_file = f"{output_dir}/tir-code-fragments-{suffix}/output_fragments.jsonl"
    novelty_output_dir = f"{output_dir}/tir-filter-novelty-significance-{suffix}/novelty_judges/"
    significance_output_dir = f"{output_dir}/tir-filter-novelty-significance-{suffix}/significance_judges/"
    dependecies = []


    extraction_cmd = (
        f"python /nemo_run/code/nemo_skills/recipes/omr1/scripts/extract_python_fragments.py "
        f"--input_file={output_dir}/tir-preprocess-generations-{suffix}/preprocessed_output.jsonl "
        f"--output_file={fragments_file} "
        f"--window_size=1500"
    )
    
    expname = f"{expname}-tir-extract-python-fragments-{suffix}"
    run_cmd(
        ctx=wrap_arguments(extraction_cmd),
        cluster=cluster,
        partition="cpu",
        run_after=run_after,
        log_dir=f"{output_dir}/tir-filter-novelty-significance-{suffix}/logs",
        exclusive=True,
        expname=expname,
    )
    run_after = expname

    expname = f"{expname}-tir-judge-novelty-{suffix}"
    generate(
        ctx=wrap_arguments(
            f"++prompt_template=qwen-instruct "
            f"++prompt_config=/nemo_run/code/nemo_skills/recipes/omr1/prompts/classify-tir-novelty.yaml "
            f"++batch_size=512 "
            f"++input_file={fragments_file} "
            f"++generation_key=fragment_novelty "
            f"++skip_filled=True "
        ),
        cluster=cluster,
        output_dir=novelty_output_dir,
        run_after=run_after,
        model="/trt_models/qwen2.5-32b-instruct",
        server_type="trtllm",
        server_gpus=8,
        server_nodes=1,
        num_random_seeds=8,
        num_chunks=4,
        expname=expname,
    )
    run_after = expname
    dependecies.append(expname)
    
    expname = f"{expname}-tir-judge-significance-{suffix}"
    generate(
        ctx=wrap_arguments(
            f"++prompt_template=qwen-instruct "
            f"++prompt_config=/nemo_run/code/nemo_skills/recipes/omr1/prompts/classify-tir-significance.yaml "
            f"++batch_size=512 "
            f"++input_file={fragments_file} "
            f"++generation_key=fragment_significance "
            f"++skip_filled=True "
        ),
        cluster=cluster,
        output_dir=significance_output_dir,
        run_after=run_after,
        model="/trt_models/qwen2.5-32b-instruct",
        server_type="trtllm",
        server_gpus=8,
        server_nodes=1,
        num_random_seeds=8,
        num_chunks=4,
        expname=expname,
    )
    run_after = expname
    dependecies.append(expname)
    
    expname = f"{expname}-tir-filter-fragments-{suffix}"
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/nemo_skills/recipes/omr1/scripts/filter_novelty_significance.py "
            f"--novelty_files '{novelty_output_dir}/output-rs*.jsonl' "
            f"--significance_files '{significance_output_dir}/output-rs*.jsonl' "
            f"--output_file {output_dir}/filter-novelty-significance-{suffix}/filtered_output.jsonl "
        ),
        cluster=cluster,
        partition="cpu",
        log_dir=f"{output_dir}/tir-filter-novelty-significance-{suffix}/logs",
        exclusive=True,
        run_after=dependecies, # run after novelty and significance judges
        expname=expname,
    )


def prepare_for_sft(output_dir, suffix, cluster, expname, extra_args="", **generate_kwargs):
    # TODO:
    # implement all filters from my pipeline
    # filter number of code blocks
    # filter answers before code block
    # filter matplotlib
    run_after = f"{expname}-judge-answers-{suffix}"

    cmd = (
        f"python -m nemo_skills.training.prepare_data "
        f"    ++input_files='{output_dir}/judged-generations-{suffix}/output-rs*.jsonl' "
        f"    ++output_path={output_dir}/sft-data-{suffix}.jsonl "
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
        log_dir=f"{output_dir}/prepare_for_sft-{suffix}/logs",
        expname=f"{expname}-prepare-for-sft-{suffix}",
        run_after=run_after,
    )


stages_map = {
    'generate_solutions': generate_solutions,
    'fill_majority_answer': fill_majority_answer,
    'judge_answers': judge_answers,
    'preprocess_tir_generations': preprocess_tir_generations,
    'filter_novelty_significance': filter_novelty_significance,
    # TODO: add summary regeneration step
    'prepare_for_sft': prepare_for_sft,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenMathReasoning-1 solution generation pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['full-tir-stage-0', 'full-tir-stage-1'],  # TODO: add tir mode, remove other parts
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
        suffix=config['solution_sdg']['suffix'],
        cluster=config['cluster'],
        expname=config['expname'],
        **config['solution_sdg']['generation'],
    )

    for stage in stages:
        stage_args = default_args.copy()
        if stage == 'generate_solutions':
            stage_args['input_file'] = input_file
        if stage == 'judge_answers':  # using different generation args here
            stage_args = dict(
                output_dir=config['output_dir'],
                suffix=config['solution_sdg']['suffix'],
                cluster=config['cluster'],
                expname=config['expname'],
                **config['solution_sdg']['judge'],
            )
        stages_map[stage](**stage_args)

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

from nemo_skills.pipeline import check_contamination, convert, eval, generate, run_cmd, train, wrap_arguments


def extract_problems(input_file, output_dir, cluster, expname, extra_args="", **generate_kwargs):
    postprocess_cmd = (
        f"python /nemo_run/code/recipes/omr1/scripts/postprocess_problem_extraction.py "
        f"    {output_dir}/extract-problems/output.jsonl "
        f"    {output_dir}/extract-problems/extracted-problems.jsonl "
    )
    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=/nemo_run/code/recipes/omr1/prompts/extract-problems.yaml "
            f"{extra_args} "
        ),
        cluster=cluster,
        output_dir=f"{output_dir}/extract-problems",
        postprocess_cmd=postprocess_cmd,
        expname=f"{expname}-extract-problems",
        **generate_kwargs,
    )


def classify_problems(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-extract-problems"
    input_file = f"{output_dir}/extract-problems/extracted-problems.jsonl"

    for mode in ['proof', 'mcq', 'binary', 'invalid']:
        postprocess_cmd = (
            f"python /nemo_run/code/recipes/omr1/scripts/postprocess_classification.py "
            f"    {output_dir}/classify-problems/{mode}/output.jsonl "
            f"    {output_dir}/classify-problems/{mode}/yes.jsonl "
            f"    {output_dir}/classify-problems/{mode}/no.jsonl "
            f"    --mode={mode}"
        )

        generate(
            ctx=wrap_arguments(
                f"++input_file={input_file} "
                f"++prompt_config=/nemo_run/code/recipes/omr1/prompts/classify-if-{mode}.yaml "
                f"{extra_args} "
            ),
            cluster=cluster,
            output_dir=f"{output_dir}/classify-problems/{mode}",
            postprocess_cmd=postprocess_cmd,
            expname=f"{expname}-classify-{mode}",
            run_after=run_after,
            **generate_kwargs,
        )
        run_after = f"{expname}-classify-{mode}"
        input_file = f"{output_dir}/classify-problems/{mode}/no.jsonl"

    return input_file, run_after


def extract_answers(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-classify-invalid"
    input_file = f"{output_dir}/classify-problems/invalid/no.jsonl"

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/omr1/scripts/postprocess_answer_extraction.py "
        f"    {output_dir}/extract-answers/output.jsonl "
        f"    {output_dir}/extract-answers/extracted-answers.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=/nemo_run/code/recipes/omr1/prompts/extract-answers.yaml "
            f"{extra_args} "
        ),
        cluster=cluster,
        output_dir=f"{output_dir}/extract-answers",
        postprocess_cmd=postprocess_cmd,
        expname=f"{expname}-extract-answers",
        run_after=run_after,
        **generate_kwargs,
    )


def convert_proofs(output_dir, cluster, expname, extra_args="", **generate_kwargs):
    run_after = f"{expname}-classify-proof"
    input_file = f"{output_dir}/classify-problems/proof/yes.jsonl"

    postprocess_cmd = (
        f"python /nemo_run/code/recipes/omr1/scripts/postprocess_proof_conversion.py "
        f"    {output_dir}/convert-proofs/output.jsonl "
        f"    {output_dir}/convert-proofs/converted-proofs.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=/nemo_run/code/recipes/omr1/prompts/convert-proofs.yaml "
            f"{extra_args} "
        ),
        cluster=cluster,
        output_dir=f"{output_dir}/convert-proofs",
        postprocess_cmd=postprocess_cmd,
        expname=f"{expname}-convert-proofs",
        run_after=run_after,
        **generate_kwargs,
    )


stages_map = {
    'extract_problems': extract_problems,
    'classify_problems': classify_problems,
    'extract_answers': extract_answers,
    'convert_proofs': convert_proofs,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenMathReasoning-1 pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['demo', 'full'],
        help="Will pick a corresponding config from configs folder",
    )
    parser.add_argument('--stages', type=str, help='Pipeline stages to run', default='all')

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

    run_after = None
    input_file = config['input_file']
    default_args = dict(
        output_dir=config['output_dir'], cluster=config['cluster'], expname=config['expname'], **config['problem_sdg']
    )

    for stage in stages:
        stage_args = default_args.copy()
        if stage == 'extract_problems':
            stage_args['input_file'] = config['input_file']
        stages_map[stage](**stage_args)

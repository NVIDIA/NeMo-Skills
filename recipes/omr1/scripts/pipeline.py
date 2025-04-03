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


def extract_problems(
    input_file,
    output_dir,
    cluster,
    expname,
    model,
    server_type,
    server_address=None,
    prompt_template=None,
    server_gpus=None,
    server_nodes=None,
):
    postprocess_cmd = (
        f"python /nemo_run/code/aops-recipe/scripts/postprocess_problem_extraction.py "
        f"    {output_dir}/extract-problems/output.jsonl "
        f"    {output_dir}/extracted-problems.jsonl "
    )

    extra_args = f"++prompt_template={prompt_template} " if prompt_template else ""

    generate(
        ctx=wrap_arguments(
            f"++input_file={input_file} "
            f"++prompt_config=/nemo_run/code/aops-recipe/prompts/forum-problem-extraction.yaml "
            f"{extra_args} "
        ),
        cluster=cluster,
        model=model,
        server_address=server_address,
        server_type=server_type,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        output_dir=f"{output_dir}/extract-problems",
        postprocess_cmd=postprocess_cmd,
        expname=f"{expname}-extract-problems",
        # samples_path=f'{output_dir}/extracted-problems.jsonl',
    )


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

    extract_problems(
        input_file=config['input_file'],
        output_dir=config['output_dir'],
        cluster=config['cluster'],
        expname=config['expname'],
        **config['problem_sdg'],
    )

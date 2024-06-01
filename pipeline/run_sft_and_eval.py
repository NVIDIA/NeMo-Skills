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

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

# adding nemo_skills to python path to avoid requiring installation
sys.path.append(str(Path(__file__).absolute().parents[1]))

from launcher import CLUSTER_CONFIG, NEMO_SKILLS_CODE, fill_env_vars

from nemo_skills.utils import setup_logging


def start_job(extra_sbatch_args: str, cmd: str) -> int:
    full_cmd = f'{extra_sbatch_args} {cmd}'

    if CLUSTER_CONFIG['cluster'] == 'local':
        subprocess.run(full_cmd, shell=True)
        return 0
    job_id = subprocess.run(full_cmd, shell=True, check=True, capture_output=True).stdout.decode()
    print(f"Submitted batch job(s) {job_id}")
    return job_id


def run_sft(
    current_folder, results_folder, checkpoints_folder, args, extra_sft_arguments, inference_path, last_job_id=None
):
    # launching SFT jobs
    dependency = f'--dependency=afterany:{last_job_id}' if last_job_id is not None else ''
    extra_sbatch_args = (
        f'EXTRA_SBATCH_ARGS="--parsable --output={checkpoints_folder}/slurm_logs_sft1.txt {dependency}"'
    )
    cmd = (
        f'{sys.executable} {current_folder}/run_sft.py '
        f'    --project {args.project} '
        f'    --expname {args.expname} '
        f'    --checkpoints_folder {checkpoints_folder}/training '
        f'    --nemo_model {args.nemo_model} '
        f'    --num_nodes {args.num_nodes} '
        f'    --num_gpus {args.num_gpus} '
        f'    {extra_sft_arguments} '
    )
    last_job_id = start_job(extra_sbatch_args, cmd)
    for i in range(args.num_sft_jobs - 1):
        extra_sbatch_args = (
            f'EXTRA_SBATCH_ARGS="--parsable --dependency=afterany:{last_job_id} '
            f'--output={checkpoints_folder}/slurm_logs_sft{i + 2}.txt"'
        )
        last_job_id = start_job(extra_sbatch_args, cmd)
    return last_job_id


def run_prepare_eval(
    current_folder, results_folder, checkpoints_folder, args, extra_sft_arguments, inference_path, last_job_id=None
):
    # preparing checkpoint for evaluation
    dependency = f'--dependency=afterany:{last_job_id}' if last_job_id is not None else ''
    extra_sbatch_args = (
        f'EXTRA_SBATCH_ARGS="--parsable {dependency} --output={checkpoints_folder}/slurm_logs_prepare_for_eval.txt"'
    )
    cmd = (
        f'{sys.executable} {current_folder}/prepare_eval.py '
        f'    --training_folder {checkpoints_folder}/training/checkpoints '
        f'    --output_path {inference_path} '
        f'    --nemo_model {args.nemo_model} '
        f'    --num_gpus {args.num_gpus} '
        f'    --server_type {args.server_type} '
        f'    {args.extra_prepare_eval_args} '
    )
    last_job_id = start_job(extra_sbatch_args, cmd)
    return last_job_id


def run_eval(
    current_folder, results_folder, checkpoints_folder, args, extra_sft_arguments, inference_path, last_job_id=None
):
    # launching evaluation
    dependency = f'--dependency=afterany:{last_job_id}' if last_job_id is not None else ''
    # logs are managed by run_eval.py script
    extra_sbatch_args = f'EXTRA_SBATCH_ARGS="--parsable {dependency.strip()}"'

    cmd = (
        f'{sys.executable} {current_folder}/run_eval.py '
        f'    --model_path {inference_path} '
        f'    --output_dir {results_folder} '
        f'    --num_gpus {args.num_gpus} '
        f'    --server_type {args.server_type} '
        f'    ++split_name=validation '
        f'    +prompt=code_sfted '
        f'    ++prompt.few_shot_examples.num_few_shots=0 '
        f'    {args.extra_eval_args} '
    )
    last_job_id = start_job(extra_sbatch_args, cmd)
    return last_job_id


stages_map = {
    'sft': run_sft,
    'prepare_eval': run_prepare_eval,
    'eval': run_eval,
}


if __name__ == "__main__":
    setup_logging(disable_hydra_logs=False)
    parser = ArgumentParser()
    # by default we are using a shared project
    parser.add_argument("--project", default="nemo-skills-exps")
    parser.add_argument("--expname", required=True, help="Experiment name for logging purposes")
    parser.add_argument("--nemo_model", required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--num_sft_jobs", type=int, default=1)
    parser.add_argument("--server_type", choices=('nemo',), default='nemo')
    parser.add_argument("--stages", nargs="+", default=["sft", "prepare_eval", "eval"])
    parser.add_argument("--extra_eval_args", default="")
    parser.add_argument("--extra_prepare_eval_args", default="")
    args, unknown = parser.parse_known_args()

    # these are the extra SFT arguments you can provide
    extra_sft_arguments = f'{" ".join(unknown)}'

    format_dict = {
        "NEMO_SKILLS_CODE": NEMO_SKILLS_CODE,
    }
    fill_env_vars(format_dict, ["NEMO_SKILLS_RESULTS"])

    exp_path = f"{format_dict['NEMO_SKILLS_RESULTS']}/{args.project}"

    checkpoints_folder = Path(f"{exp_path}/checkpoints/{args.expname}")
    checkpoints_folder.mkdir(exist_ok=True, parents=True)

    results_folder = Path(f"{exp_path}/results/{args.expname}")
    results_folder.mkdir(exist_ok=True, parents=True)

    current_folder = Path(__file__).parent.absolute()

    if args.server_type == "nemo":  # adding expname for better logging
        inference_path = f"{checkpoints_folder}/{args.expname}.nemo"
    else:
        inference_path = f"{checkpoints_folder}/{args.server_type}"

    last_job_id = None
    for stage in args.stages:
        stage_fn = stages_map[stage]
        last_job_id = stage_fn(
            current_folder,
            results_folder,
            checkpoints_folder,
            args,
            extra_sft_arguments,
            inference_path,
            last_job_id,
        )

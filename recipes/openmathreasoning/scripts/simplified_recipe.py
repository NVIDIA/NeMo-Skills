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

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.pipeline.cli import convert, eval, generate, run_cmd, sft_nemo_rl, train, wrap_arguments


def prepare(workspace, cluster, num_gpus, training_backend):
    # data preparation needs to run locally without container, so not wrapping with run_cmd
    prepare_datasets(["aime24", "aime25"])

    # download the models and prepare the data
    cmd = (
        f"huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir {workspace}/Qwen2.5-14B-Instruct && "
        f"huggingface-cli download Qwen/QwQ-32B --local-dir {workspace}/QwQ-32B && "
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/main/recipes/openmathreasoning && "
        f"wget $DOWNLOAD_PREFIX/scripts/prepare_raw_data.py && "
        f"wget $DOWNLOAD_PREFIX/prompts/extract-problems.yaml && "
        f"wget $DOWNLOAD_PREFIX/scripts/postprocess_problem_extraction.py && "
        f"python prepare_raw_data.py && "
        f"head -n 1000 raw_aops_data.jsonl > data.jsonl"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname="download-assets",
        log_dir=f"{workspace}/download-assets",
    )
    # convert QwQ trtllm format
    convert(
        ctx=wrap_arguments("--max_seq_len 10000"),
        cluster=cluster,
        input_model=f"{workspace}/QwQ-32B",
        output_model=f"{workspace}/qwq32b-trtllm",
        convert_from="hf",
        convert_to="trtllm",
        num_gpus=num_gpus,
        model_type="qwen",
        hf_model_name="Qwen/QwQ-32B",
        expname="convert-qwq-trtllm",
        run_after="download-assets",
    )

    if training_backend == "nemo-aligner":
        # convert Qwen2.5-14B-Instruct to nemo format
        convert(
            ctx=wrap_arguments(""),
            cluster=cluster,
            input_model=f"{workspace}/Qwen2.5-14B-Instruct",
            output_model=f"{workspace}/qwen2.5-14b-instruct-nemo",
            convert_from="hf",
            convert_to="nemo",
            num_gpus=num_gpus,
            model_type="qwen",
            hf_model_name="Qwen/Qwen2.5-14B-Instruct",
            expname="convert-14b-nemo",
            run_after="download-assets",
        )


def run_sdg(workspace, cluster, num_gpus, wandb_params):
    postprocess_cmd = (
        f"python {workspace}/postprocess_problem_extraction.py "
        f"    {workspace}/sdg/problems/output.jsonl "
        f"    {workspace}/sdg/extracted-problems.jsonl "
    )

    generate(
        ctx=wrap_arguments(
            f"++input_file={workspace}/data.jsonl "
            f"++prompt_config={workspace}/extract-problems.yaml "
            f"++prompt_template=qwen-instruct "
        ),
        cluster=cluster,
        output_dir=f"{workspace}/sdg/problems",
        postprocess_cmd=postprocess_cmd,
        expname="problem-extraction",
        run_after="download-assets",
        model=f"{workspace}/Qwen2.5-14B-Instruct",
        server_type="vllm",
        server_gpus=num_gpus,
        log_samples=not wandb_params['disable_wandb'],
        wandb_group=wandb_params['wandb_group'],
        wandb_project=wandb_params['wandb_project'],
    )

    generate(
        ctx=wrap_arguments(
            f"++input_file={workspace}/sdg/extracted-problems.jsonl "
            f"++prompt_config=generic/math "
            f"++inference.temperature=0.6 "
            f"++inference.tokens_to_generate=8192 "
            f"++prompt_template=qwen-instruct "
        ),
        cluster=cluster,
        output_dir=f'{workspace}/sdg/solutions',
        expname='solution-generation',
        run_after=['problem-extraction', 'convert-qwq-trtllm'],
        model=f'{workspace}/qwq32b-trtllm',
        server_type='trtllm',
        server_gpus=num_gpus,
        log_samples=not wandb_params['disable_wandb'],
        wandb_group=wandb_params['wandb_group'],
        wandb_project=wandb_params['wandb_project'],
    )


def run_training(workspace, cluster, num_gpus, training_backend, wandb_params):
    # convert the generated solutions to a format that can be used for training
    # and remove contaminated data
    run_cmd(
        ctx=wrap_arguments(
            f"python -m nemo_skills.training.prepare_data "
            f"    ++input_files={workspace}/sdg/solutions/output.jsonl "
            f"    ++output_path={workspace}/sft-data.jsonl "
            f"    ++prompt_config=generic/math "
            f"    ++prompt_template=qwen-instruct "
            f"    ++filters.remove_contaminated=false "
            f"    ++add_unlabeled=true "
            f"    ++filters.remove_no_think_tags=true "
            f"    ++filters.trim_solutions=false"
        ),
        cluster=cluster,
        expname="prepare-training-data",
        log_dir=f"{workspace}/prepare-training-data",
        run_after="solution-generation",
    )

    # train the model
    if training_backend == "nemo-aligner":
        train(
            ctx=wrap_arguments(
                f"++model.data.train_ds.max_seq_length=8192 "
                f"++model.data.train_ds.global_batch_size=32 "
                f"++model.tensor_model_parallel_size=4 "
                f"++model.context_parallel_size=2 "
                f"++model.optim.lr=1e-5 "
                f"++trainer.sft.max_epochs=2 "
            ),
            cluster=cluster,
            output_dir=f"{workspace}/training",
            nemo_model=f"{workspace}/qwen2.5-14b-instruct-nemo",
            num_gpus=num_gpus,
            num_nodes=1,
            disable_wandb=wandb_params['disable_wandb'],
            wandb_group=wandb_params['wandb_group'],
            wandb_project=wandb_params['wandb_project'],
            training_data=f"{workspace}/sft-data.jsonl",
            expname="training",
            run_after=["prepare-training-data", "convert-14b-nemo"],
        )
    elif training_backend == "nemo-rl":
        sft_nemo_rl(
            ctx=wrap_arguments(
                '++sft.max_num_epochs=2 '
                '++policy.dtensor_cfg.tensor_parallel_size=8 '
                '++policy.max_total_sequence_length=8192 '
                '++policy.train_global_batch_size=32 '
                '++policy.optimizer.kwargs.lr=1e-5 '
                '++policy.dtensor_cfg.sequence_parallel=true '
                '++policy.dtensor_cfg.activation_checkpointing=true '
            ),
            cluster=cluster,
            output_dir=f'{workspace}/training',
            hf_model=f'{workspace}/Qwen2.5-14B-Instruct',
            num_gpus=num_gpus,
            num_nodes=1,
            disable_wandb=wandb_params['disable_wandb'],
            wandb_group=wandb_params['wandb_group'],
            wandb_project=wandb_params['wandb_project'],
            training_data=f'{workspace}/sft-data.jsonl',
            cache_dir=f'{workspace}/nemo-rl-cache',
            expname="training",
            run_after="prepare-training-data",
            final_hf_path=f"{workspace}/training/qwen2.5-14b-improved-hf",
        )
    else:
        raise ValueError(f"Unknown training backend: {training_backend}")


def final_eval(workspace, cluster, num_gpus, training_backend, wandb_params):
    if training_backend == 'nemo-aligner':
        # converting back to HF format
        convert(
            ctx=wrap_arguments(""),
            cluster=cluster,
            input_model=f"{workspace}/training/model-averaged-nemo",
            output_model=f"{workspace}/training/qwen2.5-14b-improved-hf",
            convert_from="nemo",
            convert_to="hf",
            num_gpus=num_gpus,
            model_type="qwen",
            hf_model_name="Qwen/Qwen2.5-14B-Instruct",
            expname="convert-back-to-hf",
            run_after="training",
        )

    # launching evaluation
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate=16384 "),
        cluster=cluster,
        model=f"{workspace}/training/qwen2.5-14b-improved-hf",
        server_type="vllm",
        server_gpus=num_gpus,
        benchmarks="aime24:8,aime25:8",
        output_dir=f"{workspace}/evals/after-training",
        num_jobs=1 if cluster == "local" else -1,
        expname="final-eval",
        run_after=["convert-back-to-hf", "training"],
    )

    # summarize results, after the evaluation job is done
    summarize_cmd = f"ns summarize_results {workspace}/evals/after-training "
    if not wandb_params['disable_wandb']:
        summarize_cmd += (
            # not using an actual group parameter but adding to the name as grouping is harder to view in the UI
            f" --wandb_name {wandb_params['wandb_group']}-final-eval "
            f" --wandb_project {wandb_params['wandb_project']}"
        )
    run_cmd(
        ctx=wrap_arguments(summarize_cmd),
        cluster=cluster,
        expname="summarize-results",
        run_after="final-eval",
        log_dir=f"{workspace}/summarize-results/after-training",
    )


def initial_eval(workspace, cluster, num_gpus, wandb_params):
    # launching evaluation
    eval(
        ctx=wrap_arguments(""),
        cluster=cluster,
        model=f"{workspace}/Qwen2.5-14B-Instruct",
        server_type="vllm",
        server_gpus=num_gpus,
        benchmarks="aime24:8,aime25:8",
        output_dir=f"{workspace}/evals/baseline",
        num_jobs=1,
        expname="baseline-eval",
    )

    # summarize results, after the evaluation job is done
    summarize_cmd = f"ns summarize_results {workspace}/evals/baseline "
    if not wandb_params['disable_wandb']:
        summarize_cmd += (
            # not using an actual group parameter but adding to the name as grouping is harder to view in the UI
            f" --wandb_name {wandb_params['wandb_group']}-baseline-eval "
            f" --wandb_project {wandb_params['wandb_project']}"
        )
    run_cmd(
        ctx=wrap_arguments(summarize_cmd),
        cluster=cluster,
        expname="summarize-results",
        run_after="baseline-eval",
        log_dir=f"{workspace}/summarize-results/baseline",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simplified OpenMathReasoning recipe for testing the code")
    parser.add_argument(
        "--cluster",
        type=str,
        default="local",
        help="Cluster name to run the job on. Use 'local' for local execution.",
    )
    parser.add_argument("--workspace", type=str, default="/workspace", help="Workspace directory for the job.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for the job.",
    )
    parser.add_argument(
        "--training_backend",
        type=str,
        default="nemo-aligner",
        choices=["nemo-aligner", "nemo-rl"],
        help="Training backend to use.",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="test-pipeline",
        help="WandB group name for tracking experiments.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="nemo-skills",
        help="WandB project name for tracking experiments.",
    )
    args = parser.parse_args()

    wandb_params = {
        "disable_wandb": args.disable_wandb,
        "wandb_group": args.wandb_group,
        "wandb_project": args.wandb_project,
    }
    prepare(args.workspace, args.cluster, args.num_gpus, args.training_backend)
    initial_eval(args.workspace, args.cluster, args.num_gpus, wandb_params)
    run_sdg(args.workspace, args.cluster, args.num_gpus, wandb_params)
    run_training(args.workspace, args.cluster, args.num_gpus, args.training_backend, wandb_params)
    final_eval(args.workspace, args.cluster, args.num_gpus, args.training_backend, wandb_params)

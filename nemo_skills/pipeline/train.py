# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import nemo_run as run
import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import add_task, check_if_mounted, get_cluster_config, get_timeout, run_exp
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)

GRPO_VLLM_PORT = 4321


class TrainingAlgo(str, Enum):
    sft = "sft"
    dpo = "dpo"
    rm = "rm"
    grpo = "grpo"


@dataclass
class TrainingParams:
    training_script: str
    config_params: str
    nemo_model: str
    output_dir: str
    training_data: str
    validation_data: str
    num_gpus: int
    tp: int | None
    num_nodes: int
    expname: str
    training_algo: TrainingAlgo
    disable_wandb: bool
    wandb_project: str
    timeout: str
    extra_arguments: str = ""
    logging_params: str = ""

    def __post_init__(self):
        self.extra_arguments = get_extra_arguments[self.training_algo](self)


def get_cmd(params: TrainingParams) -> str:
    cmd = (
        f"export WANDB_API_KEY={os.getenv('WANDB_API_KEY', '')} && "
        f"export HYDRA_FULL_ERROR=1 && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
        f"export TEMPDIR=/dev/shm/checkpoints_$SLURM_JOB_ID && "
        f"mkdir -p $TEMPDIR && "
        f"chmod 777 $TEMPDIR && "
        f"cd /nemo_run/code && "
        f"echo 'Starting training' && "
        f"{params.training_script} "
        f"    {params.config_params}"
        f"    ++model.tensor_model_parallel_size={params.tp or params.num_gpus} "
        f"    trainer.devices={params.num_gpus} "
        f"    trainer.num_nodes={params.num_nodes} "
        f"    {params.logging_params} "
        f"    exp_manager.name={params.expname} "
        f"    exp_manager.explicit_log_dir={params.output_dir}/training "
        f"    exp_manager.exp_dir={params.output_dir}/training "
        f"    ++exp_manager.max_time_per_run={params.timeout} "
        f"    {params.extra_arguments} "
    )
    return cmd


configs = {
    TrainingAlgo.sft: "sft_config",
    TrainingAlgo.dpo: "dpo_config",
    TrainingAlgo.rm: "rm_config",
    TrainingAlgo.grpo: "grpo_config",
}

rl_extra_args_fn = lambda params: (
    f" ++model.data.data_prefix.train='[{params.training_data}]' "
    f" ++model.data.data_prefix.validation='[{params.validation_data}]' "
    f" ++model.data.data_prefix.test='[{params.validation_data}]' "
    f" pretrained_checkpoint.restore_from_path={params.nemo_model} " + params.extra_arguments
)

get_extra_arguments: dict[TrainingAlgo, Callable[[TrainingParams], str]] = {
    TrainingAlgo.sft: lambda params: (
        f" ++model.data.train_ds.file_path='{params.training_data}' "
        f" ++model.data.train_ds.index_mapping_dir='{os.path.dirname(os.path.abspath(params.training_data))}' "
        f" ++model.data.validation_ds.file_path='{params.validation_data}' "
        f" ++model.data.validation_ds.index_mapping_dir='{os.path.dirname(os.path.abspath(params.validation_data))}' "
        f" model.restore_from_path={params.nemo_model} " + params.extra_arguments
    ),
    TrainingAlgo.dpo: rl_extra_args_fn,
    TrainingAlgo.grpo: lambda params: (
        f" ++trainer.grpo.generation_save_dir={params.output_dir}/generations "
        f" ++trainer.grpo.inference_backend.config.vllm.port=$(( {GRPO_VLLM_PORT} + (SLURM_LOCALID / {params.tp}) * {params.tp} )) "
        f" ++model.grpo.share_dir=$TEMPDIR "
    )
    + rl_extra_args_fn(params),
    TrainingAlgo.rm: rl_extra_args_fn,
}


def get_grpo_vllm_cmd(params: TrainingParams) -> str:
    return (
        f"export VLLM_FLASK_SERVER_PORT=$(( {GRPO_VLLM_PORT} + SLURM_LOCALID * {params.tp} )) && "
        f"cd /opt/NeMo-Aligner/examples/nlp/gpt && "
        f"python3 serve_vllm_inference_server.py --port $VLLM_FLASK_SERVER_PORT "
    )


def get_training_cmd(
    cluster_config,
    partition,
    config_name,
    config_path,
    nemo_model,
    output_dir,
    training_data,
    validation_data,
    num_gpus,
    num_nodes,
    expname,
    training_algo,
    disable_wandb,
    wandb_project,
    extra_arguments,
):
    if validation_data is None:
        validation_data = training_data

    timeout = get_timeout(cluster_config, partition)

    logging_params = get_logging_params(expname, disable_wandb, wandb_project)

    if config_name is None:
        config_name = configs[training_algo]
    config_params = f"--config-name={config_name} --config-path={config_path} "

    training_script = f"python -m nemo_skills.training.start_{training_algo}"

    training_params = TrainingParams(
        training_script=training_script,
        config_params=config_params,
        nemo_model=nemo_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        training_algo=training_algo,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        timeout=timeout,
        extra_arguments=extra_arguments,
        logging_params=logging_params,
    )

    return get_cmd(training_params)


def get_logging_params(expname, disable_wandb, wandb_project):
    if not disable_wandb:
        if os.getenv('WANDB_API_KEY') is None:
            raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")
        logging_params = (
            f"exp_manager.create_wandb_logger=True "
            f"exp_manager.wandb_logger_kwargs.name={expname} "
            f"exp_manager.wandb_logger_kwargs.project={wandb_project} "
            f"+exp_manager.wandb_logger_kwargs.id={expname} "
            f"+exp_manager.wandb_logger_kwargs.resume=True "
        )
    else:
        logging_params = "exp_manager.create_wandb_logger=False +exp_manager.create_tensorboard_logger=True"
    return logging_params


def get_avg_checkpoints_cmd(nemo_model, output_dir, final_nemo_path, average_steps):
    name = "model" + ("-".join(average_steps[len('--steps ') :].split()) if average_steps else '') + "-averaged"
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.training.average_checkpoints "
        f"    --untarred_nemo_dir {nemo_model} "
        f"    --name_prefix=model "
        f"    --checkpoint_dir={output_dir}/training/checkpoints {average_steps} && "
        f"mkdir -p {os.path.dirname(final_nemo_path)} && "
        f"mv {output_dir}/training/checkpoints/{name} {final_nemo_path} "
    )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def train(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    final_nemo_path: str = typer.Option(None, help="Where to put the final checkpoint"),
    expname: str = typer.Option(..., help="Experiment name"),
    nemo_model: str = typer.Option(..., help="Path to the NeMo model"),
    training_data: str = typer.Option(None, help="Path to the training data"),
    validation_data: str = typer.Option(None, help="Path to the validation data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    tp: int = typer.Option(
        None, help="Tensor parallel size. Required for grpo and optional (set to num_gpus) for other algos"
    ),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    training_algo: TrainingAlgo = typer.Option(TrainingAlgo.sft, help="Training algorithm"),
    config_name: str = typer.Option(None, help="Config name"),
    config_path: str = typer.Option('/nemo_run/code/nemo_skills/training/', help="Config path"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    with_sandbox: bool = typer.Option(False, help="If sandbox is required for code generation"),
    partition: str = typer.Option(None, help="Specify partition for jobs"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    average_steps: str = typer.Option(
        'all',
        help="List of commas separated checkpoint steps to average. E.g 1000,5000. "
        "If None, will skip prepare eval stage.",
    ),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs. "),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
):
    """Train (SFT or DPO) an LLM model.

    All extra arguments are passed directly to the training script
    (need to be prefixed with ++, since NeMo uses Hydra).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    if tp is None and training_algo == TrainingAlgo.grpo:
        raise ValueError("tp is required to be explicitly specified for grpo")

    try:
        training_algo = training_algo.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, output_dir)
    check_if_mounted(cluster_config, nemo_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = output_dir

    if num_training_jobs > 0:
        if training_data is None:
            raise ValueError("training_data is required when num_training_jobs > 0")
        check_if_mounted(cluster_config, training_data)

    if not final_nemo_path:
        final_nemo_path = f"{output_dir}/model-averaged-nemo"
    check_if_mounted(cluster_config, final_nemo_path)

    if validation_data:
        check_if_mounted(cluster_config, validation_data)

    if " " in str(average_steps):
        raise ValueError("average steps should be separated with commas")

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        config_name=config_name,
        config_path=config_path,
        nemo_model=nemo_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        training_algo=training_algo,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        extra_arguments=extra_arguments,
    )

    with run.Experiment(expname) as exp:
        prev_task = None
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f'{expname}-{training_algo}-{job_id}',
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["nemo"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=num_gpus if cluster_config["executor"] == "slurm" else 1,
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                with_sandbox=with_sandbox,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )

        if average_steps is not None:
            cmd = get_avg_checkpoints_cmd(
                nemo_model=nemo_model,
                output_dir=output_dir,
                final_nemo_path=final_nemo_path,
                average_steps=f"--steps {' '.join(average_steps.split(','))} " if average_steps != 'all' else "",
            )

            add_task(
                exp,
                cmd=cmd,
                task_name=f"{expname}-prepare-eval",
                log_dir=f"{log_dir}/prepare-eval-logs",
                container=cluster_config["containers"]['nemo'],
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                num_nodes=1,
                num_tasks=1,
                num_gpus=num_gpus,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
            )

        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False)

    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

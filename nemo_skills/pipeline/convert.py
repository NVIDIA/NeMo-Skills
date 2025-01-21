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
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List

import nemo_run as run
import typer
from huggingface_hub import get_token

from nemo_skills.pipeline import add_task, check_if_mounted, get_cluster_config, run_exp
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import setup_logging

LOG = logging.getLogger(__file__)


def get_nemo_to_hf_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.conversion.nemo_to_hf_{model_type} "
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    --max-shard-size 10GB "
        f"    {extra_arguments} "
    )
    return cmd


def get_hf_to_trtllm_cmd(
    input_model,
    output_model,
    model_type,
    hf_model_name,
    dtype,
    num_gpus,
    num_nodes,
    extra_arguments,
    trt_prepare_args,
    trt_reuse_tmp_engine,
):
    dtype = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[dtype]

    tmp_engine_dir = f"{output_model}-tmp"

    setup_cmd = f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && cd /nemo_run/code && "

    hf_to_trtllm_cmd = (
        f"python -m nemo_skills.conversion.hf_to_trtllm_{model_type} "
        f"    --model_dir {input_model} "
        f"    --output_dir {tmp_engine_dir} "
        f"    --dtype {dtype} "
        f"    --tp_size {num_gpus} "
        f"    --pp_size {num_nodes} "
        f"    {trt_prepare_args} "
    )

    trtllm_build_cmd = (
        f"trtllm-build "
        f"    --checkpoint_dir {tmp_engine_dir} "
        f"    --output_dir {output_model} "
        f"    --gpt_attention_plugin {dtype} "
        f"    --use_paged_context_fmha enable "
        f"    --max_input_len 4096 "
        f"    --max_seq_len 8192 "
        f"    --max_num_tokens 8192 "
        f"    --max_batch_size 128 "
        f"    {extra_arguments} && "
        f"cp {input_model}/tokenizer* {output_model} "
    )

    if trt_reuse_tmp_engine:
        cmd = (
            setup_cmd + f"if [ ! -f {tmp_engine_dir}/config.json ]; then {hf_to_trtllm_cmd}; fi && {trtllm_build_cmd}"
        )
    else:
        cmd = setup_cmd + hf_to_trtllm_cmd + " && " + trtllm_build_cmd

    return cmd


def get_hf_to_nemo_cmd(
    input_model, output_model, model_type, hf_model_name, dtype, num_gpus, num_nodes, extra_arguments
):
    # Check if the model_type is "nemo"

    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python -m nemo_skills.conversion.hf_to_nemo_{model_type} "
        f"    --in-path {input_model} "
        f"    --out-path {output_model} "
        f"    --hf-model-name {hf_model_name} "
        f"    --precision {dtype} "
        f"    {extra_arguments} "
    )

    return cmd


class SupportedTypes(str, Enum):
    llama = "llama"
    qwen = "qwen"
    deepseek_v3 = "deepseek_v3"


class SupportedFormatsTo(str, Enum):
    nemo = "nemo"
    hf = "hf"
    trtllm = "trtllm"


class SupportedFormatsFrom(str, Enum):
    nemo = "nemo"
    hf = "hf"


class SupportedDtypes(str, Enum):
    bf16 = "bf16"
    fp16 = "fp16"
    fp32 = "fp32"


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def convert(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_model: str = typer.Option(...),
    model_type: SupportedTypes = typer.Option(..., help="Type of the model"),
    output_model: str = typer.Option(..., help="Where to put the final model"),
    convert_from: SupportedFormatsFrom = typer.Option(..., help="Format of the input model"),
    convert_to: SupportedFormatsTo = typer.Option(..., help="Format of the output model"),
    trt_prepare_args: str = typer.Option(
        "", help="Arguments to pass to the first step of trtllm conversion (that builds tmp engine)"
    ),
    trt_reuse_tmp_engine: bool = typer.Option(True, help="Whether to reuse the tmp engine for the final conversion"),
    hf_model_name: str = typer.Option(None, help="Name of the model on Hugging Face Hub to convert to/from"),
    dtype: SupportedDtypes = typer.Option("bf16", help="Data type"),
    expname: str = typer.Option("conversion", help="NeMo-Run experiment name"),
    num_nodes: int = typer.Option(1),
    num_gpus: int = typer.Option(...),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    exclusive: bool = typer.Option(False, help="If True, will use --exclusive flag for slurm"),
):
    """Convert a checkpoint from one format to another.

    All extra arguments are passed directly to the underlying conversion script (see their docs).
    """
    setup_logging(disable_hydra_logs=False)
    extra_arguments = f'{" ".join(ctx.args)}'
    LOG.info("Starting conversion job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        model_type = model_type.value
        convert_from = convert_from.value
        convert_to = convert_to.value
        dtype = dtype.value
    except AttributeError:
        pass

    # TODO: add support for conversion from NeMo to trtllm using nemo.export (need to test thoroughly)
    if convert_from == "nemo" and convert_to == "trtllm":
        raise ValueError("Conversion from NeMo to TensorRT LLM is not supported directly. Convert to HF first.")

    if convert_to != "trtllm" and hf_model_name is None:
        raise ValueError("--hf_model_name is required")

    if convert_to in ["hf", "nemo"] and model_type == "deepseek_v3":
        raise ValueError("Conversion to HF/Nemo is not yet supported for DeepSeek v3 models")

    cluster_config = get_cluster_config(cluster, config_dir)
    check_if_mounted(cluster_config, input_model)
    check_if_mounted(cluster_config, output_model)
    if log_dir:
        check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = str(Path(output_model) / "conversion-logs")

    conversion_cmd_map = {
        ("nemo", "hf"): get_nemo_to_hf_cmd,
        ("hf", "nemo"): get_hf_to_nemo_cmd,
        ("hf", "trtllm"): partial(
            get_hf_to_trtllm_cmd,
            trt_prepare_args=trt_prepare_args,
            trt_reuse_tmp_engine=trt_reuse_tmp_engine,
        ),
    }
    container_map = {
        ("nemo", "hf"): cluster_config["containers"]["nemo"],
        ("hf", "nemo"): cluster_config["containers"]["nemo"],
        ("hf", "trtllm"): cluster_config["containers"]["trtllm"],
    }
    conversion_cmd = conversion_cmd_map[(convert_from, convert_to)](
        input_model=input_model,
        output_model=output_model,
        model_type=model_type,
        hf_model_name=hf_model_name,
        dtype=dtype,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        extra_arguments=extra_arguments,
    )
    with run.Experiment(expname) as exp:
        LOG.info("Launching task with command %s", conversion_cmd)
        add_task(
            exp,
            cmd=conversion_cmd,
            task_name=expname,
            log_dir=log_dir,
            container=container_map[(convert_from, convert_to)],
            num_gpus=num_gpus,
            num_nodes=1,  # always running on a single node, might need to change that in the future
            num_tasks=1,
            cluster_config=cluster_config,
            partition=partition,
            time_min=time_min,
            run_after=run_after,
            reuse_code_exp=reuse_code_exp,
            slurm_kwargs={"exclusive": exclusive} if exclusive else None,
        )
        run_exp(exp, cluster_config)

    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()

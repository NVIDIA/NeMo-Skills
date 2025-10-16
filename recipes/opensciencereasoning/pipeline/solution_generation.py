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
import json
import shlex

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from recipes.opensciencereasoning.few_shots import few_shots
# Final output file name for each stage
OUTPUT_FILE = "final_result.jsonl"

def get_stage_expname(base_expname: str, stage_name: str, suffix: str):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"


def get_available_configs(config_dir: Path):
    """Get available YAML configuration files from the config directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        return []

    yaml_files = list(config_dir.glob("*.yaml"))
    config_names = [file.stem for file in yaml_files]

    return config_names


def filter_problems(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    option_format_regex = stage_config.get('option_format_regex', None)
    option_format_regex = f" --option_format_regex '{option_format_regex}' " if option_format_regex else ""

    cmd = (
        f"python /nemo_run/code/recipes/opensciencereasoning/scripts/filter_problems.py "
        f"{input_file} "
        f"{output_dir}/{OUTPUT_FILE}"
        + (f" --is_mcq" if stage_config.get('is_mcq', False) else "")
        + (f" --deduplicate" if stage_config.get('deduplicate', False) else "")
        + (f" --remove_images" if stage_config.get('remove_images', False) else "")
        + (f" --dataset_name {stage_config.get('dataset_name', None)}" if stage_config.get('dataset_name') else "")
        + (f" --num_options {stage_config.get('num_options', None)}" if stage_config.get('num_options') else "")
        + option_format_regex
    )
    print(cmd)
    
    wrapped_cmd = wrap_arguments(cmd)
    run_cmd(cluster=cluster, 
            container="nemo-skills",
            log_dir=f"{output_dir}/logs",
            expname=expname, 
            ctx=wrapped_cmd, 
            run_after=run_after)

def difficulty_estimation(cluster, expname, run_after, stage_config, **kwargs):
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    prompt_config = stage_config["mcq_prompt_config"] if stage_config["is_mcq"] else stage_config["openq_prompt_config"]
    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    server_args = generation_kwargs.get("server_args", "")

    expname_1 = f"{expname}-difficulty-generation"

    # run generation
    generate(
        ctx=wrap_arguments(
            f"++prompt_config={prompt_config} " + generation_kwargs.get("inline_args", "")
        ),
        cluster=cluster,
        input_file=input_file,
        output_dir=output_dir,
        expname=expname_1,
        run_after=run_after,
        **generation_kwargs,
        **stage_config.get("stage_kwargs", {}),
    )

    expname_2 = f"{expname}-difficulty-judge"

    generate(
        ctx=wrap_arguments(
            f"++prompt_config={prompt_config} " + generation_kwargs.get("inline_args", "")
        ),
        generation_type="math_judge",
        cluster=cluster,
        input_file=f"{output_dir}/{OUTPUT_FILE}",
        output_dir=output_dir,
        expname=expname_2,
        run_after=expname_1,
        **judge_kwargs,
    )


    # run judging


def decontaminate(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    """Run contamination retrieval and checking, then write decontaminated data.

    1) Retrieve near-duplicates from specified test sets against the input file.
    2) Run a model-driven contamination check on retrieved candidates.
    3) Execute a postprocess script to filter and save final results.
    """
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    test_sets = stage_config.get("test_sets", [])
    model = stage_config.get("model")
    server_type = stage_config.get("server_type")
    server_gpus = stage_config.get("server_gpus")
    server_nodes = stage_config.get("server_nodes")
    dependent_jobs = stage_config.get("dependent_jobs")
    num_chunks = stage_config.get("num_chunks")

    retrieve_from = ",".join(
        f"/nemo_run/code/nemo_skills/dataset/{test_set}/{split}.jsonl" for test_set, split in test_sets
    )
    cmd = (
        f"python -m nemo_skills.inference.retrieve_similar "
        f"    ++retrieve_from=\\'{retrieve_from}\\' "
        f"    ++compare_to='{input_file}' "
        f"    ++output_file='{output_dir}/contamination-retrieved.jsonl' "
        f"    ++top_k=5 "
    )

    run_cmd(
        cluster=cluster,
        container="nemo-skills",
        expname=f"{expname}_retrieve_similar",
        run_after=run_after,
        exclusive=False,
        ctx=wrap_arguments(cmd),
    )

    generate(
        cluster=cluster,
        generation_type="check_contamination",
        input_file=f"{output_dir}/contamination-retrieved.jsonl",
        output_dir=f"{output_dir}/decontaminate/",
        server_type=server_type,
        server_gpus=server_gpus,
        server_nodes=server_nodes,
        model=model,
        expname=f"{expname}_check_contamination",
        run_after=f"{expname}_retrieve_similar",
        exclusive=False,
        num_chunks=num_chunks,
        ctx=wrap_arguments(
            f"++check_both_ways=True "
        ),
        dependent_jobs=dependent_jobs,
    )


    run_cmd(
        ctx=wrap_arguments(
            (
                f"python /nemo_run/code/recipes/opensciencereasoning/scripts/decontaminate.py "
                f"    --input_path '{input_file}' "
                f"    --dec_path '{output_dir}/decontaminate/output.jsonl' "
                f"    --save_path {output_dir}/{OUTPUT_FILE} "
                f"    --with_duplicates False "
            )
        ),
        cluster=cluster,
        exclusive=False,
        run_after=f"{expname}_check_contamination",
        expname=expname,
    )


def topics_labeling(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    """Multi-round labeling of topics and subtopics.

    For each key in `generation_keys` (e.g., topics → subtopics):
      - Prepare inputs with allowed choices and few-shot examples.
      - Run generation using the labeling prompt for that key.
    Finally, aggregate per-level outputs and validate the hierarchy.
    """
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    model = stage_config.get("model")
    server_type = stage_config.get("server_type")
    server_gpus = stage_config.get("server_gpus")
    server_nodes = stage_config.get("server_nodes")
    dependent_jobs = stage_config.get("dependent_jobs")
    num_chunks = stage_config.get("num_chunks")
    few_shots_name = stage_config.get("few_shots_name")
    generation_keys = stage_config.get("generation_keys", [])

    prev_name = None
    save_paths = {}
    topics_structure = {}
    first_dep = run_after or None
    for i, name in enumerate(generation_keys):
        topics_structure[name] = stage_config[name]
        topics_json = json.dumps(stage_config[name], ensure_ascii=False)
        examples_json = json.dumps(few_shots[few_shots_name][name], ensure_ascii=False)
        extra_args = f"    --topic_key {shlex.quote(str(prev_name))} " if prev_name else ""
        run_cmd(
            ctx=wrap_arguments(
                f"python /nemo_run/code/recipes/opensciencereasoning/scripts/prepare_topics.py "
                f"    --input_file '{input_file}' "
                f"    --output_file '{output_dir}/tmp/prepared_for_{name}_labeling.jsonl' "
                f"    --topics_to_choose {shlex.quote(topics_json)} "
                f"    --prompt_examples {shlex.quote(examples_json)} "
                f"    --generation_key {shlex.quote(name)} "
                f"{extra_args}"
            ),
            cluster=cluster,
            exclusive=False,
            expname=f"{expname}-prepare-for-{name}-labeling-{i}",
            run_after=first_dep if i == 0 else f"{expname}-{prev_name}-labeling-{i-1}",
        )
        generate(
            ctx=wrap_arguments(
                f"++prompt_config=/nemo_run/code/recipes/opensciencereasoning/prompts/topics_labeling.yaml "
                f"++generation_key={name} ",
            ),
            cluster=cluster,
            input_file=f"{output_dir}/tmp/prepared_for_{name}_labeling.jsonl",
            output_dir=f"{output_dir}/{name}",
            model=model,
            server_type=server_type,
            num_chunks=num_chunks,
            exclusive=False,
            dependent_jobs=dependent_jobs,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            expname=f"{expname}-{name}-labeling-{i}",
            run_after=f"{expname}-prepare-for-{name}-labeling-{i}",
        )
        input_file = f"{output_dir}/{name}/output.jsonl"
        prev_name = name
        save_paths[name] = f"{output_dir}/{name}/output.jsonl"

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/aggregate_topics.py "
            f"    --input_files {shlex.quote(json.dumps(save_paths, ensure_ascii=False))} "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --topics_structure {shlex.quote(json.dumps(topics_structure, ensure_ascii=False))} "
            f"    --names {shlex.quote(json.dumps(generation_keys, ensure_ascii=False))} "
        ),
        cluster=cluster,
        exclusive=False,
        expname=expname,
        run_after=f"{expname}-{name}-labeling-{i}",
    )

def difficulty_estimation(cluster, expname, run_after, stage_config, **kwargs):
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    prompt_config = stage_config["mcq_prompt_config"] if stage_config["is_mcq"] else stage_config["openq_prompt_config"]
    
    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    expname_1 = f"{expname}-gen"
    tokens_to_generate = generation_kwargs.get("tokens_to_generate", 16000)
    
    # run generation
    # generate(
    #     ctx=wrap_arguments(
    #         f" ++prompt_config={prompt_config} "
    #         f" ++inference.tokens_to_generate={tokens_to_generate} "
    #     ),
    #     cluster=cluster,
    #     input_file=input_file,
    #     output_dir=f"{output_dir}/generation",
    #     expname=expname_1,
    #     run_after=run_after,
    #     **generation_kwargs,
    # )

    expname_2 = f"{expname}-judge"
    
    generate(
        ctx=wrap_arguments(
            f" ++prompt_config=judge/general-judge "
            f" ++inference.tokens_to_generate={tokens_to_generate} "
        ),
        generation_type="math_judge",
        cluster=cluster,
        input_dir=f"{output_dir}/generation",
        output_dir=f"{output_dir}/judge",
        expname=expname_2,
        run_after=expname_1,
        **judge_kwargs,
    )


stages_map = {
    "filter_problems": filter_problems,
    "decontaminate": decontaminate,
    "topics_labeling": topics_labeling,
    "difficulty_estimation": difficulty_estimation,
}


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "configs" / "solution_sdg"
    print(f"Looking for configs in {config_dir}")
    available_configs = get_available_configs(config_dir)

    parser = argparse.ArgumentParser(description="OpenMathReasoning-1 solution generation pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="gpt-oss",
        choices=available_configs,
        help="Will pick a corresponding config from configs folder",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )

    args = parser.parse_args()

    config_path = config_dir / f"{args.mode}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True, structured_config_mode="dict")

    if "pipeline_stages" not in config or not config["pipeline_stages"]:
        raise ValueError(f"Config file {config_path} must define a non-empty 'pipeline_stages' list.")
    full_stage_sequence = config["pipeline_stages"]

    if args.stages:
        # Stages specified via command line
        stages_to_run = args.stages.split(",")
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
    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", args.mode)
    cluster = config["cluster"]
    expname_base = config["expname"]

    # --- Run selected stages ---
    for stage in stages_to_run:
        print(f"\n--- Running stage: {stage} ---")
        stage_func = stages_map[stage]
        stage_config = config.get("stages", {}).get(stage, {})

        current_expname = get_stage_expname(expname_base, stage, suffix)

        dep_stages = stage_config.get("dependencies")
        if dep_stages is not None:
            dependencies = [get_stage_expname(expname_base, dep_stage, suffix) for dep_stage in dep_stages]
        else:
            dependencies = config.get("initial_dependency", None)

        print(f"Dependency for '{stage}': {dependencies}")

        stage_args = {
            "cluster": cluster,
            "expname": current_expname,
            "run_after": dependencies,
            "stage_config": stage_config,
        }

        # Call the stage function
        stage_func(**stage_args)
        current_run_after = current_expname

    print("\n--- Selected pipeline stages finished. ---")

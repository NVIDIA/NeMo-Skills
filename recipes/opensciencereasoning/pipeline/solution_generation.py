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
import json
from pathlib import Path
import shlex
import sys

from omegaconf import OmegaConf

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

sys.path.append(str(Path(__file__).resolve().parents[3]))
from recipes.opensciencereasoning.few_shots import few_shots
from recipes.opensciencereasoning.scripts.constants import BASE_FIELDS

# Final output file name for each stage
OUTPUT_FILE = "final_result.jsonl"

def get_stage_expname(base_expname: str, stage_name: str, suffix: str):
    return f"{base_expname}-{stage_name.replace('_', '-')}-{suffix}"

def prepare_ctx_kwargs(ctx_params: dict):
    ctx_params = " ".join(map(lambda x: f"++{x[0]}={x[1]}", ctx_params.items()))
    return ctx_params

def filter_problems(cluster: str, expname: str, run_after: str, stage_config: dict, **kwargs):
    input_file = stage_config.get("input_file")
    output_dir = stage_config["output_dir"]
    option_format_regex = stage_config.get('option_format_regex', None)
    option_format_regex = f" --option_format_regex '{option_format_regex}' " if option_format_regex else ""

    problem_field = stage_config.get("problem_field", None)
    expected_answer_field = stage_config.get("expected_answer_field", None)
    id_field = stage_config.get("id_field", None)

    cmd = (
        f"python /nemo_run/code/recipes/opensciencereasoning/scripts/filter_problems.py "
        f"{input_file} "
        f"{output_dir}/{OUTPUT_FILE}"
        + (f" --deduplicate" if stage_config.get('deduplicate', False) else "")
        + (f" --remove_images" if stage_config.get('remove_images', False) else "")
        + (f" --dataset_name {stage_config.get('dataset_name', None)}" if stage_config.get('dataset_name') else "")
        + (f" --num_options {stage_config.get('num_options', None)}" if stage_config.get('num_options') else "")
        + (f" --problem_field {problem_field}" if problem_field else "")
        + (f" --expected_answer_field {expected_answer_field}" if expected_answer_field else "")
        + (f" --id_field {id_field}" if id_field else "")
        + option_format_regex
    )
    print(cmd)
    
    wrapped_cmd = wrap_arguments(cmd)
    run_cmd(
        cluster=cluster, 
        container="nemo-skills",
        log_dir=f"{output_dir}/logs",
        expname=expname,
        ctx=wrapped_cmd,
        run_after=run_after,
    )

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
        installation_command="pip install torch sentence-transformers", # TODO remove
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

def generate_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Launch model inference, adds predicted_answer/expected_answer via regex/majority voting, optionally judge, then aggregate per-problem stats.

    Steps:
      1. Call `generate` to produce raw solver outputs.
      2. Run `extract_predictions.py`, which applies `predicted_answer_regex` if provided and can populate expected answers using majority voting.
      3. Optionally judge the generations (math_judge) if `make_judgement` is enabled.
      4. Invoke `aggregate_solutions.py` to compute per-problem correctness and generation pass rate, writing `final_result.jsonl`.
    """
    output_dir = stage_config["output_dir"]
    make_majority_voting = stage_config.get("make_majority_voting", None)
    make_judgement = stage_config.get("make_judgement", None)
    predicted_answer_regex = stage_config.get("predicted_answer_regex", None)

    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    generation_params = generation_kwargs.get("params", {})
    ctx_params = prepare_ctx_kwargs(generation_kwargs.get("ctx_params", {}))
    judge_ctx_args = prepare_ctx_kwargs(judge_kwargs.get("ctx_params", {}))
    judge_params = judge_kwargs.get("params", {})

    generate(
        ctx=wrap_arguments(ctx_params),
        cluster=cluster,
        output_dir=f"{output_dir}/generation",
        expname=f"{expname}_generate_solutions",
        run_after=run_after,
        **generation_params,
    )
    generation_dir = f"{output_dir}/with_predictions"

    predicted_answer_regex_args = f"    --predicted_answer_regex '{predicted_answer_regex}' " if predicted_answer_regex else ""
    majority_voting_args = f"    --majority_voting '{make_majority_voting}' " if make_majority_voting else ""
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/extract_predictions.py "
            f"    --input_dir '{output_dir}/generation' "
            f"    --output_dir '{generation_dir}' "
            f"{predicted_answer_regex_args} "
            f"{majority_voting_args} "
        ),
        cluster=cluster,
        expname=f"{expname}_extract_predictions",
        run_after=f"{expname}_generate_solutions",
    )
    
    if make_judgement:
        generate(
            ctx=wrap_arguments(judge_ctx_args),
            cluster=cluster,
            generation_type="math_judge",
            input_dir=generation_dir,
            output_dir=f"{output_dir}/judgement",
            expname=f"{expname}_judgement",
            run_after=f"{expname}_extract_predictions",
            **judge_params,
        )
        generation_dir = f"{output_dir}/judgement"

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/aggregate_solutions.py "
            f"    --input_dir '{generation_dir}' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --generation_model '{generation_params['model'].split('/')[-1]}' "
        ),
        cluster=cluster,
        expname=expname,
        run_after=[f"{expname}_extract_predictions", f"{expname}_judgement"],
    )

def difficulty_estimation(cluster, expname, run_after, stage_config, **kwargs):
    """Run difficulty estimation generation, judge correctness, and postprocess metrics.

    This stage:
      - Generates multiple solutions per problem using the provided model/prompt.
      - Runs LLM-based judging (math_judge) over those generations to get Yes/No per sample.
      - Postprocesses the judgements to append three keys to the final results file:
        - difficulty_model: the model used for generation
        - pass_rate: decimal fraction of correct judgements (e.g., 0.5)
        - pass_at_n: formatted fraction "correct/total" (e.g., 2/4)

    Note: The judging step extracts predicted answers using the \\boxed{...} convention.
    It will only work out-of-the-box when generations include a final answer in boxed format.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]

    generation_kwargs = stage_config.get("generation_kwargs", {})
    judge_kwargs = stage_config.get("judge_kwargs", {})

    generation_params = generation_kwargs.get("params", {})
    generation_ctx_args = prepare_ctx_kwargs(generation_kwargs.get("ctx_params", {}))

    judge_params = judge_kwargs.get("params", {})
    judge_ctx_params = prepare_ctx_kwargs(judge_kwargs.get("ctx_params", {}))
    

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/remove_redundant_fields.py "
            f"    --input_file '{input_file}' "
            f"    --output_file '{output_dir}/tmp/prepared.jsonl' "
            f"    --fields {shlex.quote(json.dumps(BASE_FIELDS, ensure_ascii=False))} "
        ),
        cluster=cluster,
        expname=f"{expname}_prepare_difficulty_estimation",
        run_after=run_after,
    )

    generate(
        ctx=wrap_arguments(generation_ctx_args),
        cluster=cluster,
        input_file=f"{output_dir}/tmp/prepared.jsonl",
        output_dir=f"{output_dir}/generation",
        expname=f"{expname}-generation",
        run_after=f"{expname}_prepare_difficulty_estimation",
        **generation_params,
    )

    generate(
        ctx=wrap_arguments(judge_ctx_params),
        generation_type="math_judge",
        cluster=cluster,
        input_dir=f"{output_dir}/generation",
        output_dir=f"{output_dir}/judgement",
        expname=f"{expname}-judgement",
        run_after=f"{expname}-generation",
        **judge_params,
    )

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/aggregate_difficulty.py "
            f"    --judgement_dir '{output_dir}/judgement' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --pass_rate_model '{generation_params['model'].split('/')[-1]}' "
        ),
        cluster=cluster,
        exclusive=False,
        run_after=f"{expname}-judgement",
        expname=expname,
    )

def aggregate(cluster, expname, run_after, stage_config, **kwargs):
    """Aggregate per-problem metadata and solutions into a final JSONL.

    This stage invokes `scripts/aggregate_matadata.py` to:
      - Merge metadata from `stage_config["metadata_files"]` (JSON list), if provided.
      - Optionally merge solutions from `stage_config["solutions_path"]`, which should be a glob pattern.
    """
    output_dir = stage_config["output_dir"]
    metadata_files = stage_config.get("metadata_files", [])
    solutions_path = stage_config.get("solutions_path", None)

    solutions_path_arg = f"    --solutions_path {shlex.quote(str(solutions_path))} " if solutions_path is not None else ""
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/aggregate_matadata.py "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"    --metadata_files {shlex.quote(json.dumps(metadata_files, ensure_ascii=False))} "
            f"{solutions_path_arg}"
        ),
        cluster=cluster,
        exclusive=False,
        run_after=run_after,
        expname=expname,
    )

def filter_solutions(cluster, expname, run_after, stage_config, **kwargs):
    """Submit the filtering job with stage-configured correctness, pass-rate, and metadata constraints.

    Supported filters (see `filter_solutions.py`):
      - `only_correct_solutions`: keep only samples marked `is_correct`.
      - `generation_model_pass_rate_range`: JSON `[min, max]` range (min exclusive, max inclusive).
      - `pass_rate_range`: JSON `[min, max]` range over difficulty pass rates.
      - `metadata_values`: dict of field -> allowed values.

    Replace `filter_solutions.py` with your own implementation if custom filtering logic is required.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    only_correct_solutions = stage_config.get("only_correct_solutions", False)
    generation_model_pass_rate_range = stage_config.get("generation_model_pass_rate_range", None)
    pass_rate_range = stage_config.get("pass_rate_range", None)
    metadata_values = stage_config.get("metadata_values", None)

    generation_model_pass_rate_range_arg = f"    --generation_model_pass_rate_range {shlex.quote(json.dumps(generation_model_pass_rate_range, ensure_ascii=False))} " if generation_model_pass_rate_range else ""
    pass_rate_range_arg = f"    --pass_rate_range {shlex.quote(json.dumps(pass_rate_range, ensure_ascii=False))} " if pass_rate_range else ""
    metadata_values_arg = f"    --metadata_values {shlex.quote(json.dumps(metadata_values, ensure_ascii=False))} " if metadata_values else ""
    only_correct_arg = "    --only_correct_solutions " if only_correct_solutions else ""
    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/filter_solutions.py "
            f"    --input_file '{input_file}' "
            f"    --output_file '{output_dir}/{OUTPUT_FILE}' "
            f"{only_correct_arg}"
            f"{generation_model_pass_rate_range_arg} "
            f"{pass_rate_range_arg} "
            f"{metadata_values_arg} "
        ),
        cluster=cluster,
        exclusive=False,
        run_after=run_after,
        expname=expname,
    )


def prepare_for_sft(cluster, expname, run_after, stage_config, **kwargs):
    """Prepare cleaned, instruction-formatted data for SFT training.

    The stage removes non-essential fields from the aggregated solutions and
    then calls `nemo_skills.training.prepare_data` with the provided prompt
    configuration and tokenizer so that the resulting `final_result.jsonl`
    can be used directly for supervised fine-tuning.
    """
    output_dir = stage_config["output_dir"]
    input_file = stage_config["input_file"]
    prompt_config = stage_config["prompt_config"]
    tokenizer = stage_config["tokenizer"]
    prepared_file = f"{output_dir}/tmp/prepared.jsonl"
    output_path = f"{output_dir}/{OUTPUT_FILE}"

    run_cmd(
        ctx=wrap_arguments(
            f"python /nemo_run/code/recipes/opensciencereasoning/scripts/remove_redundant_fields.py "
            f"    --input_file '{input_file}' "
            f"    --output_file '{prepared_file}' "
            f"    --fields {shlex.quote(json.dumps(["problem", "generation"], ensure_ascii=False))} "
        ),
        cluster=cluster,
        expname=f"{expname}_prepare_for_sft",
        run_after=run_after,
    )

    inline_args = stage_config.get("inline_args", "")
    cmd = (
        f"mkdir -p {output_dir} && python -m nemo_skills.training.prepare_data "
        f"    ++input_files='{prepared_file}' "
        f"    ++output_path='{output_path}' "
        f"    ++prompt_config={prompt_config} "
        f"    ++tokenizer={tokenizer} "
        f"    ++filters.drop_multi_boxed=false "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.remove_len_outlier_solutions=false "
        f"    ++filters.trim_solutions=false "
        f"    ++add_unlabeled=True "
        f"    {inline_args}"
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        log_dir=f"{output_dir}/logs",
        expname=expname,
        run_after=f"{expname}_prepare_for_sft"
    )


stages_map = {
    "filter_problems": filter_problems,
    "decontaminate": decontaminate,
    "topics_labeling": topics_labeling,
    "generate_solutions": generate_solutions,
    "difficulty_estimation": difficulty_estimation,
    "aggregate": aggregate,
    "filter_solutions": filter_solutions,
    "prepare_for_sft": prepare_for_sft,
}


if __name__ == "__main__":
    config_dir = Path(__file__).parents[1] / "configs" / "solution_sdg"

    parser = argparse.ArgumentParser(description="OpenMathReasoning-1 solution generation pipeline")
    parser.add_argument(
        "--config_path",
        type=str,
        default=f"{config_dir}/gpt-oss-seed-data.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated list of stages to run. If not specified, runs all stages from the config.",
    )

    args = parser.parse_args()

    config_path = args.config_path
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
        print(f"Running all stages defined in config '{config_path}': {stages_to_run}")

    for stage in stages_to_run:
        if stage not in stages_map:
            raise ValueError(f"Unknown stage specified: '{stage}'. Available stages: {list(stages_map.keys())}")
        if stage not in full_stage_sequence:
            raise ValueError(
                f"Stage '{stage}' requested but not part of the defined sequence for config '{config_path}'. "
                f"Specify one of {full_stage_sequence} or select an appropriate config."
            )

    # --- Common parameters ---
    base_output_dir = config["base_output_dir"]
    suffix = config.get("suffix", Path(config_path).stem)
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

import argparse
import sys
import os
sys.path.append("/home/stoshniwal/Research/llm/NeMo-Skills")

from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import generate, summarize_results
from nemo_skills.pipeline.cli import run_cmd
from pathlib import Path
os.environ['NEMO_SKILLS_CONFIG_DIR']= "/home/stoshniwal/Research/llm/nemo-skills-config/cluster_configs"
os.environ['NEMO_SKILLS_EXTRA_DATASETS'] = "/home/stoshniwal/Research/llm/nemo-skills-recipes/internal-datasets"

MAX_TOKENS = 40000
MAX_GEN_TOKENS = 2048


MODEL_TO_GPUS = {
    "qwen2.5-32b": 8,
    "qwen2.5-14b": 8,
    "qwen2.5-7b": 4,
}


def step_1_preprocess_data(cluster, partition, model, input_dir, output_dir, num_random_seeds, max_samples, sampling_strategy, num_input_samples):
    comparison_dir = str(Path(output_dir) / "comparison_instances")

    preprocess_command = f"python /nemo_run/code/recipes/omr1/scripts/preprocess_genrm.py --input_dir {input_dir} --output_dir {comparison_dir} --max_samples {max_samples} --sampling_strategy {sampling_strategy} --num_random_seeds {num_random_seeds}" + (f" --num_input_samples {num_input_samples}" if num_input_samples is not None else "")

    preprocess_expname = f"comparison-data-prep-{str(Path(model).name)}-{str(Path(input_dir).name)}"
    print(preprocess_expname)
    exp = run_cmd(
        ctx=wrap_arguments(preprocess_command),
        cluster=cluster,
        partition="cpu" if "dfw" in cluster else partition,
        expname=preprocess_expname,
        log_dir=f"{comparison_dir}/logs",
        time_min="00:05:00",
    )
    return preprocess_expname, comparison_dir

def step_2_score_solns(
        cluster, partition, last_exp_name, last_exp_output_dir, model, output_dir, num_random_seeds, temperature=0.7):

    output_dir = os.path.join(output_dir, "comparison_judgment")
    
    exp_names = []
    for random_seed in range(num_random_seeds):
        input_file = os.path.join(last_exp_output_dir, f'output-rs{random_seed}.jsonl')
        output_file = os.path.join(output_dir, f'output-rs{random_seed}.jsonl')

        exp_name = f"gen-rm-comparison-rs{random_seed}"
        exp_names.append(exp_name)
        generate(
            ctx=wrap_arguments(
                f"++input_file={input_file} "
                f"++output_file={output_file} "
                f"++prompt_config=/nemo_run/code/recipes/omr1/prompts/math-genrm.yaml "
                f"++prompt_template=qwen-instruct "
                f"++batch_size=256 "
                f"++skip_filled=True "
                f"++generation_key=gen_rm_comparison "
                f"++inference.tokens_to_generate={MAX_GEN_TOKENS} "
                f"++inference.temperature={temperature} "
            ),
            cluster=cluster,
            model=f"{model}",
            output_dir=output_dir,
            server_type="trtllm",
            server_gpus=MODEL_TO_GPUS[os.getenv("BASE_MODEL", "qwen2.5-14b")],
            server_nodes=1,
            partition=partition,
            dependent_jobs=0,
            expname=f"{exp_name}",
            run_after=last_exp_name,
        )
        
    return exp_names, output_dir


def step_3_postprocess_judgment(cluster, partition, last_exp_name, step_1_output_dir, 
                                step_2_output_dir, output_dir, benchmark):
    step_3_output_dir = os.path.join(output_dir, benchmark)
    postprocess_cmd = f"python /nemo_run/code/recipes/omr1/scripts/postprocess_genrm.py --step_1_output_dir {step_1_output_dir} --step_2_output_dir {step_2_output_dir} --output_dir {step_3_output_dir}"
    print(postprocess_cmd)

    expname = f"postprocess-judgment"
    run_cmd(
        ctx=wrap_arguments(postprocess_cmd),
        cluster=cluster,
        partition="cpu" if "dfw" in cluster else partition,
        expname=expname,
        run_after=last_exp_name,
        log_dir=f"{output_dir}/logs",
        time_min="00:05:00",
    )
    return expname, step_3_output_dir


def step_4_summarize_results(cluster, last_exp_name, last_exp_output_dir, expname, wandb_name, wandb_group, wandb_project):
    run_cmd(
        ctx=wrap_arguments(
            (
                f"NEMO_SKILLS_EXTRA_DATASETS=/nemo_run/code/internal-datasets "
                f"python -m nemo_skills.pipeline.summarize_results "
                f"    {last_exp_output_dir} "
                f"    --wandb_name={wandb_name} "    
                f"    --wandb_group={wandb_group} "
                f"    --wandb_project={wandb_project} "
            )
        ),
        cluster=cluster,
        partition="cpu" if "dfw" in cluster else None,
        run_after=f"{last_exp_name}",
        log_dir=f"{last_exp_output_dir}/logs",
        exclusive=True,
        expname=f"{expname}",
        time_min="00:10:00",
    )
    

def get_expname(args):
    model_path = Path(args.model)
    try:
        model_name = model_path.parent.parent.parent.name
        expname = f"{model_name}"
    except:
        expname = f"{str(Path(args.model).name)}"
    
    suffix = ""
    if args.num_input_samples is not None:
        suffix += f"input-{args.num_input_samples}"
    if args.temperature is not None:
        suffix += f"-temp-{args.temperature}"
    if args.max_samples is not None:
        suffix += f"-max-samples-{args.max_samples}"
    if args.num_random_seeds is not None:
        suffix += f"-num-random-seeds-{args.num_random_seeds}"
    if args.sampling_strategy is not None:
        suffix += f"-{args.sampling_strategy}"

    expname = f"{expname}-{suffix}"
    return expname
    

def main(args):
    benchmark = Path(args.input_dir).name
    expname = get_expname(args)
    args.output_dir = os.path.join(args.output_dir, str(Path(args.input_dir).name) + "_" + expname)

    last_exp_name, step_1_output_dir = step_1_preprocess_data(
        cluster=args.cluster,
        partition=args.partition,
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_random_seeds=args.num_random_seeds,
        max_samples=args.max_samples,
        sampling_strategy=args.sampling_strategy,
        num_input_samples=args.num_input_samples,
    )

    last_exp_name, step_2_output_dir = step_2_score_solns(
        cluster=args.cluster,
        partition=args.partition,
        last_exp_name=last_exp_name,
        last_exp_output_dir=step_1_output_dir,
        model=args.model,
        output_dir=args.output_dir,
        num_random_seeds=args.num_random_seeds,
        temperature=args.temperature,
    )

    last_exp_name, step_3_output_dir = step_3_postprocess_judgment(
        cluster=args.cluster,
        partition=args.partition,
        last_exp_name=last_exp_name,
        step_1_output_dir=step_1_output_dir,
        step_2_output_dir=step_2_output_dir,
        output_dir=args.output_dir,
        benchmark=benchmark,
    )

    if (args.wandb_name is not None) and (args.wandb_group is not None) and (args.wandb_project is not None):
        step_4_summarize_results(
            cluster=args.cluster,
            last_exp_name=last_exp_name,
            last_exp_output_dir=step_3_output_dir,
            expname=expname,
            wandb_name=args.wandb_name,
            wandb_group=args.wandb_group,
            wandb_project=args.wandb_project,
        )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forum pipeline")
    parser.add_argument("--cluster", type=str, required=True, help="Cluster")
    parser.add_argument("--partition", default=None, type=str, help="Partition")
    parser.add_argument("--model", type=str, required=True, help="Comparison model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_random_seeds", type=int, default=8, help="Number of parallel judgments")
    parser.add_argument("--num_input_samples", type=int, default=None, help="Number of solutions with different random seeds used for reranking.")
    parser.add_argument("--max_samples", type=int, required=False, default=8)
    parser.add_argument("--sampling_strategy", type=str, required=False, default="linear")
    parser.add_argument("--temperature", type=float, required=False, default=0.7, help="Temperature for generation")
    parser.add_argument("--wandb_name", type=str, required=False, default=None, help="Wandb name")  
    parser.add_argument("--wandb_group", type=str, required=False, default=None, help="Wandb group")
    parser.add_argument("--wandb_project", type=str, required=False, default=None, help="Wandb project")
    args = parser.parse_args()
    main(args)

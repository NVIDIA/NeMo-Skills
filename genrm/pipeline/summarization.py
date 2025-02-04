import argparse

from nemo_skills.pipeline import wrap_arguments
from nemo_skills.pipeline.cli import generate

OUTPUT_DIR = "/data/aops-pipeline/experiments/{expname}"
NUM_RANDOM_SEEDS = 1
SUMMARIZATION_MODEL = "Qwen2.5-32B-Instruct"

MAX_TOKENS = 32768
PARTITION = "interactive"
# PARTITION = None


def summarize_solns(model_size, cluster):
    for random_seed in range(NUM_RANDOM_SEEDS):
        output_dir = f"/experiments/eval_models/distill-r1-eval/{model_size}-summary/"
        generate(
            ctx=wrap_arguments(
                f"++input_file=/experiments/eval_models/distill-r1-eval/{model_size}/eval-results-judged/nvmath/output-rs{random_seed}.jsonl "
                f"++prompt_config=/nemo_run/code/genrm/prompts/summarize-solution.yaml "
                f"++prompt_template=qwen-instruct "
                f"++batch_size=512 "
                f"++inference.temperature=0.0 "
                f"++inference.tokens_to_generate=3072 "
                f"++output_file={output_dir}/eval-results/nvmath/output-rs{random_seed}.jsonl "
            ),
            cluster=cluster,
            model=f"/hf_models/{SUMMARIZATION_MODEL}",
            server_type="sglang",
            server_gpus=8,
            server_nodes=1,
            partition=PARTITION,
            output_dir=output_dir,
            dependent_jobs=0,
            expname=f"{model_size}-summarize-solutions-{random_seed}",
            server_args=f"--context-length {MAX_TOKENS} ",
        ),


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forum pipeline")
    parser.add_argument("--cluster", type=str, required=True, help="Cluster")
    parser.add_argument("--model_size", type=str, required=True, help="Model Size")

    args = parser.parse_args()

    # Summarize solns
    summarize_solns(model_size=args.model_size, cluster=args.cluster)

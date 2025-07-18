from nemo_skills.pipeline.cli import eval, genselect, wrap_arguments

size_to_eval_gpus = {
    '1.5b': 1,
    '7b': 2,
    '14b': 4,
    '32b': 8,
}

eval_tokens = 65536

math_seeds = 64
code_seeds = 64
science_seeds = 64

run_genselect = True

cluster = "slurm"

model_sizes = ['1.5B', '7B', '14B', '32B']

output_dir = "/workspace/open-reasoning-evals"


def eval_aai(model_size):
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} "),
        cluster=cluster,
        expname=f"eval-aai-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"aai",
        judge_model="/workspace/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=8,
        judge_server_args="--context-length 84000",
    )


def eval_math(model_size):
    math_benchmarks = [
        "aime24",
        "aime25",
        "hmmt_feb25",
        "brumo25",
    ]
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
        cluster=cluster,
        expname=f"eval-math-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=','.join([f"{bench}:{math_seeds}" for bench in math_benchmarks]),
        num_jobs=64,
    )

    if run_genselect:
        for bench in math_benchmarks:
            genselect(
                ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
                cluster=cluster,
                expname=f"genselect-{bench}-{model_size}",
                run_after=f"eval-math-{model_size}",
                output_dir=f"{output_dir}/{model_size}-genselect/{bench}",
                model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
                server_type="sglang",
                server_gpus=size_to_eval_gpus[model_size],
                preprocess_args=f"++input_dir={output_dir}/{model_size}/eval-results/{bench}",
                num_random_seeds=math_seeds,
            )


def eval_code(model_size):
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
        cluster=cluster,
        expname=f"eval-code-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        split="test_v6_2408_2505",
        benchmarks=f"livecodebench:{code_seeds}",
    )


def eval_science(model_size):
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
        cluster=cluster,
        expname=f"eval-science-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"gpqa:{science_seeds}",
    )
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
        cluster=cluster,
        expname=f"eval-science-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"mmlu-pro:{science_seeds}",
        # num_chunks=10,  # parallelize 10x for faster eval on slurm
    )
    eval(
        ctx=wrap_arguments(f"++inference.tokens_to_generate={eval_tokens} ++inference.temperature=0.6 "),
        cluster=cluster,
        expname=f"eval-science-{model_size}",
        output_dir=f"{output_dir}/{model_size}",
        model=f"/workspace/OpenReasoning-Nemotron-{model_size}",
        server_type="sglang",
        server_gpus=size_to_eval_gpus[model_size],
        benchmarks=f"hle:{science_seeds}",
        # num_chunks=4,  # parallelize 4x for faster eval on slurm
        judge_model="/workspace/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=8,
        judge_server_args="--context-length 64000",
    )


for model_size in model_sizes:
    eval_aai(model_size)
    eval_math(model_size)
    eval_code(model_size)
    eval_science(model_size)

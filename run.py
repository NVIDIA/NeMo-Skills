from nemo_skills.pipeline.cli import eval, wrap_arguments

assert False, "Currently testing in-docker"

eval(
    ctx=wrap_arguments(""),
    model="meta/llama-3.1-8b-instruct",
    cluster="local",
    server_type="openai",
    server_address="https://integrate.api.nvidia.com/v1",
    benchmarks="aime25",
    installation_command="cd /nemo_run/code && pip install -e . && pip install --upgrade 'git+https://github.com/NVIDIA-NeMo/Eval#subdirectory=packages/nemo-evaluator-launcher[all]'",
    output_dir="/workspace/test_3",
    skip_hf_home_check=True,
    generation_module="nemo_evaluator_runner.py",
)

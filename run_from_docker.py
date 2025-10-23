from nemo_skills.pipeline.cli import eval, wrap_arguments

eval(
    ctx=wrap_arguments(""),
    model="meta/llama-3.1-8b-instruct",
    server_type="openai",
    server_address="https://integrate.api.nvidia.com/v1",
    benchmarks="aime25",
    # installation_command="cd /nemo_run/code && pip install -e .",
    # installation_command="pip install --upgrade 'git+https://github.com/NVIDIA-NeMo/Eval#subdirectory=packages/nemo-evaluator-launcher'",
    installation_command="pip install -e 'nemo_evaluator_launcher @ /home/agronskiy/code/nvidia-nemo-eval/packages/nemo-evaluator-launcher'",
    output_dir="/workspace/test_3",
    skip_hf_home_check=True,
    generation_module="nemo_evaluator_runner.py",
)

from nemo_skills.pipeline.cli import eval, wrap_arguments

eval(
    ctx=wrap_arguments(""),
    cluster="local",
    model="meta/llama-3.1-8b-instruct",
    server_type="openai",
    server_address="https://integrate.api.nvidia.com/v1",
    benchmarks="aime25",
    installation_command="cd /nemo_run/code && pip install -e .",
    output_dir="/workspace/test_2",
    skip_hf_home_check=True,
    generation_module="/nemo_run/code/some.py",
)

import argparse

from nemo_skills.dataset.prepare import prepare_datasets
from nemo_skills.pipeline.cli import convert, eval, generate, run_cmd, sft_nemo_rl, train, wrap_arguments

workspace = "/lustre/fsw/portfolios/llmservice/users/wedu/experiments/ns-tests/nemo-aligner"
cluster = 'oci'
cmd = (
    f"cd {workspace} && "
    f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/wedu/unit-test-slurm/tests && "
    f"wget $DOWNLOAD_PREFIX/slurm-tests/check_sft_results.py && "
    f"python check_sft_results.py --workspace {workspace} "
)
run_cmd(
    ctx=wrap_arguments(cmd),
    cluster=cluster,
    expname=f"download-assets",
    log_dir=f"{workspace}/logs",
)

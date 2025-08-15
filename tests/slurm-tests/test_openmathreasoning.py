import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from nemo_skills.pipeline.cli import run_cmd, wrap_arguments


def run_sft_cmd(cmd: list[str], env=None, timeout=None) -> str:
    """Run a shell command and return combined stdout+stderr."""
    p = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


REPO_ROOT = Path(__file__).resolve().parents[2]
SIMPLIFIED_RECIPE = REPO_ROOT / "recipes" / "openmathreasoning" / "scripts" / "simplified_recipe.py"


def build_launch_cmd(
    cluster: str,
    backend: str,
    workspace: str,
    num_gpus: int,
    project: str,
    expname: str,
    disable_wandb: bool,
) -> list[str]:
    """Build CLI for simplified_recipe.py (no sbatch here; recipe handles scheduling)."""
    assert SIMPLIFIED_RECIPE.exists(), f"simplified_recipe.py not found at {SIMPLIFIED_RECIPE}"
    base = [
        "python",
        str(SIMPLIFIED_RECIPE),
        "--cluster",
        cluster,
        "--workspace",
        workspace,
        "--num_gpus",
        str(num_gpus),
        "--training_backend",
        backend,
        "--expname_prefix",
        expname,
    ]
    if disable_wandb:
        base.append("--disable_wandb")
    else:
        if project:
            base += ["--wandb_project", project]
    return base


def test_openmathreasoning_end2end(pytestconfig):
    """
    Submit pipeline; wait for baseline and final summarize-results on remote;
    fetch metrics (W&B first if enabled, else remote JSON); assert thresholds.
    """
    # --- read CLI options ---
    cluster = pytestconfig.getoption("cluster")
    backend = pytestconfig.getoption("backend")
    workspace = pytestconfig.getoption("workspace")
    num_gpus_opt = pytestconfig.getoption("num_gpus")
    num_gpus = int(num_gpus_opt) if num_gpus_opt is not None else 8
    wandb_project = pytestconfig.getoption("wandb_project")
    expname = pytestconfig.getoption("expname")
    disable_wandb = pytestconfig.getoption("disable_wandb")

    # --- Launch SFT jobs  ---
    cmd = build_launch_cmd(cluster, backend, workspace, num_gpus, wandb_project, expname, disable_wandb)
    print(f"[pytest] Launch: {' '.join(cmd)}")
    print(f"[pytest] W&B disabled: {disable_wandb}")
    os.environ.setdefault("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK", "1")
    run_sft_cmd(cmd)

    # --- Launch SFT results valiation jobs ---
    cmd = (
        f"cd {workspace} && "
        f"export DOWNLOAD_PREFIX=https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/refs/heads/wedu/unit-test-slurm/tests && "
        f"wget $DOWNLOAD_PREFIX/slurm-tests/check_sft_results.py && "
        f"python check_sft_results.py --workspace {workspace} "
    )
    run_cmd(
        ctx=wrap_arguments(cmd),
        cluster=cluster,
        expname=f"check-sft-resulst-for-{backend}",
        log_dir=f"{workspace}/logs",
        run_after=[f"{expname}-baseline-summarize-results", f"{expname}-final-eval-summarize-results"],
    )

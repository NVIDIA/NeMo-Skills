import os

import pytest


def pytest_addoption(parser):
    """Register minimal CLI options for the test suite."""
    parser.addoption("--cluster", action="store", default=None, help="Cluster name (e.g., oci, local)")
    parser.addoption("--backend", action="store", default=None, help="Training backend (nemo-aligner or nemo-rl)")
    parser.addoption("--workspace", action="store", default=None, help="Workspace path")
    parser.addoption("--num_gpus", type=int, default=8, help="Number of GPUs")  # typed + default
    parser.addoption("--wandb_project", action="store", default=None, help="W&B project name")
    parser.addoption("--expname", action="store", default=None, help="Experiment name prefix")
    parser.addoption("--disable_wandb", action="store_true", help="Disable W&B logging")


@pytest.fixture(autouse=True)
def inject_cli_opts_into_env(pytestconfig, monkeypatch):
    """Inject CLI options into environment variables used by the tests."""
    opt_map = {
        "cluster": ("CLUSTER", "local"),
        "backend": ("TRAINING_BACKEND", "nemo-aligner"),
        "workspace": ("WORKSPACE", "/workspace"),
        "num_gpus": ("NUM_GPUS", "8"),
        "wandb_project": ("WANDB_PROJECT", "nemo-skills"),
        "expname": ("EXPNAME_PREFIX", "test-pipeline"),
    }
    for cli_key, (env_key, default_val) in opt_map.items():
        # getoption expects the dest name without leading dashes
        val = pytestconfig.getoption(cli_key)
        if val is None:
            val = os.getenv(env_key, default_val)
        monkeypatch.setenv(env_key, str(val))

    # W&B toggle -> env
    if pytestconfig.getoption("disable_wandb"):
        monkeypatch.setenv("DISABLE_WANDB", "1")
    else:
        if "DISABLE_WANDB" not in os.environ:
            monkeypatch.setenv("DISABLE_WANDB", "0")

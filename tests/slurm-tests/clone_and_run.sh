#!/bin/bash

# schedule with chron on a machine with corresponding cluster config setup, e.g.
# @weekly NEMO_SKILLS_CONFIG_DIR=<path to the configs dir> <path to a copy of this script> <cluster name>
# the metrics will be logged in w&b and you will get emails on failure as long as it's configured in your config

LOCAL_WORKSPACE=/tmp/nemo-skills-slurm-ci

rm -rf $LOCAL_WORKSPACE
mkdir -p $LOCAL_WORKSPACE
cd $LOCAL_WORKSPACE
git clone https://github.com/NVIDIA/NeMo-Skills.git
cd NeMo-Skills

uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .

./tests/slurm-tests/run_all.sh

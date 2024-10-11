# Prerequisites

To get started first install the repo (python 3.10+). Either clone and run `pip install -e .` or install directly with

```
pip install git+https://github.com/Kipok/NeMo-Skills.git
```

Then prepare the data.

```
python -m nemo_skills.dataset.prepare
```

If you're only interested in a subset of datasets (e.g. only math-related or code-related), run with
`--dataset_groups ...` and if you only need a couple of specific datasets, list them directly e.g.

```
python -m nemo_skills.dataset.prepare gsm8k human-eval mmlu ifeval
```

If you have the repo cloned locally, the data files will be available inside `nemo_skills/dataset/<benchmark>/<split>.jsonl`
and if you installed from pip, they will be downloaded to wherever the repo is installed, which you can figure out by running
```
python -c "import nemo_skills; print(nemo_skills.__path__)"
```

You might also need define the following environment variables in your `~/.bashrc`

```
export WANDB_API_KEY=<your wandb api key if you want to use it for logging (can opt-out with --disable_wandb)>
export HF_TOKEN=<if you plan to use gated models such as llama3>
export OPENAI_API_KEY=<your openai key if you plan to use openai models>
export NVIDIA_API_KEY=<your Nvidia api key if you plan to use Nvidia-hosted models>
```

Finally, you will typically need to create an appropriate "cluster config" where you define how you want to run
your jobs and what to mount in the containers. Please read on to learn more about that.

## General information

All of the scripts inside [nemo_skills/pipeline](/nemo_skills/pipeline) can be called in 3 equivalent ways.
E.g. to [evaluate](/docs/evaluation.md) a model on 10 samples you might run it like this.

1. Through `ns` command-line entrypoint

   ```
   ns eval \
       --cluster local \
       --server_type openai \
       --model meta/llama-3.1-8b-instruct \
       --server_address https://integrate.api.nvidia.com/v1 \
       --benchmarks gsm8k:0,math:0 \
       --output_dir /workspace/test-eval \
       ++max_samples=10
   ```

2. By directly calling relevant Python module

   ```
   python -m nemo_skills.pipeline.eval \
       --cluster local \
       --server_type openai \
       --model meta/llama-3.1-8b-instruct \
       --server_address https://integrate.api.nvidia.com/v1 \
       --benchmarks gsm8k:0,math:0 \
       --output_dir /workspace/test-eval \
       ++max_samples=10
   ```

3. By calling the corresponding Python function

   ```python
   from nemo_skills.pipeline import wrap_arguments
   from nemo_skills.pipeline.cli import eval

   eval(
       cluster="local",
       server_type="openai",
       model="meta/llama-3.1-8b-instruct",
       server_address="https://integrate.api.nvidia.com/v1",
       benchmarks="gsm8k:0,math:0",
       output_dir="/workspace/test-eval",
       ctx=wrap_arguments("++max_samples=10"),  # arguments of the underlying script need to be wrapped
   )
   ```

All of the scripts inside [nemo_skills/pipeline](/nemo_skills/pipeline) accept `--cluster` argument which you can use
to control where the job gets executed. That argument picks up one of the configs inside your local [cluster_configs](/cluster_configs/)
folder by default, but you can specify another location with `--config_dir` or set it in `NEMO_SKILLS_CONFIG_DIR` env variable.
You can also use `NEMO_SKILLS_CONFIG` env variable instead of the `--cluster` parameter.
The cluster config defines an executor (local or slurm), mounts for data/model access and (slurm-only) various parameters
such as account, partition, ssh-tunnel arguments and so on.

We use [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) for managing our experiments with local and slurm-based
execution supported (please open an issue if you need to run our code on other kinds of clusters).
This means that even if you need to submit jobs on slurm, you do it from your local machine by defining an
appropriate cluster config and nemo-run will package and upload your code, data and manage
all complexities of slurm scheduling. Check their documentation to learn how to fetch logs, check status,
cancel jobs, etc.

> **_NOTE:_**  NeMo-Run will only package the code tracked by git (as well as all jsonl files from `nemo_skills/dataset`).
> Any non-tracked files will not be automatically available inside the container or uploaded to slurm.

We use [Hydra](https://hydra.cc/docs/1.3/intro/) for most of the scripts, so
it's a good idea to read through their documentation if that's the first time you see it.

Note that some of our scripts (most of what's inside [nemo_skills/pipeline](/nemo_skills/pipeline)) use a mix of normal
command-line arguments and Hydra style config overrides (usually formatted as `++arg_name`). Whenever you
see this, it means that the regular `--arg_name` parameters are used to control the wrapper script itself and
all other parameters are directly passed into the underlying `nemo_skills/...` script called by the wrapper.

## Local execution

To run scripts locally we use docker containers, so make sure you have
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
set up on your machine.

All of our scripts assume that data or models are mounted inside the appropriate container so before running any
commands make sure to modify [cluster_configs/example-local.yaml](cluster_configs/example-local.yaml). It's convenient
to rename it to local.yaml (so you can use `--cluster local`) after you defined necessary mounts.

Most of our containers are quite heavy, so the first time you run a job that requires a large container, it will take
a while to pull it. You can manually run `docker pull <container>` for all containers defined in the local config
to cache them.

## Slurm jobs

If you're running on slurm, you need to define some additional information inside cluster config.

Populate the commented out fields inside [cluster_configs/example-slurm.yaml](cluster_configs/example-slurm.yaml).
It's convenient to rename it to slurm.yaml (so you can use `--cluster slurm`) or a cluster name if you use multiple slurm clusters.

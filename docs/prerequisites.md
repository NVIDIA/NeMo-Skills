# Prerequisites

Our code requires
- Python >= 3.8
- CUDA >= 12.0 (for TensorRT-LLM support)

Since our pipeline is very large-scale, all of the scripts are meant to be run on a Slurm cluster.
To make it easier to test and develop the code, we also support local execution which mimics the Slurm workflow.

We use [Hydra](https://hydra.cc/docs/1.3/intro/) for most of the scripts, so
it's a good idea to read through their documentation if that's the first time you see it.

Note that some of our scripts (most of what's inside [pipeline](/pipeline)) use a mix of normal
command-line arguments and Hydra style config overrides (usually formatted as `++arg_name`). Whenever you
see this, it means that the regular `--arg_name` parameters are used to control the wrapper script itself and
all other parameters are directly passed into the underlying `nemo_skills/...` script called by the wrapper.

Also note that some of the scripts inside [pipeline](/pipeline) will always
finish with MPI error even if they worked correctly, so do not rely on the
scripts returned code for any downstream tasks
(e.g. don't use Slurm's `afterok` and use `afterany` instead).

## Local

The following are prerequisites for running scripts locally. If you're running on a Slurm
cluster, you can skip to the [Slurm prerequisites](#slurm) section.

1. Make sure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up on your machine.
2. Either [set up docker to run as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user) or change `docker_cmd` inside [cluster_configs/local.yaml](/cluster_configs/local.yaml) to "sudo docker".
3. Run and add the export commands to your shell configuration file (e.g. `~/.bashrc`).

   ```
   export NEMO_SKILLS_CONFIG=<path to this repo>/cluster_configs/local.yaml
   # only required for pipeline/run_sft_and_eval.py - will be used to save sft checkpoints and eval results
   export NEMO_SKILLS_RESULTS=<where you want to save results>
   # only required for running SFT jobs
   export NEMO_SKILLS_DATA=<folder containing training data file(s)>
   export WANDB_API_KEY=<your weights and biases api key if you want to use it for SFT logging>
   ```

4. Install the project and required dependencies: `pip install -e .`
5. Download and prepare all benchmark datasets: `./datasets/prepare_all.sh`

## Slurm

The following are prerequisites for running scripts on a Slurm cluster.

1. Populate [cluster_configs/slurm.yaml](cluster_configs/slurm.yaml) with required information.
2. Run and add the export commands to your shell configuration file (e.g. `~/.bashrc`).

   ```
   export NEMO_SKILLS_CONFIG=<path to this repo>/cluster_configs/slurm.yaml
   # only required for pipeline/run_sft_and_eval.py - will be used to save sft checkpoints and eval results
   export NEMO_SKILLS_RESULTS=<where you want to save results>
   # only required for running SFT jobs
   export NEMO_SKILLS_DATA=<folder containing training data file(s)>
   export WANDB_API_KEY=<your weights and biases api key if you want to use it for SFT logging>
   ```

3. We try to avoid installing packages on Slurm login nodes, but there is still one package that's required: `pip install --user pyyaml`

4. Download and prepare all benchmark datasets: `./datasets/prepare_all.sh`

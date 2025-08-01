# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import os
import shlex
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nemo_run as run
from nemo_run.core.execution.docker import DockerExecutor
from nemo_run.core.execution.local import LocalExecutor
from nemo_run.core.execution.slurm import SlurmJobDetails, get_packaging_job_key
from torchx.specs.api import AppState

from nemo_skills.pipeline.utils.cluster import get_env_variables, get_tunnel, temporary_env_update, tunnel_hash
from nemo_skills.pipeline.utils.mounts import get_mounts_from_config, get_unmounted_path
from nemo_skills.pipeline.utils.packager import get_packager, get_registered_external_repo
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command
from nemo_skills.utils import get_logger_name, remove_handlers

LOG = logging.getLogger(get_logger_name(__file__))


# keeping a global variable for first submitted experiment (per cluster) and reusing it by default
# we are using ssh tunnel as a proxy for cluster identity, since even if other parameters are different
# we can still reuse code as long as ssh matches
REUSE_CODE_EXP = {}


# caching the status assuming it doesn't change while experiment is being scheduled
# otherwise this results in too many ssh calls
@lru_cache
def get_exp_handles(expname: str, ignore_finished=True, ignore_exp_not_exists=True) -> list[str]:
    """Will return the handles of the tasks in the experiment.

    If ignore_finished=True, will only return handles for the tasks
    that are not yet finished. Useful for filtering handles to set dependencies on.

    If ignore_exp_not_exists=True, will not raise an error if the experiment does not exist.

    TODO: it's still possible that job submission fails if the tasks exist when this function
          is called, but finish before nemo-run submits a new job (which might take minutes)
    """

    def _get_handles(exp: run.Experiment):
        handles = []
        status_dict = exp.status(return_dict=True)
        assert status_dict, f"No status found for experiment {exp._id}"
        for _, status_info in status_dict.items():
            if not ignore_finished or (
                status_info['status']
                in [
                    AppState.RUNNING,
                    AppState.PENDING,
                    AppState.SUBMITTED,
                    AppState.UNKNOWN,
                ]
            ):
                handles.append(status_info['handle'])
                continue
        return handles

    # if we are given an experiment object, we can directly get the handles
    if isinstance(expname, run.Experiment):
        return _get_handles(expname)

    try:
        with run.Experiment.from_title(expname) as exp:
            return _get_handles(exp)
    except FileNotFoundError:
        try:
            with run.Experiment.from_id(expname) as exp:
                return _get_handles(exp)
        except AssertionError:
            if ignore_exp_not_exists:
                LOG.warning("Experiment %s not found!", expname)
                return []
            raise ValueError(f"Experiment {expname} not found!")


def get_sandbox_command(cluster_config):
    if cluster_config['executor'] == 'none':
        return "python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server"
    return "/entrypoint.sh && /start.sh"


@dataclass(kw_only=True)
class CustomJobDetails(SlurmJobDetails):
    # we have 1 srun per sub-task (e.g. server/sandbox/main), but only a single sbatch
    srun_prefix: str = "main"
    sbatch_prefix: str = ""

    @property
    def stdout(self) -> Path:
        return Path(self.folder) / f"{self.sbatch_prefix}%j_sbatch.log"

    @property
    def srun_stdout(self) -> Path:
        return Path(self.folder) / f"{self.srun_prefix}%j_srun.log"

    @property
    def stderr(self) -> Path:
        return Path(self.folder) / f"{self.sbatch_prefix}%j_sbatch.log"

    @property
    def srun_stderr(self) -> Path:
        return Path(self.folder) / f"{self.srun_prefix}%j_srun.log"

    @property
    def ls_term(self) -> str:
        """This term will be used to fetch the logs.

        The command used to list the files is ls -1 {ls_term} 2> /dev/null
        """
        assert self.folder
        return os.path.join(self.folder, "*%j_srun.log")


@dataclass(kw_only=True)
class CustomJobDetailsRay(CustomJobDetails):
    # ray jobs have a custom logs structure
    ray_log_prefix: str = "ray-%j-"

    @property
    def ls_term(self) -> str:
        assert self.folder
        return os.path.join(self.folder, "ray-%j-job*")


def get_executor(
    cluster_config,
    container,
    num_nodes,
    tasks_per_node,
    gpus_per_node,
    job_name,
    log_dir,
    log_prefix: str = "main",
    mounts=None,
    partition=None,
    time_min=None,
    dependencies=None,
    extra_package_dirs: tuple[str] | None = None,
    heterogeneous=False,
    het_group=None,
    total_het_groups=None,
    slurm_kwargs: dict | None = None,
    overlap: bool = False,
    with_ray: bool = False,
):
    env_vars = get_env_variables(cluster_config)
    config_mounts = get_mounts_from_config(cluster_config)

    if mounts is None:
        mounts = config_mounts
    if extra_package_dirs is not None:
        extra_package_dirs = tuple(extra_package_dirs)
    packager = get_packager(extra_package_dirs=extra_package_dirs)

    if cluster_config["executor"] != "slurm":
        if num_nodes > 1:
            raise ValueError("Local executor does not support multi-node execution")

    if cluster_config["executor"] == "none":
        return LocalExecutor()

    if cluster_config["executor"] == "local":
        env_vars["PYTHONUNBUFFERED"] = "1"  # this makes sure logs are streamed right away
        return DockerExecutor(
            container_image=container,
            packager=packager,
            ipc_mode="host",
            volumes=mounts,
            ntasks_per_node=1,
            # locally we are always asking for all GPUs to be able to select a subset with CUDA_VISIBLE_DEVICES
            num_gpus=-1 if gpus_per_node is not None else None,
            network="host",
            env_vars=env_vars,
            additional_kwargs={"entrypoint": ""},
        )

    if not heterogeneous:
        env_vars["SLURM_MASTER_NODE"] = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
    else:
        # master node will be within the same group
        env_vars["SLURM_MASTER_NODE"] = (
            f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{het_group} | head -n1)"
        )
        # in addition defining master nodes for all groups to allow communication
        for group in range(total_het_groups):
            env_vars[f"SLURM_MASTER_NODE_HET_GROUP_{group}"] = (
                f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{group} | head -n1)"
            )

    if gpus_per_node is not None and gpus_per_node > 0:
        partition = partition or cluster_config.get("partition")
    else:
        partition = partition or cluster_config.get("cpu_partition") or cluster_config.get("partition")
        if partition == cluster_config.get("cpu_partition"):
            # by default we use exclusive if no gpus are needed and use non-exclusive if gpus are required
            # as cpu jobs almost always need more resources than automatically allocated by slurm
            if slurm_kwargs is None:
                slurm_kwargs = {}
            slurm_kwargs["exclusive"] = True

    if 'timeouts' not in cluster_config:
        timeout = "10000:00:00:00"
    else:
        timeout = cluster_config["timeouts"][partition]

    additional_parameters = {'time_min': time_min} if time_min is not None else {}
    if cluster_config.get('mail_type') is not None:
        additional_parameters['mail_type'] = cluster_config['mail_type']
    if cluster_config.get('mail_user') is not None:
        additional_parameters['mail_user'] = cluster_config['mail_user']
    srun_args = [
        "--no-container-mount-home",
        "--mpi=pmix",
        '--wait=10',
        # we need to be explicit about this in srun as commands might need to run in parallel
        f"--ntasks-per-node={tasks_per_node}",
        f"--nodes={num_nodes}",
        # NeMo-run should take care of this, but we'll put it here temporarily
        f"--container-env={','.join([k.strip() for k in env_vars.keys()])}",
    ]
    if overlap:
        srun_args.append("--overlap")
    if not cluster_config.get("disable_gpus_per_node", False) and gpus_per_node is not None:
        srun_args.append(f"--gpus-per-node={gpus_per_node}")

    dependency_type = cluster_config.get("dependency_type", "afterany")
    job_details_class = CustomJobDetailsRay if with_ray else CustomJobDetails

    return run.SlurmExecutor(
        account=cluster_config["account"],
        partition=partition,
        nodes=num_nodes,
        ntasks_per_node=tasks_per_node,
        tunnel=get_tunnel(cluster_config),
        container_image=container,
        container_mounts=mounts,
        time=timeout,
        additional_parameters=additional_parameters,
        packager=packager,
        gpus_per_node=gpus_per_node if not cluster_config.get("disable_gpus_per_node", False) else None,
        srun_args=srun_args,
        job_details=job_details_class(
            job_name=cluster_config.get("job_name_prefix", "") + job_name,
            folder=get_unmounted_path(cluster_config, log_dir),
            srun_prefix=log_prefix + '_' + job_name + '_',
            sbatch_prefix=job_name + '_',
        ),
        wait_time_for_group_job=0.01,
        monitor_group_job_wait_time=20,
        dependencies=dependencies,
        dependency_type=dependency_type,
        heterogeneous=heterogeneous,
        env_vars=env_vars,
        **(slurm_kwargs or {}),
    )


def install_packages_wrap(cmd, installation_command: str | None = None):
    """Wraps the command to install packages if provided."""
    if installation_command:
        # Generate a unique ID for this job and set it as an environment variable
        # All processes in the same job will share this environment variable
        job_uuid = str(uuid.uuid4())
        lock_file = f"/tmp/pip_install_{job_uuid}_lock"

        # Use environment variable to share the UUID across processes
        setup_env = f"export NEMO_SKILLS_JOB_UUID={job_uuid}"

        # Simple installation guard - first process to create lock file installs packages
        install_guard = (
            f"{setup_env} && "
            f"if ! [ -f {lock_file} ]; then "
            f"echo 'Starting package installation with UUID: {job_uuid}'; "
            f"touch {lock_file}; "
            f"echo 'Installing packages: {installation_command}'; "
            f"if {installation_command}; then "
            f"echo 'Package installation completed successfully'; "
            f"echo 'done' > {lock_file}; "
            f"else "
            f"echo 'Package installation failed'; "
            f"echo 'failed' > {lock_file}; "
            f"exit 1; "
            f"fi; "
            f"else "
            f"echo 'Waiting for package installation to complete (UUID: {job_uuid})'; "
            f"while [ ! -f {lock_file} ] || [ \"$(cat {lock_file} 2>/dev/null)\" != \"done\" ]; do "
            f"if [ -f {lock_file} ] && [ \"$(cat {lock_file} 2>/dev/null)\" = \"failed\" ]; then "
            f"echo 'Package installation failed in another process'; "
            f"exit 1; "
            f"fi; "
            f"sleep 1; "
            f"done; "
            f"echo 'Package installation completed by another process'; "
            f"fi"
        )

        return f"{install_guard} && {cmd}"
    return cmd


# TODO: this function has become too cumbersome to use with all recently added support
#       we should make it simpler by perhaps removing separate logic for server/sandbox
#       and supporting them through a list of cmds directly
#       should also make heterogenous logic very clear and more robust
#       and all parameters that can be list should be list for consistency
def add_task(
    exp,
    cmd: str | list[str],
    task_name,
    cluster_config,
    container: str | list[str],
    num_tasks: int | list[int] = 1,
    num_gpus=None,
    num_nodes=1,
    log_dir=None,
    partition=None,
    time_min=None,
    with_sandbox=False,
    sandbox_port: int | None = None,
    server_config=None,
    reuse_code_exp: str | run.Experiment | None = None,
    reuse_code: bool = True,
    task_dependencies: list[str] = None,
    run_after: str | list[str] | None = None,
    get_server_command=get_server_command,
    extra_package_dirs: list[str] | None = None,
    slurm_kwargs: dict | None = None,
    heterogeneous: bool = False,
    with_ray: bool = False,
    installation_command: str | None = None,
    dry_run: bool = False,
):
    """Wrapper for nemo-run exp.add to help setting up executors and dependencies.

    Note that there are two parameters that control dependencies.
        - task_dependencies: list of tasks that this task depends on **within the same experiment**
        - run_after: a string with experiment name or a list of experiment names that this task
          should run after. Will schedule dependencies on all tasks inside `run_after` experiments.
          It needs to already be launched and running.

    Example of how to set task_dependencies:

    with get_exp(expname, cluster_config) as exp:
        task1 = add_task(exp, ...)
        task2 = add_task(exp, ..., task_dependencies=[task1])

    You can use `reuse_code_exp` to reuse the code from another experiment
    (and thus avoid costly packaging/ssh uploading). You can provide either experiment
    name or the experiment object itself.

    By default we will reuse the code of the first submitted experiment.
    If you want to avoid this, set `reuse_code=False`.

    installation_command argument only affects "main" task, not server or sandbox.
    """
    if run_after is not None and cluster_config["executor"] == "slurm":
        if isinstance(run_after, (str, run.Experiment)):
            run_after = [run_after]
        dependencies = []
        for dep_expname in run_after:
            exp_handles = get_exp_handles(dep_expname)
            if len(exp_handles) == 0:
                LOG.warning(
                    "No pending or running tasks found for experiment %s, cannot set dependencies.", dep_expname
                )
            dependencies.extend(exp_handles)
        if len(dependencies) == 0:
            dependencies = None
    else:
        dependencies = None

    if server_config is None and num_gpus is None and cluster_config['executor'] == "slurm":
        if not cluster_config.get('cpu_partition'):
            num_gpus = 1

    if sandbox_port is None:
        sandbox_port = get_free_port(strategy="random")

    het_group = 0
    het_group_indices = []
    total_het_groups = (server_config is not None) + bool(cmd) + with_sandbox

    LOG.info("Adding a task with commands:")

    commands = []
    executors = []
    # assuming server always has the largest resources request, so it needs to go first
    if server_config is not None:
        server_cmd, num_server_tasks = get_server_command(**server_config, cluster_config=cluster_config)
        if 'container' not in server_config:
            server_container = cluster_config["containers"][server_config['server_type']]
        server_executor = get_executor(
            cluster_config=cluster_config,
            container=server_container,
            num_nodes=server_config['num_nodes'],
            tasks_per_node=num_server_tasks,
            gpus_per_node=server_config['num_gpus'],
            partition=partition,
            time_min=time_min,
            dependencies=dependencies,
            job_name=task_name,
            log_dir=log_dir,
            log_prefix="server",
            extra_package_dirs=extra_package_dirs,
            slurm_kwargs=slurm_kwargs,
            heterogeneous=heterogeneous,
            het_group=het_group,
            total_het_groups=total_het_groups,
            with_ray=with_ray,
        )
        if cluster_config["executor"] != "slurm" and num_server_tasks > 1:
            server_cmd = f"mpirun --allow-run-as-root -np {num_server_tasks} bash -c {shlex.quote(server_cmd)}"
        commands.append(server_cmd)
        executors.append(server_executor)
        het_group_indices.append(het_group)
        het_group += 1
        LOG.info("Server command: %s", server_cmd)

    # then goes the main task(s) unless it's empty
    if cmd:
        if isinstance(cmd, str):
            cmd = [cmd]
        if isinstance(container, str):
            container = [container]
        if isinstance(num_tasks, int):
            num_tasks = [num_tasks]
        if len(cmd) != len(container) or len(cmd) != len(num_tasks):
            raise ValueError("Number of commands, containers and num_tasks must match.")
        for cur_idx, (cur_cmd, cur_container, cur_tasks) in enumerate(zip(cmd, container, num_tasks)):
            if cluster_config["executor"] != "slurm" and cur_tasks > 1:
                cur_cmd = f"mpirun --allow-run-as-root -np {cur_tasks} bash -c {shlex.quote(cur_cmd)}"
            with temporary_env_update(cluster_config, {"NEMO_SKILLS_SANDBOX_PORT": sandbox_port}):
                cur_cmd = install_packages_wrap(cur_cmd, installation_command)
                commands.append(cur_cmd)
                executors.append(
                    get_executor(
                        cluster_config=cluster_config,
                        container=cur_container,
                        num_nodes=num_nodes,
                        tasks_per_node=cur_tasks,
                        gpus_per_node=num_gpus if server_config is None else 0,
                        partition=partition,
                        time_min=time_min,
                        dependencies=dependencies,
                        job_name=task_name,
                        log_dir=log_dir,
                        log_prefix="main" if len(cmd) == 1 else f"main_{cur_idx}",
                        extra_package_dirs=extra_package_dirs,
                        slurm_kwargs=slurm_kwargs,
                        heterogeneous=heterogeneous,
                        het_group=het_group,
                        total_het_groups=total_het_groups,
                        overlap=server_config is not None,
                        with_ray=with_ray,
                    )
                )
                het_group_indices.append(het_group)
        het_group += 1
        LOG.info("Main command(s): %s", ", ".join(cmd))

    # finally a sandbox if needed
    if with_sandbox:
        sandbox_env_updates = {"LISTEN_PORT": sandbox_port}
        current_env_vars = cluster_config.get("env_vars", []).copy()
        for override in current_env_vars:
            if "PYTHONPATH" in override:
                if override.startswith("PYTHONPATH="):
                    override = override[11:]
                sandbox_env_updates["PYTHONPATH"] = override + ":/app"

        with temporary_env_update(cluster_config, sandbox_env_updates):
            commands.append(get_sandbox_command(cluster_config))
            sandbox_executor = get_executor(
                cluster_config=cluster_config,
                container=cluster_config["containers"]["sandbox"],
                num_nodes=executors[0].nodes if cluster_config["executor"] == "slurm" else 1,
                tasks_per_node=1,
                gpus_per_node=0,
                partition=partition,
                time_min=time_min,
                mounts=[],  # we don't want to mount anything
                dependencies=dependencies,
                job_name=task_name,
                log_dir=log_dir,
                log_prefix="sandbox",
                extra_package_dirs=extra_package_dirs,
                slurm_kwargs=slurm_kwargs,
                heterogeneous=heterogeneous,
                het_group=het_group,
                total_het_groups=total_het_groups,
                overlap=server_config is not None,
                with_ray=with_ray,
            )
            executors.append(sandbox_executor)
            het_group_indices.append(het_group)
        het_group += 1
        LOG.info("Sandbox command: %s", commands[-1])

    if cluster_config["executor"] != "none":
        tunnel = get_tunnel(cluster_config)
        if reuse_code:
            reuse_code_exp = reuse_code_exp or REUSE_CODE_EXP.get(tunnel_hash(tunnel))
            if reuse_code_exp is not None:
                if isinstance(reuse_code_exp, str):
                    try:
                        reuse_code_exp = run.Experiment.from_id(reuse_code_exp)
                    except Exception:
                        LOG.debug(f"Failed to create experiment from id {reuse_code_exp}, trying to find it by title")
                        reuse_code_exp = run.Experiment.from_title(reuse_code_exp)

                LOG.info("Trying to reuse code from experiment %s", reuse_code_exp._title)
                reuse_key = get_packaging_job_key(reuse_code_exp._id, "nemo-run")
                if reuse_key in reuse_code_exp.tunnels[tunnel.key].packaging_jobs:
                    reuse_dir = reuse_code_exp.tunnels[tunnel.key].packaging_jobs[reuse_key].dst_path

                    for executor in executors:
                        executor.packager.symlink_from_remote_dir = reuse_dir
                    LOG.info(f"Successfully reused code from {reuse_key}")
                else:
                    LOG.warning("Relevant packaging job not found for experiment %s", reuse_code_exp._title)
        # if current is not reused, we are refreshing the cache as there is a reason to believe it's outdated
        else:
            REUSE_CODE_EXP.pop(tunnel_hash(tunnel), None)

    # no mounting here, so assuming /nemo_run/code can be replaced with the current dir
    if cluster_config["executor"] == "none":
        # replacing /nemo_run/code/nemo_skills with the installed location

        for idx in range(len(commands)):
            commands[idx] = commands[idx].replace(
                '/nemo_run/code/nemo_skills', str(get_registered_external_repo('nemo_skills').path)
            )
            commands[idx] = commands[idx].replace('/nemo_run/code', './')

    if with_ray and cluster_config["executor"] == "slurm":
        metadata = {"use_with_ray_cluster": True}
    else:
        metadata = None

    if not task_dependencies:  # empty list
        task_dependencies = None

    if len(commands) == 1:
        # to keep sbatch script simpler, we don't wrap in a list in this case
        return exp.add(
            run.Script(inline=commands[0], metadata=metadata),
            executor=executors[0],
            name="nemo-run",
            dependencies=task_dependencies,
        )
    else:
        if heterogeneous:
            executors[0].het_group_indices = het_group_indices
        return exp.add(
            [
                run.Script(inline=command, metadata=(metadata if idx == 0 else None))
                for idx, command in enumerate(commands)
            ],
            executor=executors,
            name="nemo-run",
            dependencies=task_dependencies,
        )


def run_exp(exp, cluster_config, sequential=False, dry_run=False):
    """If sequential is not specified, using True locally and False otherwise.

    If it is specified, it will be used as is.
    """
    if dry_run:
        LOG.info("Dry run mode is enabled, not running the experiment.")
        return
    if cluster_config['executor'] != 'slurm':
        exp.run(detach=False, tail_logs=True, sequential=sequential)
    else:
        try:
            exp.run(detach=True, sequential=sequential)
        except RuntimeError as e:
            if 'Your repo has uncommitted changes.' in str(e):
                raise RuntimeError(
                    "You're running ns commands from a git repo - in this case we "
                    "always try to package it and upload to the cluster "
                    "(or store a copy in ~/.nemo_run if running locally).\n"
                    "If you don't need it to be uploaded, cd away from a git repo and rerun the command.\n"
                    "If you do want to upload the code, either commit the changes "
                    "or set NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 "
                    "environment variable to skip the check (but not-committed code will not be packaged)."
                )
            else:
                raise

        # caching the experiment code for reuse
        tunnel = get_tunnel(cluster_config)
        cur_tunnel_hash = tunnel_hash(tunnel)
        if cur_tunnel_hash not in REUSE_CODE_EXP:
            REUSE_CODE_EXP[cur_tunnel_hash] = exp


def get_exp(expname, cluster_config, _reuse_exp=None):
    # Use existing experiment if provided, otherwise create a new one
    if _reuse_exp:
        return contextlib.nullcontext(_reuse_exp)
    # nemo-run redefines the handlers, so removing ours to avoid duplicate logs
    remove_handlers()
    if cluster_config['executor'] == 'slurm':
        return run.Experiment(
            expname,
            skip_status_at_exit=True,
            serialize_metadata_for_scripts=False,
            threadpool_workers=cluster_config.get('num_workers', 4),
        )
    # hiding all nemo-run logs otherwise as they are not useful locally
    if cluster_config['executor'] == 'local':
        return run.Experiment(expname, clean_mode=True)
    return run.Experiment(expname, clean_mode=True, log_level="WARN")

# Launching a Remote Model Server

!!! info

    This pipeline starting script is [nemo_skills/pipeline/start_server.py](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/pipeline/start_server.py)

This pipeline provides a convenient way to host models remotely on a Slurm cluster and access them locally.
It is especially useful for quick debugging or when your local workstation does not have all the required compute
resources available.

To run a vLLM model available at the path `/workspace/models/Qwen3-235B-A22B-Thinking-2507` (as mounted in the cluster configuration),
```bash
ns start_server \
    --cluster=<cluster_name> \
    --server_type=vllm \
    --model=/workspace/models/Qwen3-235B-A22B-Thinking-2507 \
    --server_gpus=8 \
    --server_nodes=2 \
    --log_dir=/workspace/logs/start_server \
    --create_tunnel
```

Now, the model server is available at `localhost:5000` by default. Note that it may take a while before the
model server starts up depending on the size of the loaded model.

!!! tip

    Pressing `ctrl + c` twice will terminate all tunnels and shutdown the launched slurm job as well.

## Sandbox Server

To launch a sandbox server, provide the `--with_sandbox` argument.

When the `--create_tunnel` argument is set, the sandbox server is available at `localhost:6000` by default.

## Remote and Tunnel Ports

To avoid port conflicts on the remote hosts, use `--get_random_port` to randomly assign ports to launched server.

The local port for the model server can be changed using the `--server_tunnel_port` argument. For instance,
setting,
```bash
ns start_server ... --server_tunnel_port=9999
```
will make the model server available at `localhost:9999`.

Similarly, the local port for the sandbox server can be changed using `--sandbox_tunnel_port` argument.

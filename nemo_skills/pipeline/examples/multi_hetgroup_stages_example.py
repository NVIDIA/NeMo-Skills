import argparse

from nemo_skills.pipeline.utils.declarative import (
    Command,
    HardwareConfig,
    HetGroup,
    Pipeline,
)


def main():
    """Example showing:
    - Multi-hetgroup jobs (combining multiple HetGroups into ONE heterogeneous SLURM job)
    - Parallel independent jobs
    - Jobs depending on multiple other jobs
    - All configuration is static (no .named() or .depends_on() chaining)
    - Mock servers and evaluations that complete automatically (for demo purposes)
    """
    parser = argparse.ArgumentParser(description="Multi-stage HetGroup pipeline example")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--cluster", default="local")
    parser.add_argument("--partition", default=None)
    parser.add_argument(
        "--output_dir",
        default="/experiments/declarative/multi_stage_outputs/run_01",
    )

    args = parser.parse_args()

    # ========================================================================
    # STAGE 1: Data Preprocessing (single job)
    # ========================================================================

    preprocess = Command(
        command="""
        echo "=== Preprocessing Data ==="
        mkdir -p /experiments/data
        echo "Simulating data download and preprocessing..."
        sleep 5
        echo "Data ready!"
        """,
        container="nemo-skills",
        gpus=8,
        name="preprocess",
    )

    prep_job = HetGroup(
        commands=[preprocess],
        hardware=HardwareConfig(partition=args.partition),
        name="prep",
    )

    # ========================================================================
    # STAGE 2: Two Server Groups (COMBINED into ONE heterogeneous job)
    # This demonstrates the multi-hetgroup feature!
    # ========================================================================

    # Server 8B with sandbox (het component +0)
    # Mock servers that include health checks and run for limited time
    server_8b = Command(
        command="""
        echo "=== Starting Mock vLLM Server 8B ==="

        # Simulate server startup
        echo "Loading model Qwen/Qwen3-8B..."
        sleep 3

        # Mock health endpoint (simulate server running)
        echo "Mock server started on port 8000"

        # Simulate health check
        echo "Waiting for server to be healthy..."
        sleep 2
        echo "✅ Server 8B is healthy and ready!"

        # Keep server "running" for 20 seconds (simulating processing time)
        echo "Server running and processing requests..."
        sleep 20

        echo "Server 8B shut down complete"
        """,
        container="vllm",
        gpus=8,
        name="server_8b",
    )

    sandbox_8b = Command(
        command="""
        echo "=== Starting Mock Sandbox for 8B ==="
        echo "Sandbox initialized"

        # Keep sandbox "alive" for 20 seconds
        sleep 20

        echo "Sandbox shut down complete"
        """,
        container="sandbox",
        name="sandbox_8b",
    )

    server_8b_group = HetGroup(
        commands=[server_8b, sandbox_8b],
        hardware=HardwareConfig(num_nodes=1, num_gpus=8, partition=args.partition),
        name="server_8b_group",
    )

    # Server 32B with sandbox (het component +1)
    server_32b = Command(
        command="""
        echo "=== Starting Mock vLLM Server 32B ==="

        # Simulate server startup (larger model takes longer)
        echo "Loading model Qwen/Qwen3-32B..."
        sleep 4

        # Mock health endpoint (simulate server running)
        echo "Mock server started on port 8001"

        # Simulate health check
        echo "Waiting for server to be healthy..."
        sleep 2
        echo "✅ Server 32B is healthy and ready!"

        # Keep server "running" for 20 seconds (simulating processing time)
        echo "Server running and processing requests..."
        sleep 20

        echo "Server 32B shut down complete"
        """,
        container="vllm",
        gpus=8,
        name="server_32b",
    )

    sandbox_32b = Command(
        command="""
        echo "=== Starting Mock Sandbox for 32B ==="
        echo "Sandbox initialized"

        # Keep sandbox "alive" for 20 seconds
        sleep 20

        echo "Sandbox shut down complete"
        """,
        container="sandbox",
        name="sandbox_32b",
    )

    server_32b_group = HetGroup(
        commands=[server_32b, sandbox_32b],
        hardware=HardwareConfig(num_nodes=1, num_gpus=8, partition=args.partition),
        name="server_32b_group",
    )

    # ========================================================================
    # STAGE 3: Parallel Evaluation Jobs (each depends on the combined server job)
    # ========================================================================

    # Mock evaluation jobs (servers have already run and shut down by this point)
    eval_8b = Command(
        command="""
        echo "=== Mock Evaluation for 8B Model ==="
        echo "Simulating evaluation workload..."
        echo "Processing test dataset (100 samples)..."
        sleep 5
        echo "Computing metrics..."
        sleep 2
        echo "✅ Evaluation complete!"
        echo "Results: Accuracy=94.2%, F1=0.91"
        """,
        container="nemo-skills",
        gpus=1,
        name="eval_8b",
    )

    eval_8b_job = HetGroup(
        commands=[eval_8b],
        hardware=HardwareConfig(partition=args.partition),
        name="eval_8b",
    )

    eval_32b = Command(
        command="""
        echo "=== Mock Evaluation for 32B Model ==="
        echo "Simulating evaluation workload..."
        echo "Processing test dataset (100 samples)..."
        sleep 5
        echo "Computing metrics..."
        sleep 2
        echo "✅ Evaluation complete!"
        echo "Results: Accuracy=96.8%, F1=0.95"
        """,
        container="nemo-skills",
        gpus=1,
        name="eval_32b",
    )

    eval_32b_job = HetGroup(
        commands=[eval_32b],
        hardware=HardwareConfig(partition=args.partition),
        name="eval_32b",
    )

    # ========================================================================
    # STAGE 4: Final Report (depends on BOTH eval jobs)
    # ========================================================================

    report = Command(
        command="""
        echo "=== Generating Final Report ==="
        echo "Collecting results from both evaluations..."
        sleep 3
        echo "Report complete!"
        """,
        container="nemo-skills",
        gpus=8,
        name="report",
    )

    report_job = HetGroup(
        commands=[report],
        hardware=HardwareConfig(partition=args.partition),
        name="report",
    )

    # ========================================================================
    # Create Pipeline - All Configuration is Static
    # ========================================================================

    pipeline = Pipeline(
        name="multi_stage_demo",
        cluster=args.cluster,
        output_dir=args.output_dir,
        jobs=[
            # Job 1: Preprocessing
            {
                "name": "prep",
                "group": prep_job,
            },
            # Job 2: MULTI-HETGROUP - Combines server_8b_group and server_32b_group
            # into ONE heterogeneous SLURM job with 2 het components
            {
                "name": "servers",
                "groups": [server_8b_group, server_32b_group],  # Multi-hetgroup!
                "dependencies": ["prep"],
            },
            # Job 3 & 4: Parallel evaluations
            {
                "name": "eval_8b",
                "group": eval_8b_job,
                "dependencies": ["servers"],
            },
            {
                "name": "eval_32b",
                "group": eval_32b_job,
                "dependencies": ["servers"],
            },
            # Job 5: Final report (depends on BOTH evals)
            {
                "name": "report",
                "group": report_job,
                "dependencies": ["eval_8b", "eval_32b"],  # Multiple dependencies!
            },
        ],
    )

    print(f"Running pipeline with dry_run={args.dry_run} on cluster='{args.cluster}'")
    print("\n=== Pipeline Structure ===")
    print("Job 1: prep (CPU)")
    print("Job 2: servers (MULTI-HETGROUP with 2 het components) [depends: prep]")
    print("  Component +0: mock_server_8b + mock_sandbox_8b (8 GPUs, shared node)")
    print("    - Mock servers run for ~25s then shutdown")
    print("  Component +1: mock_server_32b + mock_sandbox_32b (8 GPUs, shared node)")
    print("    - Mock servers run for ~26s then shutdown")
    print("Job 3: eval_8b (1 GPU, mock eval) [depends: servers]")
    print("Job 4: eval_32b (1 GPU, mock eval) [depends: servers, runs parallel with Job 3]")
    print("Job 5: report (CPU) [depends: eval_8b AND eval_32b]")
    print("\nExecution Flow:")
    print("  prep → servers (mock with auto-shutdown) → (eval_8b || eval_32b) → report")
    print()

    result = pipeline.run(dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run completed successfully!")
    else:
        print("Pipeline execution completed!")

    return result


if __name__ == "__main__":
    main()

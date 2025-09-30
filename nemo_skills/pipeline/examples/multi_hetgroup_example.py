import argparse

from nemo_skills.pipeline.utils.declarative import (
    HardwareConfig,
    HetGroup,
    Pipeline,
    RunCmd,
    Sandbox,
    Server,
)


def main():
    """Example with multiple HetGroups - each becomes a separate SLURM job."""
    parser = argparse.ArgumentParser(description="Multi-HetGroup pipeline example")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run in dry-run mode (default: False)")
    parser.add_argument("--cluster", default="local", help="Cluster to run on (default: local)")
    parser.add_argument("--partition", default=None, help="Partition to run on (default: None)")
    parser.add_argument(
        "--output_dir",
        default="/experiments/declarative/multi_hetgroup_outputs/run_01",
        help="Output directory to save the results",
    )

    args = parser.parse_args()

    # Create servers and sandbox
    server_8b = Server(model="Qwen/Qwen3-8B", gpus=8)
    sandbox = Sandbox()
    server_32b = Server(model="Qwen/Qwen3-32B", gpus=8, name="vllm_server_32b")

    # Create monitor command with LAZY evaluation using lambda
    # The lambda ensures references are evaluated AFTER Pipeline assigns het_group_indices
    # This way cross-component references work correctly in heterogeneous jobs
    monitor_cmd = RunCmd(
        command=lambda: f"""
        echo '=== Starting Server Monitor ==='
        echo 'Qwen3-8B should be at: {server_8b.url_ref()}'
        echo 'Qwen3-32B should be at: {server_32b.url_ref()}'
        echo 'Sandbox should be at: {sandbox.url_ref()}'
        echo ''

        # Track which servers are ready
        server_8b_ready=false
        server_32b_ready=false
        max_attempts=150
        attempt=0

        while [ "$server_8b_ready" = false ] || [ "$server_32b_ready" = false ]; do
            attempt=$((attempt + 1))

            if [ $attempt -gt $max_attempts ]; then
                echo "ERROR: Timeout waiting for servers after $max_attempts attempts"
                exit 1
            fi

            echo "=== Attempt $attempt/$max_attempts ==="

            # Check Qwen3-8B
            if [ "$server_8b_ready" = false ]; then
                if curl -s "{server_8b.health_ref()}" > /dev/null 2>&1; then
                    echo "✅ Qwen3-8B is READY at {server_8b.url_ref()}"
                    server_8b_ready=true
                else
                    echo "⏳ Qwen3-8B is NOT ready yet at {server_8b.url_ref()}"
                fi
            else
                echo "✅ Qwen3-8B is READY"
            fi

            # Check Qwen3-32B
            if [ "$server_32b_ready" = false ]; then
                if curl -s "{server_32b.health_ref()}" > /dev/null 2>&1; then
                    echo "✅ Qwen3-32B is READY at {server_32b.url_ref()}"
                    server_32b_ready=true
                else
                    echo "⏳ Qwen3-32B is NOT ready yet at {server_32b.url_ref()}"
                fi
            else
                echo "✅ Qwen3-32B is READY"
            fi

            # Check sandbox
            if curl -s "{sandbox.url_ref()}" > /dev/null 2>&1; then
                echo "✅ Sandbox is READY at {sandbox.url_ref()}"
            else
                echo "⏳ Sandbox is NOT ready yet at {sandbox.url_ref()}"
            fi

            echo ""

            if [ "$server_8b_ready" = false ] || [ "$server_32b_ready" = false ]; then
                echo "Waiting 5 seconds before next check..."
                sleep 5
            fi
        done

        echo ""
        echo "🎉 All servers are ready!"
        echo "  - Qwen3-8B:  {server_8b.url_ref()}"
        echo "  - Qwen3-32B: {server_32b.url_ref()}"
        echo "  - Sandbox:   {sandbox.url_ref()}"
        echo ""
        echo "You can now use these services!"

        # Keep the job alive for a bit to demonstrate
        echo "Keeping job alive for 30 seconds..."
        sleep 30
        echo "Monitor job complete!"
        """,
        container="nemo-skills",
        gpus=8,
    )

    # Create HetGroups
    hetgroup_8b = HetGroup(
        [server_8b, sandbox, monitor_cmd],
        hardware=HardwareConfig(
            num_nodes=1,
            num_gpus=8,
            partition=args.partition,
        ),
    ).named("qwen3_8b_group")

    hetgroup_32b = HetGroup(
        [server_32b],
        hardware=HardwareConfig(
            num_nodes=1,
            num_gpus=8,
            partition=args.partition,
        ),
    ).named("qwen3_32b_group")

    # Create pipeline - het_group_indices will be assigned automatically
    # When monitor_cmd.command lambda is evaluated, references will work correctly!
    pipeline = Pipeline(
        name="multi_hetgroup_demo",
        cluster=args.cluster,
        output_dir=args.output_dir,
        groups=[
            hetgroup_8b,  # Component +0: server_8b + sandbox + monitor
            hetgroup_32b,  # Component +1: server_32b
        ],
    )

    print(f"Running pipeline with dry_run={args.dry_run} on cluster='{args.cluster}'")
    print("Expected behavior:")
    print("  - 1 SLURM heterogeneous job with 2 components:")
    print("    Component +0: Qwen3-8B server + sandbox + monitor (8 GPUs, shared node)")
    print("    Component +1: Qwen3-32B server (8 GPUs, separate node)")
    print("  - Each HetGroup becomes one heterogeneous component")
    print("  - Tasks within a HetGroup share the same hardware via --overlap")
    print()

    result = pipeline.run(dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run completed successfully!")
        print("To run for real, use: python multi_hetgroup_example.py --cluster ord")
    else:
        print("Pipeline execution completed!")

    return result


if __name__ == "__main__":
    main()

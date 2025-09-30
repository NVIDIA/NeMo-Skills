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
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Simple command pipeline example")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run in dry-run mode (default: False)")
    parser.add_argument("--cluster", default="local", help="Cluster to run on (default: local)")

    args = parser.parse_args()

    # Create components
    shared_server = Server(model="Qwen/Qwen3-8B", gpus=8)  # 8 GPUs as needed
    shared_sandbox = Sandbox()

    # Note: Using f-string (not lambda) is fine here because this is a single HetGroup
    # All components are in the same group, so url_ref() works correctly
    # For cross-component references (multiple HetGroups), use lambda: f"..." instead
    comand = RunCmd(
        command=f"""
        echo 'Waiting for server to be ready...' &&
        echo "Server should be at {shared_server.url_ref()}" &&
        while ! curl -s "{shared_server.health_ref()}" > /dev/null 2>&1; do
            echo "Server not ready yet, waiting 5 seconds..."
            sleep 5
        done &&
        echo 'Server is ready!'
        """,
        container="nemo-skills",
    )

    # Create pipeline
    pipeline = Pipeline(
        name="simple_command",
        cluster=args.cluster,
        output_dir="/experiments/georgea/declarative/simple_command_outputs/run_01",  # ✨ Single output directory for entire pipeline
        groups=[
            HetGroup(
                [
                    shared_server,
                    shared_sandbox,
                    comand,
                ],
                hardware=HardwareConfig(
                    num_nodes=1,
                    num_gpus=8,  # Total GPUs for the HetGroup job
                    server_gpus=8,  # ✨ Explicitly set server GPUs
                    partition="interactive",
                ),
            ).named("simple_command")
        ],
    )

    print(f"Running pipeline with dry_run={args.dry_run} on cluster='{args.cluster}'")
    result = pipeline.run(dry_run=args.dry_run)

    if args.dry_run:
        print("✅ Dry run completed successfully!")
        print("To run for real, use: python simple_command.py")
        print("For help, use: python simple_command.py --help")
    else:
        print("✅ Pipeline execution completed!")

    return result


if __name__ == "__main__":
    main()

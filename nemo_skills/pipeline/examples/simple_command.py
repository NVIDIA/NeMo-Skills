import argparse

from nemo_skills.pipeline.utils.declarative import (
    Command,
    HardwareConfig,
    HetGroup,
    Pipeline,
)


def main():
    """Simple command example - just running basic commands in containers."""
    parser = argparse.ArgumentParser(description="Simple command pipeline example")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run in dry-run mode (default: False)")
    parser.add_argument("--cluster", default="local", help="Cluster to run on (default: local)")
    parser.add_argument("--partition", default=None, help="Partition to run on (default: None)")
    parser.add_argument(
        "--output_dir",
        default="/experiments/declarative/simple_outputs",
        help="Output directory to save the results",
    )

    args = parser.parse_args()

    # Just a simple command - no HetGroup wrapper needed for single tasks
    hello_cmd = Command(command="echo 'Hello from GPU!' && nvidia-smi", container="nemo-skills", gpus=1, name="hello")

    # Multiple commands can run as separate jobs
    process_cmd = Command(
        command="""
        echo "Processing data..."
        # Add your data processing commands here
        python -c "print('Data processing complete!')"
        """,
        container="nemo-skills",
        gpus=2,
        name="processor",
    )

    # Commands with custom environment variables
    custom_env_cmd = Command(
        command="""
        echo "MY_VAR is set to: $MY_VAR"
        echo "ANOTHER_VAR is set to: $ANOTHER_VAR"
        """,
        container="nemo-skills",
        gpus=0,  # CPU only
        env_vars={
            "MY_VAR": "custom_value",
            "ANOTHER_VAR": "another_value",
        },
        name="custom_env",
    )

    # Create pipeline - each Command becomes a separate SLURM job
    # If you want them to run together, wrap them in a HetGroup
    pipeline = Pipeline(
        name="simple_demo",
        cluster=args.cluster,
        output_dir=args.output_dir,
        groups=[
            HetGroup([hello_cmd], hardware=HardwareConfig(partition=args.partition)),
            HetGroup([process_cmd], hardware=HardwareConfig(partition=args.partition)),
            HetGroup([custom_env_cmd], hardware=HardwareConfig(partition=args.partition)),
        ],
    )

    print(f"Running pipeline with dry_run={args.dry_run} on cluster='{args.cluster}'")
    print("Expected behavior:")
    print("  - 3 separate SLURM jobs, one for each command")
    print("  - hello_cmd: runs on 1 GPU")
    print("  - process_cmd: runs on 2 GPUs")
    print("  - custom_env_cmd: runs on CPU with custom environment variables")
    print()

    result = pipeline.run(dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run completed successfully!")
        print(f"To run for real, use: python simple_command.py --cluster {args.cluster}")
    else:
        print("Pipeline execution completed!")

    return result


if __name__ == "__main__":
    main()

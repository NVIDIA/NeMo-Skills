# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_container_names(jsonl_file):
    """Read all container names from the SWE-bench test.jsonl file."""
    container_names = set()

    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            # Use the same logic as in swebench.py
            container_formatter = data["container_formatter"]
            instance_id = data["instance_id"]

            container_name = container_formatter.format(instance_id=instance_id.replace("__", "_1776_"))
            container_names.add(container_name)

    return sorted(list(container_names))


def convert_to_sif(container_name, output_dir):
    """Convert a Docker container to SIF format using apptainer."""
    # Create a safe filename for the SIF file
    container_name = container_name.removeprefix("docker://")
    safe_name = container_name.replace("/", "_").replace(":", "_")
    sif_path = os.path.join(output_dir, f"{safe_name}.sif")

    # Check if SIF file already exists
    if os.path.exists(sif_path):
        print(f"✓ {container_name} -> {sif_path} (already exists)")
        return True, container_name, sif_path

    try:
        # Use apptainer to build SIF from Docker container
        cmd = ["apptainer", "build", sif_path, f"docker://{container_name}"]

        print(f"Converting {container_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {container_name} -> {sif_path}")
            return True, container_name, sif_path
        else:
            print(f"✗ Failed to convert {container_name}")
            print(f"  Error: {result.stderr.strip()}")
            return False, container_name, None

    except Exception as e:
        print(f"✗ Error converting {container_name}: {e}")
        return False, container_name, None


def make_sif_with_openhands(input_sif_path, output_dir, output_suffix, openhands_repo, openhands_commit):
    """Given an existing SIF environment, make a copy of it with OpenHands preinstalled."""
    # Create output filenames
    input_sif_path = pathlib.Path(input_sif_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_sif_path = output_dir / (input_sif_path.stem + output_suffix + ".sif")
    output_def_path = output_dir / (input_sif_path.stem + output_suffix + ".def")

    # Check if output SIF file already exists
    if output_sif_path.exists():
        print(f"✓ {input_sif_path} -> {output_sif_path} (already exists)")
        return True

    try:
        # Build the definition file for the new container
        def_file_text = f"""\
            Bootstrap: localimage
            From: {input_sif_path}

            %post -c /bin/bash
                cd /root &&
                curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" &&
                bash Miniforge3-$(uname)-$(uname -m).sh -b &&
                eval "$(/root/miniforge3/bin/conda shell.bash hook)" &&
                mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux &&
                mkdir OpenHands &&
                cd OpenHands &&
                git clone {openhands_repo} . &&
                git checkout {openhands_commit} &&
                export INSTALL_DOCKER=0 &&
                make build &&
                poetry run python -m pip install datasets &&
                if ! command -v jq >/dev/null 2>&1
                then
                    curl https://github.com/jqlang/jq/releases/download/jq-1.8.1/jq-linux-amd64 -Lo /bin/jq &&
                    chmod +x /bin/jq
                fi
        """
        output_def_path.write_text(textwrap.dedent(def_file_text))

        # Use apptainer to build SIF from definition file
        cmd = ["apptainer", "build", output_sif_path, output_def_path]

        print(f"Building OpenHands env from {input_sif_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {input_sif_path} -> {output_sif_path}")
            return True
        else:
            print(f"✗ Failed to build OpenHands env from {input_sif_path}")
            print(f"  Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"✗ Error building OpenHands env from {input_sif_path}: {e}")
        return False


def create_sifs(container_name, args):
    success, _, sif_path = convert_to_sif(container_name, args.output_directory)
    if success and args.make_openhands_envs:
        short_sha = args.openhands_commit[:8]

        oh_dir = args.openhands_envs_directory
        if oh_dir is None:
            oh_dir = os.path.join(args.output_directory, f"with-openhands-{short_sha}")

        oh_suffix = args.openhands_envs_suffix
        if oh_suffix is None:
            oh_suffix = f"_with-openhands-{short_sha}"

        return make_sif_with_openhands(sif_path, oh_dir, oh_suffix, args.openhands_repo, args.openhands_commit)
    else:
        return success


def main():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Convert Docker containers from a JSONL test file into Apptainer SIF images."
    )

    parser.add_argument("input_file", help="JSONL file to read container URLs from (SWE-bench format)")
    parser.add_argument("output_directory", help="Directory to save SIF files")
    parser.add_argument(
        "--max_workers", "-j", type=int, default=20, help="Number of parallel conversions (default: 20)"
    )

    parser.add_argument(
        "--make_openhands_envs",
        action=argparse.BooleanOptionalAction,
        help="If set, makes copies of the downloaded environments with OpenHands preinstalled.",
    )
    parser.add_argument(
        "--openhands_envs_directory",
        type=str,
        default=None,
        help="Directory to store the OpenHands environments. "
        "Defaults to <output_directory>/with-openhands-<shortened_openhands_commit_sha>.",
    )
    parser.add_argument(
        "--openhands_envs_suffix",
        type=str,
        default=None,
        help="Suffix for the OpenHands environments that is added to the end of the original environment name. "
        "Defaults to _with-openhands-<shortened_openhands_commit_sha>.",
    )
    parser.add_argument(
        "--openhands_repo",
        type=str,
        default="https://github.com/All-Hands-AI/OpenHands.git",
        help="OpenHands repo URL to download. Defaults to the official OpenHands repo.",
    )
    parser.add_argument(
        "--openhands_commit",
        type=str,
        default="9ee704a25a331d0d2eb9a8e87a4dcff1d948855b",
        help="OpenHands commit hash to use. Defaults to 9ee704a25a331d0d2eb9a8e87a4dcff1d948855b (v0.53.0 release).",
    )

    args = parser.parse_args()

    jsonl_path = args.input_file
    output_dir = args.output_directory
    max_workers = args.max_workers

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read container names from JSONL
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found!")
        sys.exit(1)

    print(f"Reading container names from {jsonl_path}...")
    container_names = read_container_names(jsonl_path)

    print(f"Found {len(container_names)} unique containers to convert.")
    print(f"Output directory: {output_dir}")
    print(f"Using {max_workers} parallel workers.\n")

    # Convert containers in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        futures = [executor.submit(create_sifs, container_name, args) for container_name in container_names]

        # Process completed tasks
        for future in as_completed(futures):
            success = future.result()
            if success:
                successful += 1
            else:
                failed += 1

    print("\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(container_names)}")


if __name__ == "__main__":
    main()

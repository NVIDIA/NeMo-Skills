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

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/He-Ren/OJBench_testdata"


def clone_dataset_repo(url, destination):
    if not shutil.which("git"):
        print("Error: Git executable not found.")
        return

    if os.path.exists(destination):
        print(f"Destination path '{destination}' already exists. Removing it.")
        try:
            shutil.rmtree(destination)
        except OSError as e:
            print(f"Error removing directory {destination}: {e}")
            return

    print(f"Cloning {url} into {destination}...")

    command = ["git", "clone", url, destination]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Clone successful.")

        source_file = os.path.join(destination, "prompts", "full.jsonl")
        target_file = os.path.join(destination, "test.jsonl")
        prompts_dir = os.path.join(destination, "prompts")

        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Expected dataset file not found at {source_file}")

        print(f"Moving {source_file} to {target_file} and replacing 'prompt' key with 'question'")
        try:
            with (
                open(source_file, "r", encoding="utf-8") as infile,
                open(target_file, "w", encoding="utf-8") as outfile,
            ):
                for line in infile:
                    data = json.loads(line)
                    data["question"] = data.pop("prompt")
                    data["subset_for_metrics"] = [data["language"], data["difficulty"]]
                    outfile.write(json.dumps(data) + "\n")

            print(f"Removing directory: {prompts_dir}")
            shutil.rmtree(prompts_dir)
            print("✅ Directory removed successfully.")

        except OSError as e:
            print(f"❌ Error during file/directory operations: {e}")

    else:
        print("❌ Clone failed.")
        print(f"Error Details:\n{result.stderr}")


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    clone_dataset_repo(REPO_URL, data_dir)

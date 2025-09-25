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

import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/He-Ren/OJBench_testdata"


def clone_dataset_repo(url, destination):
    if not shutil.which("git"):
        print("Error: Git executable not found.")
        return

    print(f"Cloning {url} into {destination}...")

    command = ["git", "clone", url, destination]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Clone successful.")
    else:
        print("❌ Clone failed.")
        print(f"Error Details:\n{result.stderr}")


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    clone_dataset_repo(REPO_URL, data_dir)

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
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/He-Ren/OJBench_testdata/raw/main"
PROMPT_URL = f"{BASE_URL}/prompts/full.jsonl"
TESTS_URL_TEMPLATE = f"{BASE_URL}/{{dataset}}/{{subdir}}/{{filename}}"


def clone_repository(repo_url: str, local_path: Path):
    """
    Clones a Git repository to a local path.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        local_path (Path): The local directory to clone the repository into.
    """
    if local_path.exists() and local_path.is_dir():
        print(f"Directory '{local_path}' already exists. Skipping clone.")
        # Optional: You could add logic here to pull the latest changes
        # repo = git.Repo(local_path)
        # repo.remotes.origin.pull()
        return

    print(f"Cloning repository from '{repo_url}' to '{local_path}'...")
    try:
        # The clone_from method handles the git clone command
        git.Repo.clone_from(repo_url, local_path, progress=sys.stdout)
        print("\nRepository cloned successfully!")
    except git.GitCommandError as e:
        print(f"An error occurred during cloning: {e}")
        print("Please ensure you have Git and Git-LFS installed and in your system's PATH.")


def download_file(url: str, dest_path: Path):
    try:
        if dest_path.exists():
            return f"Skipped: {dest_path.name}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        return f"Downloaded: {dest_path.name}"
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return f"Error downloading {url}: {e}"


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    # icpc_dir = Path(data_dir / "ICPC")
    # icpc_dir.mkdir(exist_ok=True)
    # noi_dir = Path(data_dir / "NOI")
    # noi_dir.mkdir(exist_ok=True)

    original_file = data_dir / "full.jsonl"
    download_file(PROMPT_URL, original_file)

    python_data = []
    cpp_data = []
    with open(original_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            new_entry = entry
            if entry["language"] == "python":
                python_data.append(entry)
            elif entry["language"] == "cpp":
                cpp_data.append(entry)
            else:
                raise ValueError(f"Unknown language: {entry['language']}")

            # TODO: the tests are too big to download this way
            # dataset = entry["dataset"]
            # entry_id = entry["id"]
            # if entry["dataset"] == "icpc":
            #     subdir = entry_id
            #     dest_dir = data_dir / "ICPC" / subdir
            # elif dataset == "NOI":
            #     subdir = f"loj-{entry_id}"
            #     dest_dir = data_dir / "NOI" / subdir
            # else:
            #     print(f"Warning: Unknown dataset '{dataset}'")
            #     continue
            #
            # for filename in ["init.yml", "data.zip"]:
            #     url = TESTS_URL_TEMPLATE.format(dataset=dataset.upper(), subdir=subdir, filename=filename)
            #     dest_path = dest_dir / filename
            #     download_file(url, dest_path)

    output_file = str(data_dir / f"test_python.jsonl")
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in python_data:
            fout.write(json.dumps(entry) + "\n")

    output_file = str(data_dir / f"test_cpp.jsonl")
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in cpp_data:
            fout.write(json.dumps(entry) + "\n")

    print('Please download the tests by running git clone "https://huggingface.co/datasets/He-Ren/OJBench_testdata"')

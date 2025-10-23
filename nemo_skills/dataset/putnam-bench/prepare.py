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
import re
import shutil
import urllib.request
from pathlib import Path

import requests

URL_prefix = "https://raw.githubusercontent.com/trishullab/PutnamBench/dc91ed7/lean4/src/"
URL = "https://github.com/trishullab/PutnamBench/tree/dc91ed7/lean4/src"


lean_regex = r"(^\s*theorem\s+([\S]+).+?sorry)"
lean_regex_match = re.compile(lean_regex, re.MULTILINE | re.DOTALL)
informal_prefix_regex = r"/--[\s\S]*?-/"
informal_prefix_match = re.compile(informal_prefix_regex)
header_regex = r"^(?:import|open|def|abbrev|noncomputable)\s+.*(?:\n(?:\s*\|.+|[ \t]+.+))*"
header_regex_match = re.compile(header_regex, re.MULTILINE)


def extract_theorem(filename):
    with open(filename, "r") as f:
        text = f.read()

    # retrieve the theorem name, formal statement, informal prefix, informal statement, and header
    lean_matches = lean_regex_match.findall(text)
    assert len(lean_matches) == 1, "Multiple theorems found in the file"
    informal_prefixes = informal_prefix_match.findall(text)
    assert len(informal_prefixes) == 1, "Multiple informal prefixes found in the file"
    headers = header_regex_match.findall(text)
    header = "\n".join(headers) + "\n\n"
    thm_name = lean_matches[0][1]
    full_thm = lean_matches[0][0]
    informal_prefix = informal_prefixes[0]
    informal_statement = informal_prefix.replace("/--", "").replace("-/", "").strip()

    theorem = {
        "name": thm_name,
        "formal_statement": full_thm,
        "informal_prefix": informal_prefix,
        "problem": informal_statement,
        "header": header,
    }

    return theorem


def get_file_names_from_github(url):
    response = requests.get(url)

    if response.status_code == 200:
        # Extract file names using a regular expression
        # This regex pattern matches the hrefs of the files.
        # TODO: This is a pretty fragile approach, as it depends on GitHub's current HTML structure.
        # find all names with putnam*.lean
        pattern = r'putnam[^"]+\.lean'
        matches = re.findall(pattern, response.text)
        return matches
    else:
        print(f"Failed to access {url}, Status code: {response.status_code}")
        return []


def download_with_retries(url, destination, attempts=3):
    """Download a file with basic retry logic."""
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, destination)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"Retrying download ({attempt}/{attempts}) for {url} due to error: {exc}")


def download_dataset(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # get all file names with putnam*.lean
    file_names = get_file_names_from_github(URL)
    for file_name in file_names:
        # download the file if not exists
        if not os.path.exists(os.path.join(output_path, file_name)):
            download_with_retries(URL_prefix + file_name, os.path.join(output_path, file_name))


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


def delete_file(file_path):
    # delete the folder and all its contents
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def main():
    data_dir = Path(__file__).absolute().parent
    original_folder = str(data_dir / "lean4")
    download_dataset(original_folder)

    # extract data
    theorems = []
    for filename in os.listdir(original_folder):
        if filename.endswith(".lean"):
            theorems.append(extract_theorem(os.path.join(original_folder, filename)))

    save_data(theorems, str(data_dir / "test.jsonl"))
    delete_file(original_folder)


if __name__ == "__main__":
    main()

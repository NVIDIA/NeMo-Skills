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
import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from nemo_skills.dataset.utils import get_lean4_header, add_header_to_jsonl_inplace

datasets = [d.name for d in (Path(__file__).parents[0]).glob("*") if d.is_dir() and d.name != "__pycache__"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare all datasets')
    parser.add_argument('datasets', default=[], nargs="*", help='Can specify a subset here')
    parser.add_argument(
        '--dataset_groups',
        default=[],
        nargs="*",
        choices=["math", "code", "chat", "multichoice"],
        help='Can specify a dataset groups here',
    )
    parser.add_argument(
        '--add_lean4_header',
        action='store_true',
        help='Add Lean4 header to JSONL files during preparation'
    )
    args = parser.parse_args()

    if args.datasets and args.dataset_groups:
        raise ValueError("Cannot specify both datasets and dataset_groups")

    if not args.datasets:
        args.datasets = datasets

    if args.dataset_groups:
        target_datasets = []
        for dataset in args.datasets:
            dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
            if dataset_module.DATASET_GROUP in args.dataset_groups:
                target_datasets.append(dataset)
        args.datasets = target_datasets

    datasets_dir = Path(__file__).absolute().parents[0]
    for dataset in args.datasets:
        print(f"Preparing {dataset}")
        dataset_path = datasets_dir / dataset
        subprocess.run(f"{sys.executable} {dataset_path / 'prepare.py'}", shell=True, check=True)
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        
        if dataset_module.DATASET_GROUP == "math":
            if args.add_lean4_header:
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                header = get_lean4_header()
                for jsonl_file in jsonl_files:
                    print(f"Adding Lean4 header to {jsonl_file}")
                    add_header_to_jsonl_inplace(jsonl_file, header)


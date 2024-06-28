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
import subprocess
import sys
from pathlib import Path

datasets = [d.name for d in (Path(__file__).parents[0]).glob("*") if d.is_dir() and d.name != "__pycache__"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare all datasets')
    parser.add_argument('datasets', default=[], nargs="*", help='Can specify a subset here')
    args = parser.parse_args()

    if not args.datasets:
        args.datasets = datasets

    datasets_folder = Path(__file__).absolute().parents[0]
    for dataset in args.datasets:
        print(f"Preparing {dataset}")
        subprocess.run(f"{sys.executable} {datasets_folder / dataset / 'prepare.py'}", shell=True, check=True)

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

# ruler's data and init files are generated dynamically based on the provided parameters
# will create multiple subfolders corresponding to different evaluation setups

import argparse
import concurrent.futures
import subprocess
import tempfile
from pathlib import Path


def get_ruler_data(tasks, ruler_prepare_args, tmp_data_dir=None):
    # 1. installing necessary packages
    subprocess.run(["pip install wonderwords html2text tenacity"], check=True, shell=True)

    # 2. use provided tmp_data_dir or create a temporary directory
    if tmp_data_dir is not None:
        tmpdirname = tmp_data_dir
        Path(tmpdirname).mkdir(parents=True, exist_ok=True)
        tmpdir_context = None
    else:
        tmpdir_context = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir_context.__enter__()

    try:
        json_dir = Path(tmpdirname) / "RULER" / "scripts" / "data" / "synthetic" / "json"
        required_files = [
            "english_words.json",
            "hotpotqa.json",
            "PaulGrahamEssays.json",
            "squad.json",
        ]
        # Check if all required files exist
        files_exist = all((json_dir / fname).exists() for fname in required_files)
        if not files_exist:
            subprocess.run(
                "git clone --branch igitman/add-write-manifest https://github.com/Kipok/RULER && "
                "cd RULER/scripts/data/synthetic/json && "
                "python download_paulgraham_essay.py && bash download_qa_dataset.sh",
                check=True,
                shell=True,
                cwd=tmpdirname,
            )

        # preparing the datasets based on user options, in parallel
        def prepare_task(task):
            subprocess.run(
                f"python prepare.py --save_dir {Path(__file__).parent}/ruler_data --benchmark synthetic "
                f"    --subset test --task {task} --tokenizer_type hf --model_template_type base --prepare_for_ns "
                f"    --num_samples 500 {ruler_prepare_args}",
                shell=True,
                check=True,
                cwd=Path(tmpdirname) / "RULER" / "scripts" / "data",
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(prepare_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise exception if any subprocess fails

    finally:
        if tmpdir_context is not None:
            tmpdir_context.__exit__(None, None, None)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Prepare RULER dataset.")
    argparser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
            "vt",
            "cwe",
            "fwe",
            "qa_1",
            "qa_2",
        ],
        help="List of tasks to prepare for RULER dataset.",
    )
    argparser.add_argument(
        "--tmp_data_dir",
        type=str,
        default=None,
        help="Directory to store intermediate data. If not provided, a temporary directory will be created.",
    )

    args, unknown = argparser.parse_known_args()
    ruler_prepare_args = " ".join(unknown)
    if not ruler_prepare_args:
        print("ERROR: Can't prepare ruler without arguments provided! Skipping the preparation step.")
        exit(0)
    print(f"Preparing RULER dataset for tasks: {args.tasks} with additional arguments: {ruler_prepare_args}")
    get_ruler_data(args.tasks, ruler_prepare_args, tmp_data_dir=args.tmp_data_dir)
    print("RULER dataset preparation completed.")

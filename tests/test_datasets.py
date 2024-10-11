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

import importlib
import subprocess
from pathlib import Path

# tuple of dataset name, available splits and prepared sft files
DATASETS = [
    ('aime24', ['test']),
    ('amc23', ['test']),
    ('omni-math', ['test']),
    ('algebra222', ['test']),
    ('arena-hard', ['test']),
    ('asdiv', ['test']),
    ('gsm-hard', ['test', 'test_rounded']),
    ('gsm-ic-2step', ['test']),
    ('gsm-ic-mstep', ['test']),
    ('gsm-plus', ['test', 'test_rounded']),
    ('gsm8k', ['train', 'train_full', 'validation', 'test']),
    ('human-eval', ['test']),
    ('ifeval', ['test']),
    ('math', ['train', 'train_full', 'validation', 'test']),
    ('math-odyssey', ['test']),
    ('mawps', ['test']),
    ('mbpp', ['test']),
    ('mmlu', ['test', 'dev', 'val']),
    ('svamp', ['test']),
    ('tabmwp', ['train', 'validation', 'test']),
    ('answer-judge', ['test']),
]


def test_dataset_scripts():
    # test dataset groups
    dataset_groups = ["math", "code", "chat", "multichoice"]
    prepared_datasets = set()
    for group in dataset_groups:
        result = subprocess.run(
            f'python {Path(__file__).absolute().parents[1] / "nemo_skills" / "dataset" / "prepare.py"} --dataset_groups {group}',
            shell=True,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Preparation of {group} dataset group failed"

        group_datasets = set(line.split()[1] for line in result.stdout.split('\n') if line.startswith("Preparing"))
        prepared_datasets.update(group_datasets)

        # Check if at least one dataset from the group was prepared
        assert len(group_datasets) > 0, f"No datasets were prepared for group {group}"

    all_datasets = set(dataset for dataset, _ in DATASETS)
    assert (
        prepared_datasets == all_datasets
    ), f"Not all datasets were covered. Missing: {all_datasets - prepared_datasets}"

    # checking that all expected files are created
    expected_files = []
    for dataset, splits in DATASETS:
        for split in splits:
            expected_files.append(f"{dataset}/{split}.jsonl")

    for file in expected_files:
        assert (Path(__file__).absolute().parents[1] / "nemo_skills" / "dataset" / file).exists()


def test_dataset_init_defaults():
    for dataset, _ in DATASETS:
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        assert hasattr(dataset_module, 'PROMPT_CONFIG'), f"{dataset} is missing PROMPT_CONFIG attribute"
        assert hasattr(dataset_module, 'DATASET_GROUP'), f"{dataset} is missing DATASET_GROUP attribute"
        assert dataset_module.DATASET_GROUP in [
            "math",
            "code",
            "chat",
            "multichoice",
        ], f"{dataset} has invalid DATASET_GROUP"

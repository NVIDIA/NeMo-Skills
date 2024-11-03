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

import hashlib
import subprocess


def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def test_multiple_files():
    output_file = "tests/data/processed_multifile_output.jsonl"
    subprocess.run(
        "python -m nemo_skills.training.prepare_sft_data "
        f"    ++input_files='tests/data/output-rs*.test' "
        f"    ++output_path={output_file} "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template=llama3-instruct "
        f"    ++exclude_optional_keys=false "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.drop_multi_boxed=true "
        f"    ++filters.trim_solutions=true "
        f"    ++filters.drop_incorrect_arithmetic=false "
        f"    ++filters.split_arithmetic=false "
        f"    ++num_output_samples=32 "
        f"    ++downsampling_method=fair "
        f"    ++do_shuffle=false ",
        check=True,
        shell=True,
    )

    expected_md5 = "3971b33e2dd9ed28b2edc323eac19a1f"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_sft_data.py"


def test_exclude_keys():
    output_file = "tests/data/processed_compact_output.jsonl"
    subprocess.run(
        "python -m nemo_skills.training.prepare_sft_data "
        f"    ++input_files='tests/data/output-rs*.test' "
        f"    ++output_path={output_file} "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template=llama3-instruct "
        f"    ++exclude_optional_keys=true "
        f"    ++filters.remove_len_outlier_problems=false "
        f"    ++filters.drop_multi_boxed=true "
        f"    ++filters.trim_solutions=true "
        f"    ++filters.drop_incorrect_arithmetic=false "
        f"    ++filters.split_arithmetic=false "
        f"    ++num_output_samples=32 "
        f"    ++downsampling_method=fair "
        f"    ++do_shuffle=false ",
        check=True,
        shell=True,
    )

    expected_md5 = "7144c098a6e75ffd01c29e714552db24"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_sft_data.py"


def test_code_sft_data():
    output_file = "tests/data/code_processed_output.jsonl"
    subprocess.run(
        "python -m nemo_skills.training.prepare_sft_data "
        f"    --config-name=prepare_code_sft_data "
        f"    ++preprocessed_dataset_files='tests/data/code-output.test' "
        f"    ++output_path={output_file} "
        f"    ++prompt_config=generic/codegen "
        f"    ++prompt_template=llama3-instruct "
        f"    ++exclude_optional_keys=false "
        f"    ++filters.drop_incorrect_code_blocks=false "
        f"    ++generation_suffix='\"<|eot_id|>\"' ",
        check=True,
        shell=True,
    )

    expected_md5 = "a830a174291795cc7db0d1c3ee39de25"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/training/prepare_sft_data.py"


def test_openmathinstruct2():
    output_file = "tests/data/openmathinstruct2-sft.jsonl"

    subprocess.run(
        "python -m nemo_skills.training.prepare_sft_data "
        "++preprocessed_dataset_files='tests/data/openmathinstruct2.test' "
        f"output_path={output_file} "
        "++prompt_template=llama3-instruct "
        "++prompt_config=generic/math "
        "++output_key=generated_solution "
        "++output_path='tests/data/openmathinstruct2-sft.jsonl' "
        "++filters.remove_len_outlier_problems=false "
        "++filters.drop_multi_boxed=false "
        "++filters.trim_prefix=false "
        "++filters.trim_solutions=false "
        "++filters.drop_incorrect_arithmetic=false "
        "++filters.split_arithmetic=false "
        "++generation_suffix='\"<|eot_id|>\"'",
        check=True,
        shell=True,
    )

    expected_md5 = "981e11051436be68cdc45953888a5685"
    output_md5 = compute_md5(output_file)

    assert (
        expected_md5 == output_md5
    ), "MD5 hashes do not match, something is wrong with nemo_skills/finetuning/prepare_sft_data.py"

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

"""
Usage:

python functional_helpers.py <function name> --<ANY ADDITIONAL ARGUMENTS> ...
"""
import glob
import logging
import os

from fire import Fire
from nemo_skills import utils

# Utility function that checks if there are any extra arguments passed to the function
from nemo_skills.utils import check_no_extra_args_fire, setup_logging

from output_processing import post_process_generation

logger = logging.getLogger('nemo_skills')


def rename_files_to_json(data_path: str):
    """
    Utility function to rename files to .json extension

    Args:
        data_path: Path to the data file that has a key 'is_valid_sample' in each sample
    """
    if "*" in data_path:
        # Treat as glob pattern
        all_data_path = sorted(glob.glob(data_path))

        # Filter the .done files if they exist
        all_data_path = [path for path in all_data_path if not path.endswith('.done')]
    else:
        all_data_path = [data_path]

    for data_path in all_data_path:
        # Rename the file to .json extension
        filepath, ext = os.path.splitext(data_path)
        new_filename = f"{filepath}.json"
        os.rename(data_path, new_filename)
        logger.info(f"Renamed the file {data_path} to {new_filename}")


def filter_invalid_samples(
    data_path: str,
    output_dir: str = "./output/",
    output_filename: str = "filtered_data.json",
    num_chunks: int = None,
    chunk_id: int = None,
    **kwargs,
):
    """
    Utility function to filter invalid samples from the given data file

    Args:
        data_path: Path to the data file that has a key 'is_valid_sample' in each sample
        output_dir: Output directory to save the filtered data
        output_filename: Output filename to save the filtered data
        num_chunks: Number of chunks to split the data into
        chunk_id: Chunk ID to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if "*" in data_path:
        # Treat as glob pattern
        all_data_path = sorted(glob.glob(data_path))

        # Filter the .done files if they exist
        all_data_path = [path for path in all_data_path if not path.endswith('.done')]

        out_file, ext = os.path.splitext(output_filename)
        all_output_filenames = [
            f"{out_file}_{os.path.basename(input_path)}" for idx, input_path in enumerate(all_data_path)
        ]

    else:
        # Just get the chunked file name of input, no need to load the data
        num_chunks = utils.maybe_get_env(num_chunks, "NUM_CHUNKS", None, cast=int)
        chunk_id = utils.maybe_get_env(chunk_id, "CHUNK_ID", None, cast=int)
        if num_chunks is not None and chunk_id is not None:
            _, data_path = utils.chunk_data([], data_path, chunk_id, num_chunks)
            _, output_filename = utils.chunk_data([], output_filename, chunk_id, num_chunks)

        all_data_path = [data_path]
        all_output_filenames = [output_filename]

    for data_path, output_filename in zip(all_data_path, all_output_filenames):
        # Load the data
        data = utils.jload(data_path)
        data = [sample for sample in data if sample is not None]
        num_samples = len(data)
        logger.info(f"Loaded {num_samples} samples from {data_path}")

        # Check for invalid samples with no 'is_valid_sample' key
        wrong_ids = []
        for sample_id, sample in enumerate(data):
            if 'is_valid_sample' not in sample:
                wrong_ids.append(sample_id)

        # Log the invalid samples
        if wrong_ids:
            logger.warning(f"Found {len(wrong_ids)} samples with no 'is_valid_sample' key")
            logger.warning(f"Sample IDs: {wrong_ids}")

        # Filter invalid samples
        data[:] = [sample for sample in data if sample.get('is_valid_sample', False)]
        num_dropped = num_samples - len(data)
        logger.info(f"Filtered {num_dropped} invalid samples out of {num_samples} samples")

        # Save the filtered data
        utils.jdump(data, os.path.join(output_dir, output_filename))
        logger.info(f"Saved the filtered data ({len(data)}) to {os.path.join(output_dir, output_filename)}")


def filter_code_samples(
    data_path: str,
    output_dir: str = "./output/",
    output_filename: str = "filtered_data.json",
    keep_explanations: bool = True,
    do_ast_check: bool = True,
    filter_reasoning: bool = True,
    reasoning_start_tag: str = "<think>",
    reasoning_end_tag: str = "</think>",
    num_chunks: int = None,
    chunk_id: int = None,
    **kwargs,
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if "*" in data_path:
        # Treat as glob pattern
        all_data_path = sorted(glob.glob(data_path))

        # Filter the .done files if they exist
        all_data_path = [path for path in all_data_path if not path.endswith('.done')]

        out_file, ext = os.path.splitext(output_filename)
        all_output_filenames = [
            f"{out_file}_{os.path.basename(input_path)}" for idx, input_path in enumerate(all_data_path)
        ]


    else:
        # Just get the chunked file name of input, no need to load the data
        num_chunks = utils.maybe_get_env(num_chunks, "NUM_CHUNKS", None, cast=int)
        chunk_id = utils.maybe_get_env(chunk_id, "CHUNK_ID", None, cast=int)
        if num_chunks is not None and chunk_id is not None:
            _, data_path = utils.chunk_data([], data_path, chunk_id, num_chunks)
            _, output_filename = utils.chunk_data([], output_filename, chunk_id, num_chunks)

        all_data_path = [data_path]
        all_output_filenames = [output_filename]

    for data_path, output_filename in zip(all_data_path, all_output_filenames):
        # Load the data
        data = utils.jload(data_path)
        data = [sample for sample in data if sample is not None]
        num_samples = len(data)
        logger.info(f"Loaded {num_samples} samples")

        # Find position of Final Solution
        for sample_idx, sample in enumerate(data):
            # Inject default value for is_valid_sample
            if 'is_valid_sample' not in sample:
                sample['is_valid_sample'] = True

            output = sample['output']

            # If filter reasoning is enabled, check the output exists only in output part
            if filter_reasoning:
                reasoning_start_idx = output.find(reasoning_start_tag)
                reasoning_end_idx = output.find(reasoning_end_tag)
                if reasoning_start_idx != -1 and reasoning_end_idx != -1:
                    checked_output = output[reasoning_end_idx + len(reasoning_end_tag) :]

                elif reasoning_start_idx < 0 and reasoning_end_idx != -1:
                    checked_output = output[reasoning_end_idx + len(reasoning_end_tag) :]

                    # Inject the think tag at the beggining of the output
                    output = reasoning_start_tag + output

                else:
                    # If both reasoning tags are not found, reject the sample
                    sample['is_valid_sample'] = False
                    continue

            else:
                checked_output = output

            # Check the generated code solutions
            check_output = {
                'text': checked_output,
                'finish_reason': 'stop',
            }
            new_instructions = post_process_generation(
                check_output, keep_explanations=keep_explanations, do_ast_check=do_ast_check
            )

            # If after processing the output, the checks failed, reject the sample
            if new_instructions is None:
                sample['is_valid_sample'] = False
                continue

            sample['output'] = output

        # Save the filtered data
        utils.jdump(data, os.path.join(output_dir, output_filename))
        logger.info(f"Saved the filtered data ({len(data)}) to {os.path.join(output_dir, output_filename)}")


if __name__ == '__main__':
    setup_logging(disable_hydra_logs=False)
    check_no_extra_args_fire()  # Warn extra args for Fire
    Fire()

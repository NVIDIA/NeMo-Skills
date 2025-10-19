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
import logging
import shutil
import sys
from pathlib import Path

LOG = logging.getLogger(__file__)


def download_data(input_path, output_path):
    """
    Download and prepare MultiChallenge dataset.

    Args:
        input_path: Path to the original MultiChallenge benchmark_questions.jsonl file
        output_path: Output directory path
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy the original benchmark questions file
    input_file = Path(input_path) / "benchmark_questions.jsonl"
    output_file = output_path / "test.jsonl"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read and convert the data to NeMo-Skills format
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line)

            # Convert to NeMo-Skills format
            converted_data = {
                "question_id": data["QUESTION_ID"],
                "axis": data["AXIS"],
                "conversation": data["CONVERSATION"],
                "target_question": data["TARGET_QUESTION"],
                "pass_criteria": data["PASS_CRITERIA"],
                "expected_answer": None,  # Multi-challenge uses LLM judge, no ground truth
            }

            f_out.write(json.dumps(converted_data) + '\n')

    LOG.info(f"Data prepared successfully at {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare.py <input_path> <output_path>")
        print("Example: python prepare.py /path/to/multi_challenge/data /path/to/output")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    download_data(input_path, output_path)

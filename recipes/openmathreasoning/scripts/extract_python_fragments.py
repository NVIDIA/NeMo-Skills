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

import argparse
import json
import os
import re
from typing import List


def extract_python_blocks_with_context(document: str, window_size: int = 500) -> List[str]:
    """
    Extract Python code blocks with surrounding context from a document.

    Args:
        document (str): The input document text
        window_size (int): Number of characters to include before and after the code block

    Returns:
        list: List of extracted fragments, each containing one Python block with context
    """
    pattern = r'```python\n(.*?)\n```\n```output\n(.*?)\n```'

    matches = list(re.finditer(pattern, document, re.DOTALL))

    if not matches:
        return []

    fragments = []

    for i, match in enumerate(matches):
        block_start = match.start()
        block_end = match.end()

        # Calculate context window start position
        if i == 0:
            context_start = max(0, block_start - window_size)
        else:
            prev_end = matches[i - 1].end()
            context_start = prev_end if (block_start - prev_end < window_size) else (block_start - window_size)

        # Calculate context window end position
        if i == len(matches) - 1:
            context_end = min(len(document), block_end + window_size)
        else:
            next_start = matches[i + 1].start()
            context_end = next_start if (next_start - block_end < window_size) else (block_end + window_size)

        output = match.group(2)
        if "Traceback (most recent call last)" in output:
            continue

        fragment = document[context_start:context_end]
        fragments.append(fragment)

    return fragments


def process_jsonl_file(input_file: str, output_file: str, window_size: int = 500) -> None:
    """
    Process a JSONL file, extracting Python blocks from each entry's 'generation' field.

    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        window_size (int): Number of characters for context window
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, line in enumerate(f_in):
            try:
                entry = json.loads(line.strip())

                if 'generation' not in entry:
                    print(f"Warning: Line {idx} does not contain a 'generation' field, skipping.")
                    continue

                generation = entry['generation']
                fragments = extract_python_blocks_with_context(generation, window_size)

                for fragment_idx, fragment in enumerate(fragments):
                    output_entry = {
                        'index': idx,  # Original line position
                        'fragment_index': fragment_idx,  # Position within the fragments of this generation
                        'fragment': fragment,
                        'original_generation': generation,  # Store the original generation for reference
                    }
                    # Include all original fields except 'generation'
                    for key, value in entry.items():
                        if key != 'generation':
                            output_entry[key] = value

                    f_out.write(json.dumps(output_entry) + '\n')

            except json.JSONDecodeError:
                print(f"Warning: Line {idx} contains invalid JSON, skipping.")
            except Exception as e:
                print(f"Error processing line {idx}: {str(e)}")

    print(f"Processing complete. Results written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract Python code blocks with context from JSONL file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument(
        '--window_size', type=int, default=1500, help='Size of context window before and after code block'
    )

    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    process_jsonl_file(args.input_file, args.output_file, args.window_size)


if __name__ == "__main__":
    main()

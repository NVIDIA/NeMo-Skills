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
#
"""Helper utilities for SLURM environment parsing."""

import sys


def parse_first_hostname(nodelist: str) -> str:
    """Extract the first hostname from a SLURM nodelist.

    Handles various SLURM nodelist formats:
    - Single node: "batch-block1-0075" → "batch-block1-0075"
    - Node range: "batch-block1-[0075-0078]" → "batch-block1-0075"
    - Node list: "batch-block1-0075,batch-block1-0076" → "batch-block1-0075"

    Args:
        nodelist: SLURM nodelist string from environment variable

    Returns:
        First hostname in the nodelist
    """
    # Handle comma-separated list - take first item
    first = nodelist.split(",")[0]

    # Handle bracket notation like "node[001-004]"
    if "[" in first:
        prefix = first.split("[")[0]
        # Get first value from range (e.g., "001-004" → "001")
        bracket_content = first.split("[")[1].rstrip("]")
        suffix = bracket_content.split("-")[0]
        return prefix + suffix

    # Simple hostname, return as-is
    return first


if __name__ == "__main__":
    # Command-line interface for shell usage
    if len(sys.argv) != 2:
        print("Usage: python -m nemo_skills.pipeline.utils.slurm_helpers <nodelist>", file=sys.stderr)
        sys.exit(1)

    nodelist = sys.argv[1]
    print(parse_first_hostname(nodelist))

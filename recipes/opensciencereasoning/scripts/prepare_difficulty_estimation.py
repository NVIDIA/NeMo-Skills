#!/usr/bin/env python3
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

"""Prepare inputs for the difficulty estimation stage."""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path


LOG = logging.getLogger(__name__)

REQUIRED_KEYS = ("problem", "expected_answer", "id")


def process_file(input_path: Path, output_path: Path, *, keep_metadata: bool) -> None:
    """
    Prepare inputs for the difficulty estimation stage by keeping core fields.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            sample = {key: value for key, value in json.loads(line).items() if key in REQUIRED_KEYS}
            fout.write(json.dumps(sample) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare difficulty estimation inputs by keeping core fields.")
    parser.add_argument("--input_file", required=True, type=Path, help="Path to the JSONL input file")
    parser.add_argument("--output_file", required=True, type=Path, help="Path to write the prepared JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    LOG.info("Reading input from %s", args.input_file)
    process_file(args.input_file, args.output_file, keep_metadata=args.keep_metadata)


if __name__ == "__main__":
    main()

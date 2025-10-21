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

# TODO: make actual prepare script

import argparse
import json
from pathlib import Path

NV_INTERNAL_TASKS = [
    "aimo-airline-departures",
    "blind-maze-explorer-5x5",
    "blind-maze-explorer-algorithm",
    "blind-maze-explorer-algorithm.easy",
    "blind-maze-explorer-algorithm.hard",
    "build-initramfs-qemu",
    "build-tcc-qemu",
    "cartpole-rl-training",
    "chem-property-targeting",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "conda-env-conflict-resolution",
    "count-dataset-tokens",
    "crack-7z-hash",
    "crack-7z-hash.easy",
    "crack-7z-hash.hard",
    "cross-entropy-method",
    "csv-to-parquet",
    "decommissioning-service-with-sensitive-data",
    "eval-mteb",
    "eval-mteb.hard",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "fix-permissions",
    "form-filling",
    "get-bitcoin-nodes",
    "gpt2-codegolf",
    "grid-pattern-transform",
    "hello-world",
    "heterogeneous-dates",
    "hf-model-inference",
    "incompatible-python-fasttext",
    "incompatible-python-fasttext.base_with_hint",
    "intrusion-detection",
    "lean4-proof",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "model-extraction-relu-logits",
    "modernize-fortran-build",
    "movie-helper",
    "new-encrypt-command",
    "oom",
    "openssl-selfsigned-cert",
    "organization-json-generator",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "play-zork",
    "play-zork-easy",
    "polyglot-c-py",
    "polyglot-rust-c",
    "prove-plus-comm",
    "pytorch-model-cli",
    "pytorch-model-cli.easy",
    "pytorch-model-cli.hard",
    "raman-fitting",
    "raman-fitting.easy",
    "recover-obfuscated-files",
    "reverse-engineering",
    "run-pdp11-code",
    "sanitize-git-repo",
    "sanitize-git-repo.hard",
    "solana-data",
    "sqlite-db-truncate",
    "stable-parallel-kmeans",
    "super-benchmark-upet",
    "swe-bench-astropy-1",
    "swe-bench-astropy-2",
    "vim-terminal-task",
    "word2vec-from-scratch",
    "write-compressor",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    args = parser.parse_args()

    output_file = Path(__file__).parent / f"{args.setup}.jsonl"
    with open(output_file, "w") as fout:
        for task in NV_INTERNAL_TASKS:
            print(json.dumps({"task_id": task}), file=fout)

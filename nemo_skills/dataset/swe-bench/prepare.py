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

from pathlib import Path

import datasets

if __name__ == "__main__":
    # TODO: support these with options
    dataset = "princeton-nlp/SWE-bench_Verified"
    split = "test"
    # container_formatter = (
    #     "/lustre/fsw/portfolios/llmservice/users/snarenthiran/swe-bench/containers/sweb.eval.x86_64.{instance_id}.sqsh"
    # )
    container_formatter = "swebench/sweb.eval.x86_64.{instance_id}"
    dataset = datasets.load_dataset(path=dataset, split=split)
    output_file = Path(__file__).parent / "test.jsonl"
    dataset = dataset.map(lambda example: {**example, "container_formatter": container_formatter})
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset.to_json(output_file, orient="records", lines=True)

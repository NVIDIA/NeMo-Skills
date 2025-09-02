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
import io
import json
import os

TOOLCALLING_METRIC_PATHS = {
    "overall_accuracy": ["overall_accuracy", "accuracy"],
    "overall_non_live": ["non_live_single_turn", "overall_non_live", "accuracy"],
    "non_live_ast": ["non_live_single_turn", "non_live_ast", "accuracy"],
    "irrelevance": ["non_live_single_turn", "irrelevance", "accuracy"],
    "overall_live": ["live_single_turn", "overall_live", "accuracy"],
    "live_ast": ["live_single_turn", "live_ast", "accuracy"],
    "live_irrelevance": ["live_single_turn", "live_irrelevance", "accuracy"],
    "live_relevance": ["live_single_turn", "live_relevance", "accuracy"],
    "overall_multi_turn": ["multi_turn", "overall_multi_turn", "accuracy"],
}

TOOLCALLING_METRIC_RANGES = {
    "overall_accuracy": (61.0, 67.0),
    "overall_non_live": (84.0, 90.0),
    "non_live_ast": (85.0, 92.0),
    "irrelevance": (79.0, 86.0),
    "overall_live": (76.0, 83.0),
    "live_ast": (79.0, 86.0),
    "live_irrelevance": (73.0, 80.0),
    "live_relevance": (85.0, 92.0),
    "overall_multi_turn": (20.0, 30.0),
}


def load_json(path: str):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: dict, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def detect_mode(name: str):
    if "reasoning_on" in name:
        return "reasoning_on"
    if "reasoning_off" in name:
        return "reasoning_off"
    return None


def check_toolcalling(bucket: str, mode: str):
    f = os.path.join(bucket, "eval-results", "bfcl_v3", "metrics.json")
    data = load_json(f)
    for cat, path in TOOLCALLING_METRIC_PATHS.items():
        val = float(get_nested(data, path))
        lo, hi = TOOLCALLING_METRIC_RANGES[mode][cat]
        assert lo <= val <= hi, f"TOOL {cat}={val} out of range [{lo},{hi}]"


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Assert-only verifier: reasoning + tool-calling + ruler")
    ap.add_argument("--workspace", required=True, help="Workspace root containing eval buckets")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.workspace))
    for bucket in sorted(os.listdir(root)):
        bpath = os.path.join(root, bucket)
        if not os.path.isdir(os.path.join(bpath, "eval-results")):
            continue
        mode = detect_mode(bucket)
        if not mode:
            continue
        if "tool_calling" in bucket:
            check_toolcalling(bpath, mode)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

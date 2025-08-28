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

REASONING_TASKS = [
    "math-500",
    "aime24",
    "aime25",
    "gpqa",
    "mmlu-pro",
    "livecodebench",
    "scicode",
    "hle",
]

REASONING_BENCHMARKS_SCIENCE_HLE = {"scicode", "hle"}

REASONING_REQUIRED_FIELDS = {
    "math-500": ["symbolic_correct"],
    "aime24": ["symbolic_correct"],
    "aime25": ["symbolic_correct"],
    "gpqa": ["symbolic_correct"],
    "mmlu-pro": ["symbolic_correct"],
    "livecodebench": ["accuracy"],
    "scicode": ["problem_accuracy", "subtask_accuracy"],
    "hle": ["judge_correct", "symbolic_correct"],
}

REASONING_METRIC_RANGES = {
    "reasoning_on": {
        "math-500": (90.0, 100.0),
        "aime24": (84.0, 94.0),
        "aime25": (78.0, 88.0),
        "gpqa": (70.0, 80.0),
        "mmlu-pro": (75.0, 85.0),
        "livecodebench": (65.0, 75.0),
        "scicode": {
            "problem_accuracy": (2.0, 5.0),
            "subtask_accuracy": (25.0, 35.0),
        },
        "hle": {
            "judge_correct": (4.0, 12.0),
            "symbolic_correct": (1.0, 10.0),
        },
    },
    "reasoning_off": {
        "math-500": (70.0, 80.0),
        "aime24": (11.0, 21.0),
        "aime25": (3.0, 9.0),
        "gpqa": (46.0, 56.0),
        "mmlu-pro": (65.0, 75.0),
        "livecodebench": (25.0, 35.0),
        "scicode": {
            "problem_accuracy": (0.0, 3.0),
            "subtask_accuracy": (15.0, 25.0),
        },
        "hle": {
            "judge_correct": (2.5, 5.0),
            "symbolic_correct": (0.5, 2.5),
        },
    },
}

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
    "reasoning_on": {
        "overall_accuracy": (60.0, 80.0),
        "overall_non_live": (80.0, 95.0),
        "non_live_ast": (75.0, 95.0),
        "irrelevance": (75.0, 95.0),
        "overall_live": (75.0, 90.0),
        "live_ast": (75.0, 90.0),
        "live_irrelevance": (75.0, 90.0),
        "live_relevance": (65.0, 80.0),
        "overall_multi_turn": (35.0, 50.0),
    },
    "reasoning_off": {
        "overall_accuracy": (60.0, 80.0),
        "overall_non_live": (80.0, 95.0),
        "non_live_ast": (75.0, 95.0),
        "irrelevance": (75.0, 95.0),
        "overall_live": (75.0, 90.0),
        "live_ast": (70.0, 90.0),
        "live_irrelevance": (77.0, 93.0),
        "live_relevance": (47.0, 62.0),
        "overall_multi_turn": (28.0, 42.0),
    },
}

RULER_TASKS = [
    "ruler.nemotron_super_128k",
    "ruler.nemotron_super_128k.niah_single_1",
    "ruler.nemotron_super_128k.niah_single_2",
    "ruler.nemotron_super_128k.niah_single_3",
    "ruler.nemotron_super_128k.niah_multikey_1",
    "ruler.nemotron_super_128k.niah_multikey_2",
    "ruler.nemotron_super_128k.niah_multikey_3",
    "ruler.nemotron_super_128k.niah_multivalue",
    "ruler.nemotron_super_128k.niah_multiquery",
    "ruler.nemotron_super_128k.vt",
    "ruler.nemotron_super_128k.cwe",
    "ruler.nemotron_super_128k.fwe",
    "ruler.nemotron_super_128k.qa_1",
    "ruler.nemotron_super_128k.qa_2",
]

RULER_METRIC_RANGES = {
    "reasoning_on": {
        "ruler.nemotron_super_128k": (55.0, 75.0),
        "ruler.nemotron_super_128k.niah_single_1": (90.0, 100.0),
        "ruler.nemotron_super_128k.niah_single_2": (90.0, 100.0),
        "ruler.nemotron_super_128k.niah_single_3": (90.0, 100.0),
        "ruler.nemotron_super_128k.niah_multikey_1": (65.0, 80.0),
        "ruler.nemotron_super_128k.niah_multikey_2": (50.0, 65.0),
        "ruler.nemotron_super_128k.niah_multikey_3": (15.0, 25.0),
        "ruler.nemotron_super_128k.niah_multivalue": (85.0, 100.0),
        "ruler.nemotron_super_128k.niah_multiquery": (85.0, 95.0),
        "ruler.nemotron_super_128k.vt": (50.0, 65.0),
        "ruler.nemotron_super_128k.cwe": (0.0, 2.0),
        "ruler.nemotron_super_128k.fwe": (80.0, 95.0),
        "ruler.nemotron_super_128k.qa_1": (40.0, 50.0),
        "ruler.nemotron_super_128k.qa_2": (35.0, 45.0),
    },
    "reasoning_off": {
        "ruler.nemotron_super_128k": (55.0, 75.0),
        "ruler.nemotron_super_128k.niah_single_1": (95.0, 100.0),
        "ruler.nemotron_super_128k.niah_single_2": (90.0, 100.0),
        "ruler.nemotron_super_128k.niah_single_3": (90.0, 100.0),
        "ruler.nemotron_super_128k.niah_multikey_1": (60.0, 75.0),
        "ruler.nemotron_super_128k.niah_multikey_2": (45.0, 55.0),
        "ruler.nemotron_super_128k.niah_multikey_3": (15.0, 25.0),
        "ruler.nemotron_super_128k.niah_multivalue": (80.0, 90.0),
        "ruler.nemotron_super_128k.niah_multiquery": (80.0, 90.0),
        "ruler.nemotron_super_128k.vt": (75.0, 85.0),
        "ruler.nemotron_super_128k.cwe": (0.0, 2.0),
        "ruler.nemotron_super_128k.fwe": (80.0, 95.0),
        "ruler.nemotron_super_128k.qa_1": (42.0, 55.0),
        "ruler.nemotron_super_128k.qa_2": (37.0, 47.0),
    },
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


# ---------------- Assert-only checks ----------------
def check_reasoning(bucket: str, mode: str):
    for bench in REASONING_TASKS:
        f = os.path.join(bucket, "eval-results", bench, "metrics.json")
        data = load_json(f)
        if bench in {"math-500", "aime24", "aime25", "gpqa", "livecodebench", "scicode"}:
            result_block = get_nested(data[bench], ["pass@1[avg-of-4]"])
        elif bench in {"mmlu-pro", "hle"}:
            result_block = get_nested(data[bench], ["pass@1"])
        else:
            raise AssertionError(f"Unexpected benchmark: {bench}")

        if bench in REASONING_BENCHMARKS_SCIENCE_HLE:
            for field in REASONING_REQUIRED_FIELDS[bench]:
                val = float(result_block[field])
                lo, hi = REASONING_METRIC_RANGES[mode][bench][field]
                assert lo <= val <= hi, f"{bench}:{field}={val} out of range [{lo},{hi}]"
        else:
            field = REASONING_REQUIRED_FIELDS[bench][0]
            val = float(result_block[field])
            lo, hi = REASONING_METRIC_RANGES[mode][bench]
            assert lo <= val <= hi, f"{bench}:{field}={val} out of range [{lo},{hi}]"


def check_toolcalling(bucket: str, mode: str):
    f = os.path.join(bucket, "eval-results", "bfcl_v3", "metrics.json")
    data = load_json(f)
    for cat, path in TOOLCALLING_METRIC_PATHS.items():
        val = float(get_nested(data, path))
        lo, hi = TOOLCALLING_METRIC_RANGES[mode][cat]
        assert lo <= val <= hi, f"TOOL {cat}={val} out of range [{lo},{hi}]"


def check_ruler(bucket: str, mode: str):
    f = os.path.join(bucket, "eval-results", "ruler.nemotron_super_128k", "metrics.json")
    data = load_json(f)
    for task in RULER_TASKS:
        val = float(data[task]["pass@1"]["accuracy"])
        lo, hi = RULER_METRIC_RANGES[mode][task]
        assert lo <= val <= hi, f"RULER {task}={val} out of range [{lo},{hi}]"


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
        elif "ruler" in bucket:
            check_ruler(bpath, mode)
        else:
            check_reasoning(bpath, mode)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

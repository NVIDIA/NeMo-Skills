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
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# Required accuracy fields for pass@1[avg-of-16]; scicode/hle require different fields.
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

# Expected metric ranges for reasoning tasks
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

# --------------------------- Tool-calling (bfcl_v3) ---------------------------
# Nested JSON paths for categories → list of keys to descend, then read "accuracy".
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

# Expected ranges for tool-calling tasks
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

# --------------------------- RULER (ruler.nemotron_super_128k) ---------------------------
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

# Expected ranges for RULER tasks
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


# --------------------------- Helpers ---------------------------
def load_json(path):
    """Load JSON as UTF-8."""
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_numeric(value):
    """Return True if value is numeric."""
    try:
        float(value)
        return True
    except Exception:
        return False


def format_percentage(value):
    """Format as percentage string."""
    try:
        return "%.2f%%" % float(value)
    except Exception:
        return str(value)


def detect_reasoning_mode(dir_name):
    """Detect reasoning mode from directory name."""
    if "reasoning_on" in dir_name:
        return "reasoning_on"
    if "reasoning_off" in dir_name:
        return "reasoning_off"
    return None


def get_nested_value(dictionary, path):
    """Return nested value from dict by a list of keys; None if missing."""
    cur = dictionary
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


# ---------------- Math Science General Reasoning verify ----------------
def verify_reasoning_metrics(bucket_path, mode):
    """Verify core reasoning benchmarks (pass@1[avg-of-16])."""
    success = True
    for benchmark in REASONING_TASKS:
        metrics_path = os.path.join(bucket_path, "eval-results", benchmark, "metrics.json")
        if not os.path.isfile(metrics_path):
            LOG.error("[REASONING:%-12s] metrics.json missing", benchmark)
            success = False
            continue
        try:
            data = load_json(metrics_path)
        except Exception as e:
            LOG.error("[REASONING:%-12s] unreadable: %s", benchmark, e)
            success = False
            continue

        pass1_block = data.get(benchmark, {}).get("pass@1[avg-of-16]")
        if not isinstance(pass1_block, dict):
            LOG.error("[REASONING:%-12s] missing pass@1[avg-of-16]", benchmark)
            success = False
            continue

        required_fields = REASONING_REQUIRED_FIELDS[benchmark]
        expected_range = REASONING_METRIC_RANGES[mode][benchmark]

        if benchmark in REASONING_BENCHMARKS_SCIENCE_HLE:
            # ALL fields required and each must be in its own expected sub-range
            for field in required_fields:
                if field not in pass1_block or not is_numeric(pass1_block[field]):
                    LOG.error("[REASONING:%-12s] missing/non-numeric %s", benchmark, field)
                    success = False
                    continue
                lower_bound, upper_bound = expected_range[field]
                value = float(pass1_block[field])
                if not (lower_bound <= value <= upper_bound):
                    LOG.error(
                        "[REASONING:%-12s] %s=%s out of range [%.2f%%, %.2f%%]",
                        benchmark,
                        field,
                        format_percentage(value),
                        lower_bound,
                        upper_bound,
                    )
                    success = False
        else:
            # Single required field with a tuple range
            field = required_fields[0]
            if field not in pass1_block or not is_numeric(pass1_block[field]):
                LOG.error("[REASONING:%-12s] missing/non-numeric %s", benchmark, field)
                success = False
                continue
            lower_bound, upper_bound = expected_range
            value = float(pass1_block[field])
            if not (lower_bound <= value <= upper_bound):
                LOG.error(
                    "[REASONING:%-12s] %s=%s out of range [%.2f%%, %.2f%%]",
                    benchmark,
                    field,
                    format_percentage(value),
                    lower_bound,
                    upper_bound,
                )
                success = False
    return success


# ---------------- Tool-calling verify ----------------
def verify_toolcalling_metrics(bucket_path, mode):
    """Verify tool-calling (bfcl_v3) metrics by nested accuracy paths."""
    success = True
    metrics_path = os.path.join(bucket_path, "eval-results", "bfcl_v3", "metrics.json")
    if not os.path.isfile(metrics_path):
        LOG.error("[TOOL] metrics.json missing: %s", metrics_path)
        return False
    try:
        data = load_json(metrics_path)
    except Exception as e:
        LOG.error("[TOOL] unreadable: %s", e)
        return False

    expected_ranges = TOOLCALLING_METRIC_RANGES[mode]
    for category, path in sorted(TOOLCALLING_METRIC_PATHS.items()):
        value = get_nested_value(data, path)
        if not is_numeric(value):
            LOG.error("[TOOL:%-20s] missing/non-numeric accuracy at %s", category, "/".join(path))
            success = False
            continue
        lower_bound, upper_bound = expected_ranges[category]
        value = float(value)
        if not (lower_bound <= value <= upper_bound):
            LOG.error(
                "[TOOL:%-20s] accuracy=%s out of range [%.2f%%, %.2f%%]",
                category,
                format_percentage(value),
                lower_bound,
                upper_bound,
            )
            success = False
    return success


# ---------------- RULER verify ----------------
def verify_ruler_metrics(bucket_path, mode):
    """Verify RULER tasks; read each task at data[task]['pass@1']['accuracy']."""
    success = True
    metrics_path = os.path.join(bucket_path, "eval-results", "ruler.nemotron_super_128k", "metrics.json")
    if not os.path.isfile(metrics_path):
        LOG.error("[RULER] metrics.json missing: %s", metrics_path)
        return False
    try:
        data = load_json(metrics_path)
    except Exception as e:
        LOG.error("[RULER] unreadable: %s", e)
        return False

    expected_ranges = RULER_METRIC_RANGES[mode]
    for task in RULER_TASKS:
        node = data.get(task, {})
        if not (isinstance(node, dict) and "pass@1" in node and isinstance(node["pass@1"], dict)):
            LOG.error("[RULER:%s] missing pass@1", task)
            success = False
            continue
        value = node["pass@1"].get("accuracy")
        if not is_numeric(value):
            LOG.error("[RULER:%s] missing/non-numeric Accuracy", task)
            success = False
            continue
        lower_bound, upper_bound = expected_ranges[task]
        value = float(value)
        if not (lower_bound <= value <= upper_bound):
            LOG.error(
                "[RULER:%s] Accuracy=%s out of range [%.2f, %.2f]",
                task,
                format_percentage(value),
                lower_bound,
                upper_bound,
            )
            success = False
    return success


# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Hard verify: reasoning + tool-calling + ruler (errors only)")
    parser.add_argument("--workspace", required=True, help="Workspace root containing eval buckets")
    args = parser.parse_args()

    workspace_root = os.path.abspath(os.path.expanduser(args.workspace))
    if not os.path.isdir(workspace_root):
        LOG.error("Workspace not found: %s", workspace_root)
        sys.exit(2)

    LOG.setLevel(logging.ERROR)

    verification_passed = True
    for bucket_name in sorted(os.listdir(workspace_root)):
        bucket_path = os.path.join(workspace_root, bucket_name)
        if not (os.path.isdir(bucket_path) and os.path.isdir(os.path.join(bucket_path, "eval-results"))):
            continue
        mode = detect_reasoning_mode(bucket_name)
        if not mode:
            continue

        if "tool_calling" in bucket_name:
            if not verify_toolcalling_metrics(bucket_path, mode):
                verification_passed = False
        elif "ruler" in bucket_name:
            if not verify_ruler_metrics(bucket_path, mode):
                verification_passed = False
        else:
            if not verify_reasoning_metrics(bucket_path, mode):
                verification_passed = False

    if verification_passed:
        LOG.info("OVERALL_OK: True")
        sys.exit(0)
    else:
        LOG.error("OVERALL_OK: False")
        sys.exit(1)


if __name__ == "__main__":
    main()

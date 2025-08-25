import argparse
import io
import json
import os
import sys

# ---------------- Core (reasoning_on/off) ----------------
REASONING_BENCH = ["math-500", "aime24", "aime25", "gpqa", "mmlu-pro", "livecodebench", "scicode", "hle"]

REASONING_ACC_FIELD = {
    "math-500": ["symbolic_correct"],
    "aime24": ["symbolic_correct"],
    "aime25": ["symbolic_correct"],
    "gpqa": ["symbolic_correct"],
    "mmlu-pro": ["symbolic_correct"],
    "livecodebench": ["accuracy"],
    "scicode": ["problem_accuracy", "subtask_accuracy"],  # both required
    "hle": ["judge_correct", "symbolic_correct"],  # both required
}
REASONING_REQUIRE_ALL = set(["scicode", "hle"])

# The metrics ranges for reasoning tasks
REASONING_RANGE = {
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
        "aime24": (11, 21.0),
        "aime25": (3, 9),
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

# ---------------- Tool-calling (bfcl_v3) ----------------
# Nested JSON paths for categories → list of keys to descend, then read 'accuracy'
BFCL_PATHS = {
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

# The metrics ranges for tool calling tasks
RANGE_TOOL = {
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

# ---------------- RULER (ruler.nemotron_super_128k) ----------------
# Exact task keys (with "super" in name). Each value is taken from ["pass@1"]["accuracy"] (numeric)
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

# The metrics ranges for ruler tasks
RANGE_RULER = {
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


# ---------------- Helper Functions----------------
def load_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_num(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def fmt_pct(x):
    try:
        return "%.2f%%" % float(x)
    except Exception:
        return str(x)


def detect_mode(name):
    if "reasoning_on" in name:
        return "reasoning_on"
    if "reasoning_off" in name:
        return "reasoning_off"
    return None


# ---------------- Math Science General Reasoning verify ----------------
def verify_reasoning_bucket(bucket_dir, mode):
    ok = True
    for b in REASONING_BENCH:
        mpath = os.path.join(bucket_dir, "eval-results", b, "metrics.json")
        if not os.path.isfile(mpath):
            print("[REASONING:%-12s] ❌ metrics.json missing" % b)
            ok = False
            continue
        try:
            data = load_json(mpath)
        except Exception as e:
            print("[REASONING:%-12s] ❌ unreadable: %s" % (b, e))
            ok = False
            continue

        blk = data.get(b, {}).get("pass@1[avg-of-16]")
        if not isinstance(blk, dict):
            print("[REASONING:%-12s] ❌ missing pass@1[avg-of-16]" % b)
            ok = False
            continue

        fields = REASONING_ACC_FIELD[b]
        spec = REASONING_RANGE[mode][b]

        if b in REASONING_REQUIRE_ALL:
            for f in fields:
                if f not in blk or not is_num(blk[f]):
                    print("[REASONING:%-12s] ❌ missing/non-numeric %s" % (b, f))
                    ok = False
                    continue
                lo, hi = spec[f]
                val = float(blk[f])
                if not (lo <= val <= hi):
                    print("[REASONING:%-12s] ❌ %s=%s out of range [%.2f%%, %.2f%%]" % (b, f, fmt_pct(val), lo, hi))
                    ok = False
        else:
            f = fields[0]
            if f not in blk or not is_num(blk[f]):
                print("[REASONING:%-12s] ❌ missing/non-numeric %s" % (b, f))
                ok = False
                continue
            lo, hi = spec
            val = float(blk[f])
            if not (lo <= val <= hi):
                print("[REASONING:%-12s] ❌ %s=%s out of range [%.2f%%, %.2f%%]" % (b, f, fmt_pct(val), lo, hi))
                ok = False
    return ok


# ---------------- Tool-calling verify ----------------
def _deep_get(d, path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def verify_tool_bucket(bucket_dir, mode):
    ok = True
    mpath = os.path.join(bucket_dir, "eval-results", "bfcl_v3", "metrics.json")
    if not os.path.isfile(mpath):
        print("[TOOL] ❌ metrics.json missing: %s" % mpath)
        return False
    try:
        data = load_json(mpath)
    except Exception as e:
        print("[TOOL] ❌ unreadable: %s" % e)
        return False

    spec = RANGE_TOOL[mode]
    for cat, path in BFCL_PATHS.items():
        val = _deep_get(data, path)
        if not is_num(val):
            print("[TOOL:%-20s] ❌ missing/non-numeric accuracy at %s" % (cat, "/".join(path)))
            ok = False
            continue
        lo, hi = spec[cat]
        val = float(val)
        if not (lo <= val <= hi):
            print("[TOOL:%-20s] ❌ accuracy=%s out of range [%.2f%%, %.2f%%]" % (cat, fmt_pct(val), lo, hi))
            ok = False
    return ok


# ---------------- RULER verify ----------------
def verify_ruler_bucket(bucket_dir, mode):
    ok = True
    mpath = os.path.join(bucket_dir, "eval-results", "ruler.nemotron_super_128k", "metrics.json")
    if not os.path.isfile(mpath):
        print("[RULER] ❌ metrics.json missing: %s" % mpath)
        return False
    try:
        data = load_json(mpath)
    except Exception as e:
        print("[RULER] ❌ unreadable: %s" % e)
        return False

    spec = RANGE_RULER[mode]
    # Each task value at data[task]["pass@1"]["accuracy"]
    for task in RULER_TASKS:
        node = data.get(task, {})
        if not (isinstance(node, dict) and "pass@1" in node and isinstance(node["pass@1"], dict)):
            print("[RULER:%s] ❌ missing pass@1" % task)
            ok = False
            continue
        val = node["pass@1"].get("accuracy")
        if not is_num(val):
            print("[RULER:%s] ❌ missing/non-numeric Accuracy" % task)
            ok = False
            continue
        lo, hi = spec[task]
        val = float(val)
        if not (lo <= val <= hi):
            print("[RULER:%s] ❌ Accuracy=%.2f out of range [%.2f, %.2f]" % (task, val, lo, hi))
            ok = False
    return ok


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Hard verify: reasoning + tool-calling + ruler (errors only)")
    ap.add_argument("--workspace", required=True, help="Workspace root")
    args = ap.parse_args()

    ws = os.path.abspath(os.path.expanduser(args.workspace))
    if not os.path.isdir(ws):
        print("Workspace not found: %s" % ws)
        sys.exit(2)

    overall = True
    for name in sorted(os.listdir(ws)):
        full = os.path.join(ws, name)
        if not (os.path.isdir(full) and os.path.isdir(os.path.join(full, "eval-results"))):
            continue
        mode = detect_mode(name)
        if not mode:
            continue
        if "tool_calling" in name:
            if not verify_tool_bucket(full, mode):
                overall = False
        elif "ruler" in name:
            if not verify_ruler_bucket(full, mode):
                overall = False
        else:
            if not verify_reasoning_bucket(full, mode):
                overall = False

    print("\nOVERALL_OK: %s" % ("True" if overall else "False"))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

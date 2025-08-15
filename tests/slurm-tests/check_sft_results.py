import argparse
import json
import os
import sys
from typing import Any, Dict


def load_json(p: str):
    """Load a JSON file from the given path."""
    if not os.path.isfile(p):
        raise FileNotFoundError(f"File not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_root(d: Dict[str, Any], bench_key: str) -> Dict[str, Any]:
    """
    Support two common JSON structures:
      1) {"aime24": {...}, "aime25": {...}}
      2) {"benchmarks": {"aime24": {...}, "aime25": {...}}}
    Returns the dictionary corresponding to the given benchmark key.
    """
    if bench_key in d:
        return d[bench_key]
    if "benchmarks" in d and bench_key in d["benchmarks"]:
        return d["benchmarks"][bench_key]
    raise KeyError(f"Cannot find benchmark '{bench_key}'. Top-level keys: {list(d.keys())}")


def get_metric(d: Dict[str, Any], bench_key: str, metric_path: str) -> float:
    """
    Retrieve a metric value from the JSON dictionary using a dot-path.

    Example metric_path values:
      - 'pass@1'
      - 'pass@1[avg-of-8].symbolic_correct'

    The dot-path is split on '.' and each part is used as a key
    to navigate nested dictionaries.
    """
    node: Any = _resolve_root(d, bench_key)
    for part in metric_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"Cannot find metric path '{metric_path}' under '{bench_key}'. "
                f"Missing key: '{part}'. Available keys: {list(node.keys()) if isinstance(node, dict) else type(node)}"
            )
        node = node[part]
    try:
        return float(node)
    except Exception as e:
        raise ValueError(f"Metric not numeric for {bench_key}.{metric_path}: {node!r}") from e


def main():
    ap = argparse.ArgumentParser(description="Compare after-training vs baseline metrics for AIME24/25.")
    ap.add_argument("--workspace", required=True, help="Base workspace directory containing eval results.")
    ap.add_argument("--baseline_path", default="", help="Optional explicit path to baseline metric.json")
    ap.add_argument("--after_path", default="", help="Optional explicit path to after-training metric.json")
    ap.add_argument(
        "--benchmarks", default="aime24,aime25", help="Comma-separated benchmark keys, e.g. 'aime24,aime25'"
    )
    ap.add_argument(
        "--metric",
        default="pass@1[avg-of-8].symbolic_correct",
        help="Dot-path to metric, e.g. 'pass@1' or 'pass@1[avg-of-8].symbolic_correct'",
    )
    ap.add_argument("--strict_increase", action="store_true", help="Require strict improvement (>) instead of >=.")
    args = ap.parse_args()

    # Resolve default metric file paths
    ws = args.workspace.rstrip("/")
    baseline_metric = args.baseline_path or os.path.join(ws, "evals", "baseline", "eval-results", "metrics.json")
    after_metric = args.after_path or os.path.join(ws, "evals", "after-training", "eval-results", "metrics.json")
    # Load both JSON files
    baseline = load_json(baseline_metric)
    after = load_json(after_metric)

    # Parse benchmarks and metric path
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    metric_path = args.metric

    all_ok = True
    lines = []
    for b in benches:
        # Get values from both baseline and after-training
        b_val = get_metric(baseline, b, metric_path)
        a_val = get_metric(after, b, metric_path)

        # Check comparison rule
        if args.strict_increase:
            ok = a_val > b_val
            cmp = ">"
        else:
            ok = a_val >= b_val
            cmp = ">="
        status = "OK" if ok else "FAIL"
        lines.append(f"{status}: after[{b}.{metric_path}]={a_val:.6f} {cmp} baseline[{b}.{metric_path}]={b_val:.6f}")
        if not ok:
            all_ok = False

    # Print results
    print(f"[check] baseline: {baseline_metric}")
    print(f"[check] after-training: {after_metric}")
    for ln in lines:
        print("[check]", ln)

    # Exit with error code if any check failed
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

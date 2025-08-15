import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from nemo_skills.pipeline.utils import get_cluster_config, get_tunnel


# ---------- Basic utilities ----------
def run_cmd(cmd: list[str], env=None, timeout=None) -> str:
    """Run a shell command and return combined stdout+stderr."""
    p = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


# Resolve absolute path to the recipe
REPO_ROOT = Path(__file__).resolve().parents[2]
SIMPLIFIED_RECIPE = REPO_ROOT / "recipes" / "openmathreasoning" / "scripts" / "simplified_recipe.py"


def build_launch_cmd(
    cluster: str,
    backend: str,
    workspace: str,
    num_gpus: int,
    project: str,
    expname: str,
    disable_wandb: bool,
) -> list[str]:
    """Build CLI for simplified_recipe.py (no sbatch here; recipe handles scheduling)."""
    assert SIMPLIFIED_RECIPE.exists(), f"simplified_recipe.py not found at {SIMPLIFIED_RECIPE}"
    base = [
        "python",
        str(SIMPLIFIED_RECIPE),
        "--cluster",
        cluster,
        "--workspace",
        workspace,
        "--num_gpus",
        str(num_gpus),
        "--training_backend",
        backend,
        "--expname_prefix",
        expname,
    ]
    if disable_wandb:
        base.append("--disable_wandb")
    else:
        if project:
            base += ["--wandb_project", project]
    return base


# ---------- Check Cluster Files ----------
def _open_sftp(cluster_config):
    """Open an SFTP client over an active SSH tunnel."""
    tunnel = get_tunnel(cluster_config)
    if getattr(tunnel, "session", None) is None or getattr(getattr(tunnel, "session", None), "client", None) is None:
        if hasattr(tunnel, "connect"):
            tunnel.connect()
    if getattr(tunnel, "session", None) is None or getattr(tunnel.session, "client", None) is None:
        raise RuntimeError("SSH tunnel is not connected.")
    sftp = tunnel.session.client.open_sftp()
    return tunnel, sftp


def remote_path_exists(cluster_config, remote_path: str) -> bool:
    """Return True if remote file/dir exists (via SFTP.stat)."""
    tunnel, sftp = _open_sftp(cluster_config)
    try:
        sftp.stat(remote_path)
        return True
    except FileNotFoundError:
        return False
    finally:
        try:
            sftp.close()
        except Exception:
            pass


def remote_dir_list_json(cluster_config, remote_dir: str) -> list[str]:
    """List *.json files under a remote directory (non-recursive)."""
    tunnel, sftp = _open_sftp(cluster_config)
    try:
        try:
            st = sftp.stat(remote_dir)
            if not stat.S_ISDIR(st.st_mode):
                return []
        except FileNotFoundError:
            return []
        names = sftp.listdir(remote_dir)
        return [f"{remote_dir.rstrip('/')}/{n}" for n in names if n.lower().endswith(".json")]
    finally:
        try:
            sftp.close()
        except Exception:
            pass


def remote_read_json(cluster_config, remote_file: str) -> dict:
    """Read a JSON file from remote to a Python dict."""
    tunnel, sftp = _open_sftp(cluster_config)
    try:
        with sftp.open(remote_file, "r") as f:
            data = f.read()
        return json.loads(data)
    finally:
        try:
            sftp.close()
        except Exception:
            pass


def stage_done_dir(workspace: str, stage: str) -> str:
    """
    Map logical stage -> summarize-results directory.
    'baseline' -> {workspace}/summarize-results/baseline
    'final'    -> {workspace}/summarize-results/after-training
    """
    stage = stage.lower()
    if stage == "baseline":
        return f"{workspace}/summarize-results/baseline"
    if stage == "final":
        return f"{workspace}/summarize-results/after-training"
    raise ValueError(f"Unknown stage: {stage}")


def wait_for_stage_ready(
    cluster: str,
    workspace: str,
    stage: str,
    poll_interval=300,
    timeout_sec=60 * 60 * 12,
    config_dir=None,
) -> str:
    """
    Poll remote cluster until the summarize-results dir exists and contains at least one *.json.
    Returns the chosen JSON file path (prefers common names).
    """
    cluster_config = get_cluster_config(cluster, config_dir)
    target_dir = stage_done_dir(workspace, stage)
    preferred = ("results.json", "summary.json", "metrics.json")
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        if remote_path_exists(cluster_config, target_dir):
            jsons = remote_dir_list_json(cluster_config, target_dir)
            if jsons:
                lower_map = {p.split("/")[-1].lower(): p for p in jsons}
                for name in preferred:
                    if name in lower_map:
                        print(f"[wait] Using preferred summary file: {lower_map[name]}")
                        return lower_map[name]
                jsons_sorted = sorted(jsons)
                print(f"[wait] Using summary file: {jsons_sorted[-1]}")
                return jsons_sorted[-1]
        print(f"[wait] Stage '{stage}' not ready at {target_dir}; retry in {poll_interval}s")
        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for stage '{stage}' at {target_dir}")


# ---------- W&B fetch (version-agnostic) ----------
def fetch_metrics_wandb(project: str, exp_prefix: str, keys: tuple[str, str], wait_s=900, poll=15):
    """
    Fetch metrics from W&B and wait until desired keys appear.
    Compatible with older wandb versions (no util.auto_project_entity).
    """
    import wandb

    wandb.login()
    api = wandb.Api()
    candidates = []
    if "/" in project:
        candidates.append(project)
    else:
        entity = os.getenv("WANDB_ENTITY")
        if entity:
            candidates.append(f"{entity}/{project}")
        candidates.append(project)
    end = time.time() + wait_s
    last_err = None
    while time.time() < end:
        for path in candidates:
            try:
                runs = api.runs(path, {"order": "-createdAt"})
            except Exception as e:
                last_err = e
                continue
            for r in runs:
                if not r.name or not r.name.startswith(exp_prefix):
                    continue
                hist = r.history(samples=1, keys=list(keys))
                if len(hist) > 0:
                    row = hist.iloc[-1].to_dict()
                    if all(k in row and row[k] is not None for k in keys):
                        return {k: float(row[k]) for k in keys}
        time.sleep(poll)
    raise TimeoutError(f"W&B metrics not ready. Last error: {last_err}")


def test_openmathreasoning_end2end(pytestconfig):
    """
    Submit pipeline; wait for baseline and final summarize-results on remote;
    fetch metrics (W&B first if enabled, else remote JSON); assert thresholds.
    """
    # --- read CLI options ---
    cluster = pytestconfig.getoption("cluster")
    backend = pytestconfig.getoption("backend")
    workspace = pytestconfig.getoption("workspace")
    num_gpus_opt = pytestconfig.getoption("num_gpus")
    num_gpus = int(num_gpus_opt) if num_gpus_opt is not None else 8
    wandb_project = pytestconfig.getoption("wandb_project")
    expname = pytestconfig.getoption("expname")
    disable_wandb = pytestconfig.getoption("disable_wandb")

    # --- launch recipe (submits Slurm jobs and exits quickly) ---
    cmd = build_launch_cmd(cluster, backend, workspace, num_gpus, wandb_project, expname, disable_wandb)
    print(f"[pytest] Launch: {' '.join(cmd)}")
    print(f"[pytest] W&B disabled: {disable_wandb}")
    os.environ.setdefault("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK", "1")
    run_cmd(cmd)

    # --- wait for baseline stage ready (initial eval summary json available) ---
    baseline_json_remote = wait_for_stage_ready(
        cluster=cluster,
        workspace=workspace,
        stage="baseline",
        poll_interval=300,  # check every 5 minutes
        timeout_sec=60 * 60 * 12,  # up to 12 hours
    )

    # --- wait for final stage ready (after-training summary json available) ---
    final_json_remote = wait_for_stage_ready(
        cluster=cluster, workspace=workspace, stage="final", poll_interval=300, timeout_sec=60 * 60 * 12
    )

    # --- metric keys & expectations ---
    metric_keys = ("initial_eval", "final_eval")
    expected_initial = ">=0.20"
    expected_final = ">=0.25"

    # --- helper to read metrics from a remote summary json (with key aliases) ---
    def metrics_from_remote_file(remote_json_path: str):
        data = remote_read_json(get_cluster_config(cluster, None), remote_json_path)
        aliases = [
            (metric_keys[0], metric_keys[1]),
            ("initial_eval_score", "final_eval_score"),
            ("initial/score", "final/score"),
            ("initial_accuracy", "final_accuracy"),
        ]
        for a, b in aliases:
            if a in data and b in data:
                return {metric_keys[0]: float(data[a]), metric_keys[1]: float(data[b])}
        raise AssertionError(f"No expected keys in {remote_json_path}. Keys: {list(data.keys())}")

    # --- get initial/final via W&B (if enabled) or remote files ---
    initial_val, final_val = None, None

    if not disable_wandb:
        # try W&B first
        try:
            metrics = fetch_metrics_wandb(wandb_project, expname, metric_keys)
            print("[pytest] Metrics from W&B:", metrics)
            initial_val = metrics[metric_keys[0]]
            final_val = metrics[metric_keys[1]]
        except Exception as e:
            print(f"[pytest] W&B fetch failed: {e}")

    # if W&B not used or failed, fall back to remote files (read baseline and final separately)
    if initial_val is None or final_val is None:
        # initial from baseline summary json
        m_baseline = metrics_from_remote_file(baseline_json_remote)
        # final from after-training summary json
        m_final = metrics_from_remote_file(final_json_remote)
        # merge (prefer exact stage-specific values)
        initial_val = m_baseline[metric_keys[0]]
        final_val = m_final[metric_keys[1]]
        print("[pytest] Metrics from remote files:", {"initial_eval": initial_val, "final_eval": final_val})

    # --- assertions ---
    def check_expect(expect: str, value: float):
        e = expect.strip()
        if e.startswith(">="):
            assert value >= float(e[2:]), f"{value} < {e}"
        elif e.startswith("<="):
            assert value <= float(e[2:]), f"{value} > {e}"
        else:
            assert abs(value - float(e)) < 1e-6, f"{value} != {e}"

    check_expect(expected_initial, initial_val)
    check_expect(expected_final, final_val)
    assert final_val >= initial_val, "Final score did not improve vs initial"

### Objective
Introduce a dedicated Nemo Evaluator pipeline subcommand (`ns nemo_evaluator`) that orchestrates evaluator runs without starting LLM servers. It selects the correct container per task and runs the evaluator launcher, passing parameters to an underlying `NemoEvaluatorGeneration` (a `GenerationTask` subclass) for container policy and, optionally, command construction.

### Scope of edits (only planning here)
- Add `nemo_skills/pipeline/nemo_evaluator.py` as a new Typer subcommand, registered under the root CLI as `ns nemo_evaluator`.
- Add `nemo_skills/inference/nemo_evaluator.py` with:
  - `NemoEvaluatorConfig` (minimal additions for evaluator use; may reuse parts of `GenerateSolutionsConfig` only if needed for Hydra compatibility).
  - `NemoEvaluatorGeneration` (subclass of `GenerationTask`) exposing:
    - `get_client_container(task_queries, ...)` to decide the container via evaluator mapping
    - optionally, a helper to build the evaluator shell command
  - `GENERATION_TASK_CLASS = NemoEvaluatorGeneration` for reuse/tests.

### Step-by-step plan
1) Create `pipeline/nemo_evaluator.py`
   - Implement a slim Typer CLI: `ns nemo_evaluator` with core options: `--cluster`, `--output_dir`, `--expname`, `--tasks`, optional `--log_dir`, `--mount_paths`, `--partition`, `--qos`, `--time_min`, `--exclusive`, `--with_sandbox`, and evaluator Hydra overrides `++nemo_eval_config_dir`, `++nemo_eval_config_name`, plus mapping flags (e.g., `--latest_mapping`, `--tasks_mapping_toml`).
   - No server-related options and no LLM inference parameters. Orchestration knobs remain: `--partition`, `--qos`, `--time_min`, `--exclusive`, `--reuse_code`, `--reuse_code_exp`, `--dependent_jobs`, `--run_after`, `--dry_run`, `--log_dir`, etc.
   - Add job-level resource knobs for the evaluator container (not a server): `--job_gpus`, `--job_nodes`. These set the `gpus`/`nodes` on the client `Command` and thus on the `HardwareConfig`.
   - Build the evaluator command and `.done` tracking directly (no `get_generation_cmd`).

2) Define Nemo Evaluator-specific generator module `inference/nemo_evaluator.py`
   - Config design: add evaluator-specific knobs:
     - `nemo_eval_config_dir: str | None = None`
     - `nemo_eval_config_name: str = "config"`
     - `stream_subprocess_output: bool = True`
   - Keep other fields minimal; do not require LLM-specific parameters.
   - Class outline:
```python
# nemo_skills/inference/nemo_evaluator.py
import hydra
from dataclasses import field
from nemo_skills.utils import nested_dataclass, setup_logging, get_help_message
from nemo_skills.inference.generate import GenerationTask

@nested_dataclass(kw_only=True)
class NemoEvaluatorConfig:
    # Copy all fields from GenerateSolutionsConfig verbatim to keep full arg parity
    # (input_file, output_file, prompt_config, tokenizer, inference, server, sandbox, etc.)
    # Plus Nemo Evaluator-specific additions below:
    nemo_eval_config_dir: str | None = None
    nemo_eval_config_name: str = "config"
    stream_subprocess_output: bool = True

class NemoEvaluatorGeneration(GenerationTask):
    @classmethod
    def get_generation_default_args(cls) -> str:
        # Preserve compatibility; no additional default args for now
        return ""

    def __init__(self, cfg: NemoEvaluatorConfig):
        # Skip LLM/model initialization; we call nemo_evaluator_launcher instead
        self.cfg = cfg

    def generate(self):
        # 1) Build nemo_evaluator_launcher RunConfig via Hydra (config_dir/name + overrides)
        # 2) For each task name (or a single one), build eval command via get_eval_factory_command
        # 3) Spawn subprocess with shell command; stream stdout if configured
        # 4) Surface return code and basic errors
        pass

# hydra boilerplate mirroring inference/generate.py
GENERATION_TASK_CLASS = NemoEvaluatorGeneration

@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: NemoEvaluatorConfig):
    cfg = NemoEvaluatorConfig(_init_nested=True, **cfg)
    task = NemoEvaluatorGeneration(cfg)
    task.generate()
```
   - Help/usage text: provide evaluator-centric help; do not surface server or LLM arguments.

3) Wrap nemo_evaluator_launcher inside `NemoEvaluatorGeneration.generate`
   - Implementation responsibilities:
     - Build a `RunConfig` via `RunConfig.from_hydra(config_dir, config_name)`.
     - Load tasks mapping (`load_tasks_mapping`), iterate over `run_cfg.evaluation.tasks` and resolve each task by name via `get_task_from_mapping`.
     - For each resolved task, call `get_eval_factory_command(run_cfg, task_cfg, task_definition)` to obtain the shell command.
     - Execute with `subprocess.Popen(..., shell=True)` and, if `stream_subprocess_output` is True, print lines as they arrive.
     - Propagate non-zero return codes as exceptions.
   - Notes:
     - We intentionally do not call `GenerationTask.setup_llm` or LLM-related utilities in this generator.
     - `wait_for_server` is unnecessary; evaluator talks to its own endpoints from its config.
     - `output_file`/`output_dir` in `GenerateSolutionsConfig` can be used to form an override like `++evaluation.nemo_evaluator_config.config.output_dir=...` (optional in v1; can be derived if user sets output_dir in the pipeline CLI).

4) Register a new CLI subcommand
   - Expose `ns nemo_evaluator` via `nemo_skills/pipeline/nemo_evaluator.py` under the existing `ns` entrypoint.
   - Do not piggyback on `pipeline.generate` or `--generation_module`.

### Integration and extensibility notes
- Optional: expose evaluator as a `GenerationType` for reuse in contexts that still call into generation modules, but this is not required for `ns nemo_evaluator`.
- Metrics: `nemo_skills.evaluation.metrics.nemo_evaluator_metrics` exists; in future, we can wire its consumption if we want to post-process evaluator outputs inside `NemoEvaluatorGeneration.postprocess()`.
- Sandboxing/hosting: Keep the CLI knobs present but they are ignored by the generator; if `server_gpus>0` are passed, `pipeline` would try to start a model server. Users should pass `--server_gpus 0` (or just omit hosting args) for evaluator runs. We can also override `get_server_command_fn` to a no-op if we want to hard-disable hosting in this path.
- Hydra: Config store name remains `base_generation_config` for parity with the rest of the tooling.

### How to run the Nemo Evaluator launcher (current and target UX)

This aligns with the existing `nemo_evaluator_runner.py` implementation and the pipeline launcher behavior.

- Direct invocation (quick sanity check):

  ```bash
  # Uses an embedded Hydra config and a hardcoded task name (aime_2025_nemo)
  # Streams evaluator output to stdout
  python -m nemo_evaluator_runner
  ```

  Notes:
  - Builds a `RunConfig` via `RunConfig.from_hydra(...)` using a temporary folder and an embedded YAML.
  - Resolves a task via `load_tasks_mapping()` and `get_task_from_mapping("aime_2025_nemo", ...)`.
  - Constructs the shell command with `get_eval_factory_command(...)` and executes it with `subprocess.Popen(..., shell=True)`.

- Via the pipeline (recommended orchestration path):

  ```bash
  # Run through the pipeline entrypoint once `ns nemo_evaluator` subcommand exists
  ns nemo_evaluator \
    --cluster local \
    --output_dir /path/to/output_dir \
    --expname evaluator-aime \
    --tasks aime_2025_nemo \
    ++nemo_eval_config_dir=/configs/evaluator \
    ++nemo_eval_config_name=config
  ```

  What happens:
  - The dedicated `ns nemo_evaluator` pipeline subcommand constructs an evaluator command (no server), handles `.done` tracking, and passes only evaluator-relevant Hydra overrides.
  - Servers are not started in this flow; there is no `--server_type` knob here.

### Passing evaluator config and selecting tasks

Current behavior (no code changes):
- Config directory/name and task are embedded in `nemo_evaluator_runner.py` (temporary config + `aime_2025_nemo`). They are not overridable via CLI yet.

Planned (to match the design in this plan):
- Add generator-specific knobs so users can control the evaluator without changing code:
  - `++nemo_eval_config_dir=/path/to/configs`
  - `++nemo_eval_config_name=config`
  - Optional task selector to replace hardcoded `aime_2025_nemo` (single or list).
- Use these to construct `RunConfig.from_hydra(...)` and to resolve tasks via `load_tasks_mapping()`/`get_task_from_mapping()`.

### Practical example (target UX via `ns nemo_evaluator`)

```bash
# Run the provided example config from the repo tests folder
ns nemo_evaluator \
  --cluster local \
  --output_dir /results \
  --expname evaluator-run \
  --tasks ifeval \
  ++nemo_eval_config_dir=tests/data/nemo_evaluator \
  ++nemo_eval_config_name=example-eval-config
```

Notes:
- `output_dir` is used for logs and `.done` tracking; evaluator output placement is controlled via the Hydra config (e.g., in `tests/data/nemo_evaluator/example-eval-config.yaml`).
- If chunking/seeds are supported in this subcommand, they only affect orchestration; the evaluator command itself is agnostic.

### Task-based container selection via evaluator mapping

Goal: Select the correct container image for the client process based on the evaluator task, using the mapping provided by the launcher.

Relevant APIs from the launcher (already imported in the runner):
- `load_tasks_mapping(latest: bool = False, mapping_toml: Path | str | None = None) -> dict`
- `get_task_from_mapping(query: str, mapping: dict) -> dict`

What the mapping provides:
- Each resolved task dictionary contains at least: `{"task": <name>, "harness": <name>, "container": <image>, "endpoint_type": <type>, ...}`.
- The `container` value is the image string recommended for that harness/task.

Proposed selection algorithm:
1) Determine the set of evaluator task queries to run (e.g., from Hydra `evaluation.tasks` or an explicit CLI argument).
2) Call `load_tasks_mapping()` (use `latest=True` for freshness if wanted; default packaged mapping is fine for reproducibility).
3) For each task query, call `get_task_from_mapping(query, mapping)` and collect `task_info["container"]`.
4) Validate that all selected tasks resolve to the same container image; if not, either:
   - Group tasks by container image and create separate job groups per image, or
   - Fail fast with a clear error and ask the user to split runs by container.
5) Use the resulting container image for the client `Command` instead of the default `cluster_config["containers"]["nemo-skills"]`.

Multi-task considerations:
- If multiple evaluator tasks are specified and they map to different images, grouping by container is preferred to preserve correctness. Pipelines will then emit one job-chain per container group.
- If grouping is not implemented initially, enforce a single-container constraint with a helpful error.

Mapping freshness and overrides:
- Default to packaged mapping for stability. Add an opt-in flag to request the latest mapping from GitHub.
- Optionally support a `--tasks_mapping_toml` path for fully offline or pinned mapping usage.

Safeguards:
- If mapping fails to load, fall back to the default client container (`nemo-skills`) and warn.
- If a task name is ambiguous without harness prefix, surface the list suggested by `get_task_from_mapping` and request a disambiguated `harness.task` query.

### Injecting container selection into the new `ns nemo_evaluator` pipeline

Preferred approach (localized, low-risk):
- Implement a dedicated `pipeline/nemo_evaluator.py` Typer subcommand registered under the root CLI (`ns nemo_evaluator`).
- Early in the command construction, resolve evaluator tasks and compute `client_container` using the algorithm above.
- When creating the client `Command`, set `container=client_container` instead of `cluster_config["containers"]["nemo-skills"]`.
- Keep `get_server_command_fn()` a no-op for this path to avoid starting a model server.

Sketch (illustrative, not final code):

```python
from nemo_evaluator_launcher.common.mapping import load_tasks_mapping, get_task_from_mapping

def _resolve_client_container(task_queries: list[str], use_latest: bool = False, mapping_toml: str | None = None) -> str:
    mapping = load_tasks_mapping(latest=use_latest, mapping_toml=mapping_toml)
    containers = {
        get_task_from_mapping(q, mapping)["container"] for q in task_queries
    }
    if len(containers) != 1:
        raise ValueError(f"Tasks map to multiple containers: {sorted(containers)}. Split the run per container.")
    return next(iter(containers))

# ... later, when building the client Command
client_container = _resolve_client_container(task_queries, use_latest=flags.latest_mapping, mapping_toml=flags.mapping_toml)
client_cmd = Command(
    command=generation_cmd,
    container=client_container,  # override per-task container
    name=task_name,
    installation_command=installation_command,
    metadata={"log_prefix": "main", "environment": client_env},
)
```

Where to get `task_queries` in the pipeline:
- Primary: add a CLI option on `ns nemo_evaluator` like `--tasks "aime_2025_nemo,math_foo"` and split by comma.
- Optional: if `++nemo_eval_config_dir/name` is provided, the subcommand can build a `RunConfig` up front and read `cfg.evaluation.tasks` to augment or validate the list.

Fallbacks and overrides:
- If `--client_container <image>` is provided, use it directly and skip mapping.
- If mapping resolution fails, fall back to `cluster_config["containers"]["nemo-skills"]` with a warning.

Why a dedicated pipeline module:
- Keeps existing `pipeline/generate.py` unchanged.
- Allows evaluator-specific behavior (no server, per-task container, different logging) without affecting other generation flows.
- Still benefits from the shared declarative pipeline and scheduling (dependent jobs, sandbox, logs, done-files).

### What to keep vs drop from `pipeline/generate.py` for evaluator

Given evaluator runs do not start model servers or perform LLM inference via our generator, we should adopt a slim subset of the pipeline features and adjust command construction accordingly.

- Keep (valuable orchestration pieces):
  - Cluster config resolution, mounts checking, and log directory handling.
  - Declarative `Pipeline` and `CommandGroup` with job dependencies, logs, and `.done` file tracking.
  - Optional sandbox sidecar if explicitly requested (default off).
  - Experiment lifecycle and code packaging/reuse (`reuse_code`, `reuse_code_exp`).
  - Scheduling knobs: `partition`, `qos`, `time_min`, `exclusive`, `run_after`, `dependent_jobs`, `dry_run`.

- Drop/disable (not applicable here):
  - Server hosting and `configure_client` logic; enforce `--server_type none` and ignore `server_gpus`/`server_nodes`.
  - LLM-specific extra arguments (e.g., `++inference.*`, prompt/tokenizer options) injected via `get_generation_cmd`.
  - Seed/chunk-based prompt generation semantics; chunking can remain supported only for orchestration if desired, but not required by the evaluator.

- Replace with evaluator-specific command builder:
  - Construct a command that executes the evaluator module/runner (e.g., `python -m nemo_skills.inference.nemo_evaluator` or `python -m nemo_evaluator_runner`) with Hydra overrides for the evaluator config location and any task-specific overrides.
  - Wrap with the same `.done` file touch logic the pipeline uses, so retries and “skip completed” continue to work.
  - Do not add LLM inference flags; only pass evaluator-relevant Hydra overrides (config dir/name, mapping freshness, tasks, output_dir if desired).
  - If GPU is needed for certain evaluator tasks, allocate via `--job_gpus/--job_nodes`; the client `Command` will request those resources directly (no server component).

### How orchestration still works without servers

- The `ns nemo_evaluator` subcommand still constructs a `CommandGroup` and `Pipeline`, but omits the server `Command`. The only `Command` is the client that runs the evaluator.
- Job resources are requested at the job level via `--job_gpus`/`--job_nodes` and translated to `HardwareConfig` through the client `Command`'s `gpus`/`nodes`.
- All standard orchestration remains intact: cluster selection, mounts, logging, experiment management, dependencies, retries, `.done` tracking, optional sandbox, and dry-run.
- Container selection is performed up front (via `NemoEvaluatorGeneration.get_client_container`) so scheduling uses the correct image per task.

### Centralizing container choice in `NemoEvaluatorGeneration`

Rationale: The evaluator task definition is the source of truth for which container image to use. Placing container inference inside `NemoEvaluatorGeneration` keeps the policy close to the evaluator logic and avoids duplicating mapping logic in the pipeline.

- Proposed API on the generation class:

  ```python
  class NemoEvaluatorGeneration(GenerationTask):
      @classmethod
      def get_client_container(cls, *, task_queries: list[str], use_latest: bool = False, mapping_toml: str | None = None) -> str:
          """Return the recommended client container image for the given evaluator tasks.
          Must resolve to a single image; raise if tasks map to different images."""
          # Implementation uses load_tasks_mapping/get_task_from_mapping and returns the `container` field
  ```

- How the pipeline uses it (without server hosting):
  - In `pipeline/nemo_evaluator.py`, before creating the client `Command`, call `NemoEvaluatorGeneration.get_client_container(...)` with task list resolved from the evaluator Hydra config or from a CLI arg.
  - Set `container=<returned_image>` on the client `Command`.
  - Keep `server_config=None` and skip server `Command` creation entirely.

- Benefits of this split:
  - The generation class owns the evaluator mapping logic and can evolve independently.
  - The pipeline stays a thin orchestrator that simply asks the generation class for the correct container and constructs the job.
  - Testing is easier: the container policy can be unit-tested at the classmethod level.

- Edge cases and policy:
  - Multiple tasks resolving to different images: either group by container in the pipeline (advanced) or raise with a precise error asking users to split runs (initial version).
  - Explicit override: if the user passes `--client_container`, the pipeline should prefer the override and skip classmethod resolution.
  - Mapping availability: support `latest` and `mapping_toml` knobs; if mapping cannot be loaded, fall back to the default `nemo-skills` container with a warning.

### Testing plan (minimal mocking, high-confidence)

- Unit tests (no network, minimal mocks):
  - Container resolution:
    - Single-task maps to one container via `get_client_container` using a small local `mapping.toml` (pass via `mapping_toml`).
    - Multiple tasks map to the same container (OK).
    - Multiple tasks map to different containers -> raises helpful error.
    - Ambiguous task name without harness -> raises with suggested fully-qualified options.
  - Command builder:
    - `build_eval_command` includes `++nemo_eval_config_dir/name` and forwards passthrough overrides.
    - Tasks are embedded correctly (either via generator `--tasks` or `++evaluation.tasks=...` when supported).
    - Shell-safe: spaces/quotes in paths do not break the command (basic cases).
  - CLI parsing (Typer):
    - `--tasks` comma list parsed to list.
    - `--job_gpus/--job_nodes` are integers and optional.
    - Mapping flags `--latest_mapping/--tasks_mapping_toml` accepted and forwarded.

- Pipeline orchestration (local executor, no server):
  - Dry run builds a `Pipeline` with exactly one client `Command` and no server `Command`.
  - Client `Command.container` equals container resolved by `get_client_container`.
  - HardwareConfig reflects `--job_gpus/--job_nodes` on the client.
  - `.done` tracking is wired: command string ends with touch of the `.done` file (single-job path). If chunking is added later, verify merge pattern.
  - Optional sandbox off by default; enabled when `--with_sandbox`.

- Error handling and edge cases:
  - Unknown/ambiguous `--tasks` -> fails fast with clear message; no Pipeline is run.
  - Mapping download failure path (when `--latest_mapping`) -> falls back to packaged mapping (simulate by pointing to bad URL or mocking downloader to raise; prefer packaged-path test without network).
  - Non-zero evaluator exit (simulate by substituting a failing command via a test hook in `build_eval_command`) -> task fails, `.done` not touched.

- Optional smoke (skipped if deps missing):
  - If `nemo_evaluator_launcher` is installed, run a tiny evaluator config that produces a benign command and verify stdout is streamed (can be a very small config or a mocked task that echoes). Mark as `xfail` when package is unavailable.

Notes on minimizing mocks:
- Prefer passing a temporary `mapping.toml` via `mapping_toml` over mocking `load_tasks_mapping`.
- Prefer `dry_run` and command string assertions over intercepting subprocess.
- If a failing command is needed, inject via a narrow test hook in `build_eval_command` or environment flag rather than broad patching.

### Clarification questions
- Will tasks be provided entirely via the Hydra config under `evaluation.tasks`? Any defaults you want us to assume?
- Should we forbid starting an inference server in this flow (override `get_server_command_fn` to a no-op), or keep it possible in case some evaluator targets rely on locally hosted models?
- Is there a default `nemo_eval_config_dir` you want to ship or reference?

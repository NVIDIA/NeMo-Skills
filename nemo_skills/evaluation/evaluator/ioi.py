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
import asyncio
import json
import multiprocessing
import os
import re
import time
from typing import Dict, Set

from nemo_skills.code_execution.sandbox import LocalSandbox
from nemo_skills.file_utils import jdump
from nemo_skills.utils import nested_dataclass, unroll_files


@nested_dataclass(kw_only=True)
class IOIEvaluatorConfig:
    dataset: str = "ioi"
    num_workers: int = 4  # number of test workers
    test_batch_size: int = 5  # number of tests to run concurrently
    # where test cases are stored in automatically mounted eval datasets folder.
    test_file: str = "/eval_dataset/ioi/test_metadata.json"
    overwrite: bool = False


# A dedicated event loop for all synchronous sandbox operations in the main process.
_precompile_loop = None  # type: asyncio.AbstractEventLoop | None


def _sandbox_exec_sync(sandbox: LocalSandbox, cmd: str, *, language: str = "shell", timeout: int = 120):
    """Run sandbox.execute_code synchronously with a persistent event loop.

    Re-creating and immediately closing a loop for every call can leave background
    tasks (e.g., httpx/anyio socket reads) unfinished, causing "Event loop is
    closed" errors.  We therefore maintain a single loop for all such
    pre-compile operations.
    """
    global _precompile_loop
    if _precompile_loop is None:
        _precompile_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_precompile_loop)

    return _precompile_loop.run_until_complete(sandbox.execute_code(cmd, language=language, timeout=timeout))[0]


def wait_for_sandbox(sandbox, timeout: int = 120, poll: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _sandbox_exec_sync(sandbox, "echo hello world", language="shell", timeout=10)
            if resp.get("stdout", "").strip() == "hello world":
                return
        except Exception:
            pass
        time.sleep(poll)
    raise RuntimeError(f"Sandbox not ready after waiting {timeout}s")


def init_worker(sandbox_arg):
    global worker_sandbox
    worker_sandbox = sandbox_arg
    # Create and set a dedicated event loop for this worker process.
    # Re-using the same loop for all subsequent sandbox calls avoids the
    # "Event loop is closed" error that occurs when each call spins up
    # and closes its own loop (as happens with asyncio.run).
    global worker_loop
    worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(worker_loop)


def _precompile_grader(
    problem_name: str, grader_files, compile_code: str, run_code: str, sandbox: LocalSandbox
) -> str:
    """Precompile checker/grader for a problem once and return the directory path."""
    pre_dir = f"/tmp/ioi_pre_{problem_name}_{os.getpid()}"
    # Build shell script to create files and invoke compile.sh.
    creation_cmds = [
        f"mkdir -p {pre_dir}/graders",
    ]
    # Dump grader related files
    for filepath, content in grader_files:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            creation_cmds.append(f"mkdir -p {pre_dir}/{dir_name}")
        creation_cmds.append(f"cat <<'_EOT_' > {pre_dir}/{filepath}\n{content}\n_EOT_\n")

    # Write compile.sh and run.sh as provided (needed later in workers)
    creation_cmds.append(
        f"cat <<'_EOT_' > {pre_dir}/compile.sh\n{compile_code}\n_EOT_\nchmod +x {pre_dir}/compile.sh\n"
    )
    creation_cmds.append(f"cat <<'_EOT_' > {pre_dir}/run.sh\n{run_code}\n_EOT_\nchmod +x {pre_dir}/run.sh\n")

    setup_script = "\n".join(creation_cmds)
    # 1. create files
    _sandbox_exec_sync(sandbox, setup_script, language="shell", timeout=120)

    # 2. run compile.sh but ignore final failure when problem cpp missing
    _sandbox_exec_sync(sandbox, f"cd {pre_dir} && ./compile.sh || true", language="shell", timeout=120)

    return pre_dir


def run_test_case(task_args: dict, worker_id: int) -> dict:
    global worker_sandbox

    unique_dir = f"/tmp/ioi_run_{worker_id}_{os.getpid()}"

    try:
        # 1. Create all necessary files in one batch command
        precompiled_dir = task_args.get("precompiled_dir")
        file_creation_commands = [
            f"mkdir -p {unique_dir}",
            # Copy precompiled artifacts (graders/, checker/, compile.sh, run.sh)
            f"cp -r {precompiled_dir}/* {unique_dir}/",
        ]

        # Write the candidate solution CPP file overwriting if necessary
        file_creation_commands.append(
            f"cat <<'_EOT_' > {unique_dir}/graders/{task_args['problem_id']}.cpp\n{task_args['generated_code']}\n_EOT_\n"
        )

        # Prepare input and expected output files
        file_creation_commands.append(f"cat <<'_EOT_' > {unique_dir}/input.txt\n{task_args['test_input']}\n_EOT_\n")
        file_creation_commands.append(
            f"cat <<'_EOT_' > {unique_dir}/correct_output.txt\n{task_args['test_output']}\n_EOT_\n"
        )

        setup_script = "\n".join(file_creation_commands)
        setup_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(setup_script, language='shell', timeout=120)
        )
        if setup_result.get('stderr'):
            raise Exception(f"File setup failed: {setup_result['stderr']}")

        # 2. Compile only the problem solution (skip checker/grader recompilation)
        # Compile the solution together with optional grader/stub sources without
        # recompiling the checker/manager again.
        compile_command = (
            f"cd {unique_dir} && "
            f"SRC=\"graders/{task_args['problem_id']}.cpp\"; "
            f"[ -e graders/grader.cpp ] && SRC=\"$SRC graders/grader.cpp\"; "
            f"[ -e graders/stub.cpp ] && SRC=\"$SRC graders/stub.cpp\"; "
            f"g++ -DEVAL -std=gnu++17 -O2 -pipe -s -o graders/{task_args['problem_id']} $SRC"
        )
        compile_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(compile_command, language='shell', timeout=120)
        )

        result = {
            "compile_success": not compile_result.get('stderr'),
            "compile_stdout": compile_result.get('stdout', ''),
            "compile_stderr": compile_result.get('stderr', ''),
            "run_stdout": "",
            "run_stderr": "",
            "score": 0.0,
        }

        if not result["compile_success"]:
            return result

        # 3. Run the code
        run_command = f"cd {unique_dir} && ./run.sh"
        run_result, _ = worker_loop.run_until_complete(
            worker_sandbox.execute_code(run_command, language='shell', timeout=120)
        )

        run_stdout = run_result.get('stdout', '')
        run_stderr = run_result.get('stderr', '')

        result.update(
            {
                "run_stdout": run_stdout,
                "run_stderr": run_stderr,
            }
        )

        try:
            result["score"] = float(result["run_stdout"].strip())
        except (ValueError, TypeError):
            result["score"] = 0.0

        return result

    except Exception as e:
        return {"score": 0.0, "output": "", "error": str(e)}

    finally:
        # 4. Clean up the directory
        # Fire and forget; ignore return values
        try:
            worker_loop.run_until_complete(
                worker_sandbox.execute_code(f"rm -rf {unique_dir}", language='shell', timeout=120)
            )
        except Exception:
            pass


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


def add_includes(code: str, problem_id: str) -> str:
    """
    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code
    # has most of the useful functions
    code_header = '#include <bits/stdc++.h>\n'
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + '\n'
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    # add missing dummy implementations for IOI 25 triples problem
    dummy = ""
    if problem_id == "triples":
        has_count = re.search(r"\bcount_triples\s*\(", code) is not None
        has_construct = re.search(r"\bconstruct_range\s*\(", code) is not None
        if has_construct and not has_count:
            dummy += "long long count_triples(std::vector<int> H){return 0LL;}\n"
        elif has_count and not has_construct:
            dummy += "std::vector<int> construct_range(int M,int K){return {};}\n"
    return code_header + code + ("\n" + dummy if dummy else "")


def eval_ioi(cfg):
    eval_config = IOIEvaluatorConfig(_init_nested=True, **cfg.eval_config)
    sandbox = LocalSandbox()
    wait_for_sandbox(sandbox)
    batch_size = eval_config.test_batch_size
    start_time = time.monotonic()
    if not os.path.exists(eval_config.test_file):
        raise ValueError(f"Failed to find test cases in eval dataset directory: {eval_config.test_file}")

    with open(eval_config.test_file) as f:
        metadata = json.load(f)

    pool = multiprocessing.Pool(processes=batch_size, initializer=init_worker, initargs=(sandbox,))

    for jsonl_file in unroll_files(cfg.input_files):
        all_samples = []
        with open(jsonl_file) as f:
            for line in f:
                sample = json.loads(line)
                all_samples.append(sample)

        if len(all_samples) == 0:
            raise ValueError(
                f"No samples found in the file {jsonl_file}.\n"
                f"Make sure the file contains jsonl data with 'codes' key which is a list containing "
                f"individual code samples."
            )

        for sample in all_samples:
            if 'eval_status' not in sample:
                sample['eval_status'] = {'complete': False, 'eval_error_log': []}

        if not eval_config.overwrite:
            samples = [s for s in all_samples if not s['eval_status']['complete']]
        else:
            samples = all_samples

        print(f"Processing {len(samples)} entries that have not been evaluated")

        outputs = []
        # Cache for precompiled grader dirs keyed by IOI problem id
        precompiled_cache: Dict[str, str] = {}

        for x, entry in enumerate(samples):
            try:
                print(f"Evaluating {x}/{len(samples)}")
                completion = extract_final_cpp_block(entry['generation'])
                completion = add_includes(completion, entry['ioi_id'])

                pid = entry['ioi_id']
                if pid not in precompiled_cache:
                    precompiled_cache[pid] = _precompile_grader(
                        problem_name=pid,
                        grader_files=entry['grader_files'],
                        compile_code=entry['compile'],
                        run_code=entry['run'],
                        sandbox=sandbox,
                    )
                pre_dir = precompiled_cache[pid]

                test_case_results = {}
                problem_name = entry['name']
                problem_metadata = metadata[problem_name]

                # -------------------------------------------------------------
                # Batch tests across *all* subtasks so that each pool invocation
                # works on a larger chunk, making the `test_batch_size` flag
                # meaningful and speeding up evaluation.
                # -------------------------------------------------------------

                # 1. Flatten all tests and create per-subtask bookkeeping.
                all_tests = []  # (subtask_name, test_name, test_data)
                subtask_state = {}
                for subtask, subtask_data in problem_metadata.items():
                    subtask_state[subtask] = {
                        "score": subtask_data['score'],
                        "precision": subtask_data['score_precision'],
                        "outputs": [],
                        "scores": [],
                        "passed": True,
                    }
                    for test_name, test_data in subtask_data['tests'].items():
                        all_tests.append((subtask, test_name, test_data))

                # 2. Walk through the flattened list in chunks of `batch_size`.
                for i in range(0, len(all_tests), batch_size):
                    # Filter out tests whose subtask already failed.
                    batch = [t for t in all_tests[i : i + batch_size] if subtask_state[t[0]]["passed"]]
                    if not batch:
                        continue

                    tasks = []
                    for _, _, test_data in batch:
                        # prepare precompiled dir
                        task_args = {
                            "generated_code": completion,
                            "problem_id": pid,
                            "precompiled_dir": pre_dir,
                            "test_input": test_data['input'],
                            "test_output": test_data['output'],
                        }
                        # local_idx is irrelevant after flattening
                        tasks.append((task_args, 0))

                    results = pool.starmap(run_test_case, tasks)

                    # 3. Scatter results back to the corresponding subtask.
                    for (subtask, test_name, _), result in zip(batch, results):
                        st = subtask_state[subtask]
                        result_with_name = dict(result)
                        result_with_name['test_name'] = test_name
                        st['outputs'].append(result_with_name)
                        st['scores'].append(float(result['score']))
                        if float(result['score']) == 0.0:
                            st['passed'] = False

                        # If compilation failed, surface the compiler output for easy debugging.
                        if not result.get("compile_success", True):
                            print(
                                f"Compile failed for problem '{problem_name}', test '{test_name}':\n"
                                f"--- STDOUT ---\n{result.get('compile_stdout', '').strip()}\n"
                                f"--- STDERR ---\n{result.get('compile_stderr', '').strip()}\n"
                            )
                        # Also surface runtime stderr if test failed (score 0) and stderr is non-empty
                        elif float(result['score']) == 0.0 and result.get('run_stderr'):
                            print(
                                f"Runtime error for problem '{problem_name}', test '{test_name}':\n"
                                f"--- STDOUT ---\n{result.get('run_stdout', '').strip()}\n"
                                f"--- STDERR ---\n{result.get('run_stderr', '').strip()}\n"
                            )

                # 4. Compute per-subtask scores and build `test_case_results`.
                for subtask, st in subtask_state.items():
                    if st['scores']:
                        effective_score = round(min(st['scores']) * st['score'], st['precision'])
                    else:
                        effective_score = 0.0
                    test_case_results[subtask] = {
                        "score": effective_score,
                        "outputs": st['outputs'],
                    }

                outputs.append(
                    {
                        "name": entry['name'],
                        "subtask": entry['subtask'],
                        "test_case_results": test_case_results,
                        "eval_status": {'complete': True, 'eval_error_log': []},
                    }
                )
            except Exception as e:
                error_msg = f"Error evaluating {entry.get('name', 'unknown')}: {str(e)}"
                print(error_msg)
                outputs.append(
                    {
                        "name": entry['name'],
                        "subtask": entry['subtask'],
                        "test_case_results": {},
                        "eval_status": {'complete': False, 'eval_error_log': [error_msg]},
                    }
                )

        for s, o in zip(samples, outputs):
            s['test_case_results'] = o['test_case_results']
            s['eval_status'] = o['eval_status']

        jdump(all_samples, jsonl_file, mode='wt')

        total_passed = 0
        total_problems = len(outputs)
        for o in outputs:
            for subtask_result in o["test_case_results"].values():
                if subtask_result["score"] > 0:
                    total_passed += 1
        print(f"Subtasks passed: {total_passed} out of {total_problems * len(metadata[o['name']])}")

    pool.close()
    pool.join()

    print(f"Total evaluation time: {(time.monotonic() - start_time)/60:.2f} minutes")

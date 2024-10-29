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


import copy
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from nemo_skills.code_execution import extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.server.model import BaseModel, OpenAIModel, get_model, models, trim_after_stop_phrases
from nemo_skills.utils import nested_dataclass, python_doc_to_cmd_help

LOG = logging.getLogger(__name__)


@nested_dataclass(kw_only=True)
class ErrorRecoveryConfig:
    # Number of attempts to recover from code execution error
    recovery_attempts: int = 0
    # If true, take code block based on majority voting of `recovery_attempts` code outputs.
    # Otherwise take the first valid code output.
    # So `majority_voting=False` is potentially faster.
    majority_voting: bool = True
    # Temperature for recovery requests
    temperature: float = 0.7
    # Top-p for recovery requests
    top_p: float = 0.95
    # Top-k for recovery requests
    top_k: int = 0


@nested_dataclass(kw_only=True)
class CodeExecutionConfig:
    max_code_output_characters: int = 1000
    code_execution_timeout: float = 10.0
    max_code_executions: int = 3
    stop_on_code_error: bool = False
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)


class CodeExecutionWrapper:
    def __init__(self, model: BaseModel, sandbox: Sandbox, config: CodeExecutionConfig):
        self.model = model
        self.sandbox = sandbox
        self.config = config

    def generate(
        self,
        prompts: list[str | dict],
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        # TODO: support properly prompt as dict of messages

        if stop_phrases is None:
            stop_phrases = []
        # making a copy of prompts to not corrupt original data
        new_prompts = copy.deepcopy(prompts)

        # prompts are added later
        request = {
            "prompts": new_prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_phrases": stop_phrases + [code_end],
            "remove_stop_phrases": False,  # we need to see where the model stopped
        }

        # making requests to LLM and iterating on prompts that produce code tokens
        # after executing code and getting result back into the prompt
        new_outputs = [
            {
                'prompt': new_prompts[idx],
                'execution_dict': None,
                'session_id': None,
            }
            for idx in range(len(new_prompts))
        ]
        remaining_ids = list(range(len(new_outputs)))
        num_executions = 0

        # Using 32 max executions at a time to not hit timeouts in a sandbox
        with ThreadPoolExecutor(max_workers=32) as executor:
            while len(remaining_ids) > 0:
                num_executions += 1
                request["prompts"] = [new_outputs[idx]['prompt'] for idx in remaining_ids]
                outputs = [self._handle_stop_words(output['generation']) for output in self.model.generate(**request)]
                new_ids = []
                # checking if any of the outputs need code execution and submitting requests in parallel
                futures = [None] * len(prompts)
                for idx, output in zip(remaining_ids, outputs):
                    # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
                    # that the last code_begin is not closed to ensure that we are inside the code block
                    if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                        futures[idx] = executor.submit(
                            self.sandbox.execute_code,
                            generated_code=extract_code_to_execute(output, code_begin, code_end),
                            timeout=self.config.code_execution_timeout,
                            max_output_characters=self.config.max_code_output_characters,
                            session_id=new_outputs[idx]['session_id'],
                        )
                for idx, output in zip(remaining_ids, outputs):
                    # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
                    # that the last code_begin is not closed to ensure that we are inside the code block
                    if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                        execution_dict, new_outputs[idx]['session_id'] = futures[idx].result()
                        if execution_dict['stderr']:
                            # TODO: error recovery should happen here that might change output
                            new_outputs[idx]['prompt'] += output
                        else:
                            new_outputs[idx]['prompt'] += output

                        # adding code output to the prompt
                        new_outputs[idx]['prompt'] += format_code_output(
                            execution_dict, code_output_begin, code_output_end, code_output_format
                        )
                        # setting a limit on max code executions to speed things up
                        # (sometimes keeps repeating the same sequence forever)
                        if num_executions >= self.config.max_code_executions:
                            new_outputs[idx]['prompt'] += "<max number of code executions reached>"
                        elif not (self.config.stop_on_code_error and execution_dict['stderr']):
                            new_ids.append(idx)
                    else:
                        # that's the only case where we might need to remove stop words, so doing it here
                        # we cannot do it at the end, since this needs to be done only on the last generation
                        output = trim_after_stop_phrases(output, stop_phrases)
                        new_outputs[idx]['prompt'] += output
                remaining_ids = new_ids

        # removing original prompt
        outputs = []
        for output, orig_prompt in zip(new_outputs, prompts):
            if output['session_id'] is not None:
                self.sandbox.clear_session(output['session_id'])
            outputs.append({'generation': output['prompt'][len(orig_prompt) :]})
        return outputs

    def _recover_from_error(self, request, new_output, executor):
        assert False, "this logic is currently broken, needs to be fixed"
        recovery_request = {key: value for key, value in request.items() if key != 'prompts'}
        recovery_request['prompts'] = [new_output['prompt']]

        recovery_request['temperature'] = self.config.error_recovery.temperature
        recovery_request['top_p'] = self.config.error_recovery.top_p
        recovery_request['top_k'] = self.config.error_recovery.top_k

        outputs = []
        futures = [None] * self.config.error_recovery.recovery_attempts
        execution_dicts = [None] * self.config.error_recovery.recovery_attempts
        for rs in range(self.config.error_recovery.recovery_attempts):
            recovery_request['random_seed'] = rs
            output = self._handle_stop_words(self.model.generate(**recovery_request)[0]['generation'])
            outputs.append(output)
            if output.strip().endswith(CODE_SEPARATORS[-1]):
                futures[rs] = executor.submit(
                    self.sandbox.execute_code,
                    generated_code=extract_code_to_execute(output),
                    timeout=self.config.code_execution_timeout,
                    max_output_characters=self.config.max_code_output_characters,
                    session_id=new_output['session_id'],
                )

                if not self.config.error_recovery.majority_voting:
                    execution_dict, _ = futures[rs].result()
                    # quit on first correct output if not majority voting
                    if not execution_dict['stderr']:
                        execution_dicts[rs] = execution_dict
                        break

        for idx, output in enumerate(outputs):
            if not output.strip().endswith(CODE_SEPARATORS[-1]) or not self.config.error_recovery.majority_voting:
                continue
            execution_dict, _ = futures[idx].result()
            if execution_dict['stderr']:
                continue
            execution_dicts[idx] = execution_dict

        # majority voting on valid code output results
        # if majority voting is disabled, we just take the first valid output
        counts = Counter(res for res in execution_dicts if res)
        # all errors
        if not counts:
            return

        most_common = counts.most_common(1)[0][0]
        valid_idx = execution_dicts.index(most_common)
        new_output['prompt'] += outputs[valid_idx]

        return most_common

    def _handle_stop_words(self, output: str):
        """
        OpenAI chat API remove stop word from the output, so we need to add it back
        to enable code execution.
        """
        if not isinstance(self.model, OpenAIModel):
            return output
        if output.find(CODE_SEPARATORS[0]) > output.find(CODE_SEPARATORS[-1]):
            return output + CODE_SEPARATORS[-1]
        return output


def server_params():
    """Returns server documentation (to include in cmd help)."""
    # TODO: This needs a fix now
    prefix = f'\n        server_type: str = MISSING - Choices: {list(models.keys())}'
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")


def get_code_execution_model(server_type, code_execution=None, sandbox=None, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model = get_model(server_type=server_type, **kwargs)
    if code_execution is None:
        code_execution = {}
    code_execution_config = CodeExecutionConfig(**code_execution)
    return CodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)

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


import abc
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import requests

from nemo_skills.code_execution import (
    CODE_OUTPUT_SEPARATORS,
    CODE_SEPARATORS,
    extract_code_to_execute,
    extract_error_message,
)
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.prompt.utils import Prompt
from nemo_skills.utils import python_doc_to_cmd_help

LOG = logging.getLogger(__name__)

# TODO: we should make this more efficient by asynchronously submitting eval
#       requests and getting new prompts in a queue to make sure we can always
#       pack full batches, instead of only resubmitting a small set of prompts
#       that require additional code executions


@dataclass
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


def remove_stop_tokens(text: str, stop_phrases: List[str]) -> str:
    return re.split("|".join(stop_phrases), text, maxsplit=1)[0]


class BaseModel(abc.ABC):
    """Base model class for handling requests to the inference server.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
        sandbox: Optional[Sandbox] = None - Sandbox for executing code.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through SSH_KEY_PATH env var.
        max_code_output_characters: Optional[int] = 1000 - Maximum number of characters for code execution output.
        code_execution_timeout: Optional[float] = 10.0 - Timeout for code execution in seconds.
        max_code_executions: Optional[int] = 3 - Maximum number of code executions per generation.
        stop_on_code_error: Optional[bool] = True - Whether to stop generation if code execution fails.
        handle_code_execution: Optional[bool] = True - Whether to handle code execution in this class
            or make a single call to the server. If set to False, the server needs to have special logic
            for communicating with the sandbox.
        error_recovery: Optional[dict] = None - Configuration for error recovery.
    """

    def __init__(
        self,
        host='127.0.0.1',
        port='5000',
        sandbox=None,
        ssh_server=None,
        ssh_key_path=None,
        max_code_output_characters=1000,
        code_execution_timeout=10.0,
        max_code_executions=3,
        stop_on_code_error=True,
        handle_code_execution=True,
        error_recovery=None,
    ):
        self.server_host = host
        self.server_port = port
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.max_code_output_characters = max_code_output_characters
        self.code_execution_timeout = code_execution_timeout
        self.max_code_executions = max_code_executions
        if error_recovery is None:
            error_recovery = {}
        self.error_recovery = ErrorRecoveryConfig(**error_recovery)
        self.handle_code_execution = handle_code_execution
        self.stop_on_code_error = stop_on_code_error
        if self.handle_code_execution and sandbox is None:
            raise ValueError("Sandbox is required for handling code execution")
        if not self.handle_code_execution and not stop_on_code_error:
            # TODO: same warning for other ignored parameters
            LOG.warning("When code execution is not handled here, stop_on_code_error is ignored.")
        self.sandbox = sandbox

    @abc.abstractmethod
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        pass

    def __call__(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        if self.handle_code_execution:
            full_stop_phrases = stop_phrases + [CODE_SEPARATORS[-1]]
        else:
            full_stop_phrases = stop_phrases
        # prompts are added later
        request = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_phrases": full_stop_phrases,
        }

        # if code execution is handled by the inference framework, we only need to make a single call
        # and then apply postprocessing to extract errors from the output
        if not self.handle_code_execution:
            request["prompts"] = prompts
            outputs = self._single_call(**request)
            outputs = [
                {
                    'generated_solution': remove_stop_tokens(output, stop_phrases),
                    'predicted_answer': extract_answer(output),
                    'error_message': extract_error_message(output),
                }
                for output in outputs
            ]
            return outputs

        # making requests to LLM and iterating on prompts that produce code tokens
        # after executing code and getting result back into the prompt
        new_outputs = [
            {
                'full_prompt': prompts[idx],
                'result': None,
                'error_message': Sandbox.NOT_EXECUTED,
                'session_id': None,
            }
            for idx in range(len(prompts))
        ]
        remaining_ids = list(range(len(new_outputs)))
        num_executions = 0
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            while len(remaining_ids) > 0:
                num_executions += 1
                request["prompts"] = [new_outputs[idx]['full_prompt'] for idx in remaining_ids]

                outputs = self._single_call(**request)
                new_ids = []
                # checking if any of the outputs need code execution and submitting requests in parallel
                futures = [None] * len(prompts)
                for idx, output in zip(remaining_ids, outputs):
                    if output.endswith(CODE_SEPARATORS[-1]):
                        futures[idx] = executor.submit(
                            self.sandbox.execute_code,
                            generated_code=extract_code_to_execute(output),
                            timeout=self.code_execution_timeout,
                            max_output_characters=self.max_code_output_characters,
                            session_id=new_outputs[idx]['session_id'],
                        )
                for idx, output in zip(remaining_ids, outputs):
                    if output.endswith(CODE_SEPARATORS[-1]):
                        result, new_outputs[idx]['session_id'] = futures[idx].result()
                        # for now if there is any error or no output, we stop generation
                        # might revise in the future to allow LLM to recover
                        if result['error_message']:
                            new_outputs[idx]['error_message'] = result['error_message']
                            if self.stop_on_code_error:
                                new_outputs[idx]['full_prompt'].generated_solution += output
                                continue
                            text_only_part = output.split(CODE_SEPARATORS[0])[0]
                            new_outputs[idx]['full_prompt'].generated_solution += text_only_part
                            code_output = self._recover_from_error(request, new_outputs[idx], executor)
                            # if re-generation did not help
                            if code_output is None:
                                code_output = result["result"]
                                new_outputs[idx]['full_prompt'].generated_solution += output[len(text_only_part) :]
                        else:
                            new_outputs[idx]['full_prompt'].generated_solution += output
                            new_outputs[idx]['error_message'] = ''
                            code_output = result["result"]

                        # adding code output to the prompt
                        code_output = f'\n{CODE_OUTPUT_SEPARATORS[0]}\n{code_output}\n{CODE_OUTPUT_SEPARATORS[1]}\n'
                        new_outputs[idx]['full_prompt'].generated_solution += code_output
                        # setting a limit on max code executions to speed things up
                        # (sometimes keeps repeating the same sequence forever)
                        if num_executions >= self.max_code_executions:
                            new_outputs[idx]['error_message'] = "Max code executions reached"
                        else:
                            new_ids.append(idx)
                    else:
                        new_outputs[idx]['full_prompt'].generated_solution += output
                remaining_ids = new_ids

        # removing original prompt and stop tokens from the end of the generated text
        outputs = []
        for output in new_outputs:
            if output['session_id'] is not None:
                self.sandbox.clear_session(output['session_id'])
            generated_solution = remove_stop_tokens(output['full_prompt'].generated_solution, stop_phrases)
            outputs.append(
                {
                    'generated_solution': generated_solution,
                    'predicted_answer': extract_answer(generated_solution),
                    'error_message': output['error_message'],
                }
            )
        return outputs

    def _recover_from_error(self, request, new_output, executor):
        recovery_request = {key: value for key, value in request.items() if key != 'prompts'}
        recovery_request['prompts'] = [new_output['full_prompt']]

        recovery_request['temperature'] = self.error_recovery.temperature
        recovery_request['top_p'] = self.error_recovery.top_p
        recovery_request['top_k'] = self.error_recovery.top_k

        outputs = []
        futures = [None] * self.error_recovery.recovery_attempts
        results = [None] * self.error_recovery.recovery_attempts
        for rs in range(self.error_recovery.recovery_attempts):
            recovery_request['random_seed'] = rs
            output = self._single_call(**recovery_request)[0]
            outputs.append(output)
            if output.endswith(CODE_SEPARATORS[-1]):
                futures[rs] = executor.submit(
                    self.sandbox.execute_code,
                    generated_code=extract_code_to_execute(output),
                    timeout=self.code_execution_timeout,
                    max_output_characters=self.max_code_output_characters,
                    session_id=new_output['session_id'],
                )

                if not self.error_recovery.majority_voting:
                    result, _ = futures[rs].result()
                    # quit on first correct output if not majority voting
                    if not result['error_message']:
                        results[rs] = result['result']
                        break

        for idx, output in enumerate(outputs):
            if not output.endswith(CODE_SEPARATORS[-1]) or not self.error_recovery.majority_voting:
                continue
            result, _ = futures[idx].result()
            if result['error_message']:
                continue
            results[idx] = result['result']

        # majority voting on valid code output results
        # if majority voting is disabled, we just take the first valid output
        counts = Counter(res for res in results if res)
        # all errors
        if not counts:
            return

        most_common = counts.most_common(1)[0][0]
        valid_idx = results.index(most_common)
        new_output['full_prompt'].generated_solution += outputs[valid_idx]
        new_output['error_message'] = ''

        return most_common

    def _send_request(self, request):
        # temperature of 0 means greedy, but it's not always supported by the server
        # so setting explicit greedy parameters instead
        if request["temperature"] == 0:
            request["temperature"] = 1.0
            request["top_k"] = 1
            request["top_p"] = 1.0

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            outputs = sshtunnel_request.put(
                url="http://{}:{}/generate".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        else:
            outputs = requests.put(
                url="http://{}:{}/generate".format(self.server_host, self.server_port),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()

        return outputs


class TensorRTLLMModel(BaseModel):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        string_prompts = [str(prompt) for prompt in prompts]
        request = {
            "prompts": string_prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_words_list": stop_phrases,
        }
        return self._send_request(request)


class NemoModel(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        # nemo inference handles code execution directly
        if 'handle_code_execution' not in kwargs:
            kwargs['handle_code_execution'] = False
        if not kwargs['handle_code_execution']:
            unsupported_arguments = [
                'max_code_output_characters',
                'code_execution_timeout',
                'max_code_executions',
                'error_recovery',
                'stop_on_code_error',
            ]
            for arg in unsupported_arguments:
                if arg in kwargs:
                    raise ValueError(
                        f"`{arg}` is not supported by NemoModel if handle_code_execution=False. To use it, set handle_code_execution=True."
                    )

        super().__init__(**kwargs)

    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        string_prompts = [str(prompt) for prompt in prompts]
        request = {
            "sentences": string_prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "end_strings": ["<|endoftext|>"] + stop_phrases,
        }
        outputs = self._send_request(request)
        outputs = outputs['sentences']
        # always returns full prompt, so we need to remove the original prompt
        for idx, output in enumerate(outputs):
            outputs[idx] = output[len(string_prompts[idx]) :]
        return outputs


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model,
        api_key=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import OpenAI

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
        top_k=0,  # is not supported by OpenAI
    ):
        if top_k != 0:
            raise ValueError("`top_k` is not supported by OpenAI, please set it to default value `0`.")

        responses = []
        for prompt in prompts:
            response = self._send_request(
                prompt=prompt,
                tokens_to_generate=tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                stop_phrases=stop_phrases,
            )
            responses.append(response)
        return responses

    def _send_request(
        self,
        prompt: Prompt,
        tokens_to_generate,
        temperature,
        top_p,
        repetition_penalty,
        random_seed,
        stop_phrases: List[str],
    ):
        messages = prompt.build_chat_prompt()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=tokens_to_generate,
            presence_penalty=repetition_penalty,
            seed=random_seed,
            stop=stop_phrases,
            messages=messages,
        ).choices[0]
        content = response.message.content

        # OpenAI removes stop tokens so we need to add them back
        if (
            response.finish_reason == "stop"
            and content.find(CODE_SEPARATORS[0]) != -1
            and not content[content.find(CODE_SEPARATORS[0]) :].count(CODE_SEPARATORS[-1])
        ):
            content += CODE_SEPARATORS[-1]

        return content


models = {
    'tensorrt_llm': TensorRTLLMModel,
    'nemo': NemoModel,
    'openai': OpenAIModel,
}


def get_model(server_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(**kwargs)


def server_params():
    """Returns server documentation (to include in cmd help)."""
    prefix = f'\n        server_type: str = MISSING - Choices: {list(models.keys())}'
    return python_doc_to_cmd_help(BaseModel, docs_prefix=prefix, arg_prefix="server.")

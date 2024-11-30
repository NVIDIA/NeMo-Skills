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
from concurrent.futures import ThreadPoolExecutor

from nemo_skills.code_execution import extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.server.model import BaseModel, get_model, models, trim_after_stop_phrases
from nemo_skills.utils import nested_dataclass, python_doc_to_cmd_help

LOG = logging.getLogger(__name__)


@nested_dataclass(kw_only=True)
class CodeExecutionConfig:
    max_code_output_characters: int = 1000
    code_execution_timeout: float = 10.0
    max_code_executions: int = 3


class CodeExecutionWrapper:
    def __init__(self, model: BaseModel, sandbox: Sandbox, config: CodeExecutionConfig):
        self.model = model
        self.sandbox = sandbox
        self.config = config

    def _generate_single(
        self,
        prompt: str | dict,
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str] | None = None,
    ):
        if not isinstance(prompt, str):
            raise NotImplementedError("OpenAI API is not supported yet.")

        if stop_phrases is None:
            stop_phrases = []
        # making a copy of prompts to not corrupt original data
        new_prompt = copy.deepcopy(prompt)

        request = {
            "prompt": new_prompt,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            "repetition_penalty": repetition_penalty,
            "stop_phrases": stop_phrases + [code_end],
        }
        session_id = None
        # adding plus one to make sure there is always some completion after the last requested code block
        for generation_index in range(self.config.max_code_executions + 1):
            output_dict = self.model._generate_single(**request)
            output, num_generated_tokens = output_dict['generation'], output_dict.get('num_generated_tokens', 0)
            request['prompt'] += output
            # if it's the extra iteration, we don't execute the code block and just finish
            if generation_index == self.config.max_code_executions:
                break
            # adjusting requested tokens to account for what has been generated already
            request['tokens_to_generate'] -= num_generated_tokens
            # TODO: currently we don't account for tokens in the code output that we add to the prompt
            #       in most cases the output should be small though
            if request['tokens_to_generate'] <= 0:
                break
            # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
            # that the last code_begin is not closed to ensure that we are inside the code block
            if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                execution_dict, session_id = self.sandbox.execute_code(
                    generated_code=extract_code_to_execute(output, code_begin, code_end),
                    timeout=self.config.code_execution_timeout,
                    max_output_characters=self.config.max_code_output_characters,
                    session_id=session_id,
                )
                # adding code output to the prompt
                request['prompt'] += format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format
                )
            else:  # if no code was generated, we need to finish
                break

        # removing original prompt
        return {'generation': request['prompt'][len(prompt) :]}

    # TODO: is there a way to reuse this with BaseModel?
    def generate(
        self,
        prompts: list[str | dict],
        code_begin: str | list[str],
        code_end: str | list[str],
        code_output_begin: str | list[str],
        code_output_end: str | list[str],
        code_output_format: str | list[str],
        tokens_to_generate: int | list[int] = 512,
        temperature: float | list[float] = 0.0,
        top_p: float | list[float] = 0.95,
        top_k: int | list[int] = 0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        # TODO: currently nemo server would get separate 1-batch requests, which is likely really inefficient
        #       but the alternative is to have a fully separate implementation, which is also not nice
        #       If we find ourselves needing to use nemo with code execution often, we should fix this
        kwargs = {
            'code_begin': code_begin,
            'code_end': code_end,
            'code_output_begin': code_output_begin,
            'code_output_end': code_output_end,
            'code_output_format': code_output_format,
            'tokens_to_generate': tokens_to_generate,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
        }
        for key, value in kwargs.items():
            is_list = False
            if key == 'stop_phrases' and (value and isinstance(value[0], list)):
                is_list = True
            if key != 'stop_phrases' and isinstance(value, list):
                is_list = True
            if is_list and len(value) != len(prompts):
                raise ValueError(f"Length of {key} should match the number of prompts.")
            if not is_list:
                kwargs[key] = [value for _ in range(len(prompts))]

        futures = []
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            for request_idx in range(len(prompts)):
                request = {key: value[request_idx] for key, value in kwargs.items()}
                request['prompt'] = prompts[request_idx]
                self.model.preprocess_request(request)
                futures.append(executor.submit(self._generate_single, **request))
        outputs = [future.result() for future in futures]

        if remove_stop_phrases:
            for output in outputs:
                output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

        return outputs


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

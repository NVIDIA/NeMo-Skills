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
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
from dataclasses import field

from nemo_skills.code_execution import extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.server.model import BaseModel, get_model, models, trim_after_stop_phrases
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
    # Number of tokens in a code block for recovery request
    tokens_to_generate: int = 256


@nested_dataclass(kw_only=True)
class CodeExecutionConfig:
    max_code_output_characters: int = 1000
    code_execution_timeout: float = 10.0
    max_code_executions: int = 3
    error_recovery: ErrorRecoveryConfig = field(default_factory=ErrorRecoveryConfig)


class CodeExecutionWrapper:
    def __init__(self, model: BaseModel, sandbox: Sandbox, config: CodeExecutionConfig):
        self.model = model
        self.sandbox = sandbox
        self.config = config

        self.gen_id_to_params = {}
        self.gen_id_to_future = {}

        self.executor = ThreadPoolExecutor(max_workers=1024)  # is this too much?

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
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        is_recovery: bool = False,
    ):
        if not isinstance(prompt, str):
            raise NotImplementedError("OpenAI API is not supported yet.")
        if top_logprobs is not None:  # TODO: add this
            raise NotImplementedError("top_logprobs is not supported yet.")

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
            "min_p": min_p,
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
            # if we are inside error recovery run, we want to return just the first code block
            if is_recovery:
                return {'generation': code_begin + output}
            # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
            # that the last code_begin is not closed to ensure that we are inside the code block
            if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                generated_code = extract_code_to_execute(output, code_begin, code_end)
                execution_dict, session_id = self.sandbox.execute_code(
                    generated_code=generated_code,
                    timeout=self.config.code_execution_timeout,
                    max_output_characters=self.config.max_code_output_characters,
                    session_id=session_id,
                )

                # Check for errors and attempt recovery if needed
                # Only attempt recovery if this is not already a recovery attempt
                if not is_recovery and self.is_code_error(execution_dict):
                    recovered_dict = self._recover_from_error_async(
                        request,
                        session_id,
                        code_begin,
                        code_end,
                        code_output_begin,
                        code_output_end,
                        code_output_format,
                    )
                    if recovered_dict:
                        recovered_code, execution_dict = recovered_dict
                        cur_prompt = request['prompt']
                        cur_prompt = cur_prompt[:cur_prompt.rfind(code_begin)] + f"{code_begin}{recovered_code}{code_end}"
                        request['prompt'] = cur_prompt
                # adding code output to the prompt
                request['prompt'] += format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format
                )
            else:  # if no code was generated, we need to finish
                break

        # removing original prompt
        return {'generation': request['prompt'][len(prompt) :]}

    def _recover_from_error_async(
        self,
        request: Dict[str, Any],
        session_id: Optional[str],
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover from code execution errors by running multiple attempts with different random seeds."""
        if not hasattr(self.config, 'error_recovery') or self.config.error_recovery.recovery_attempts == 0:
            return None

        # Create base recovery prompts
        current_prompt = request['prompt']
        last_code_begin_pos = current_prompt.rfind(code_begin)
        if last_code_begin_pos == -1:
            return None
            
        # Base prompt up to the last code block
        base_recovery_prompt = current_prompt[:last_code_begin_pos] + code_begin
        recovery_prompts = [base_recovery_prompt] * self.config.error_recovery.recovery_attempts
        
        # Set up recovery-specific parameters
        recovery_tokens_to_generate = self.config.error_recovery.tokens_to_generate
        recovery_temperature = self.config.error_recovery.temperature
        recovery_top_p = self.config.error_recovery.top_p
        recovery_top_k = self.config.error_recovery.top_k
        
        recovery_generations = self.generate(
            prompts=recovery_prompts,
            code_begin=code_begin,
            code_end=code_end,
            code_output_begin=code_output_begin,
            code_output_end=code_output_end,
            code_output_format=code_output_format,
            tokens_to_generate=recovery_tokens_to_generate,
            temperature=recovery_temperature,
            top_p=recovery_top_p,
            top_k=recovery_top_k,
            min_p=request.get('min_p', 0.0),
            repetition_penalty=request.get('repetition_penalty', 1.0),
            random_seed=list(range(self.config.error_recovery.recovery_attempts)),
            stop_phrases=request.get('stop_phrases', []),
            remove_stop_phrases=False,
            is_recovery=True,
        )
        
        # Save original session state if needed
        original_session_state = None
        if session_id is not None and session_id in self.sandbox.sessions:
            original_session_state = self.sandbox.sessions[session_id].copy()
        
        # Execute the generated code in parallel
        code_execution_futures = []
        with ThreadPoolExecutor(max_workers=len(recovery_generations)) as executor:
            for rs, gen_dict in enumerate(recovery_generations):
                output = gen_dict['generation']
                
                # Check if we got a complete code block
                if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                    generated_code = extract_code_to_execute(output, code_begin, code_end)
                    
                    # Create a temporary session ID for this recovery attempt
                    temp_session_id = None
                    if original_session_state is not None:
                        temp_session_id = f"{session_id}_recovery_{rs}"
                        self.sandbox.sessions[temp_session_id] = original_session_state.copy()
                        
                    future = executor.submit(
                        self.sandbox.execute_code,
                        generated_code=generated_code,
                        timeout=self.config.code_execution_timeout,
                        max_output_characters=self.config.max_code_output_characters,
                        session_id=temp_session_id,
                    )
                    code_execution_futures.append((generated_code, rs, temp_session_id, future))
        # If we don't have any valid code execution futures, return None
        if not code_execution_futures:
            return None
                
        # Process code execution results
        successful_executions = []
        
        # Collect successful executions
        for generated_code, rs, temp_session_id, future in code_execution_futures:
            try:
                execution_dict, _ = future.result()
                if not self.is_code_error(execution_dict):
                    successful_executions.append((generated_code, rs, temp_session_id, execution_dict))
                    # If not using majority voting, return the first successful execution
                    if not self.config.error_recovery.majority_voting:
                        self._apply_successful_recovery(
                            session_id, 
                            original_session_state,
                            generated_code, 
                            execution_dict,
                            code_execution_futures,
                            future
                        )
                        return generated_code, execution_dict
            except Exception as e:
                LOG.warning(f"Recovery attempt {rs} failed with exception: {str(e)}")
                continue
            finally:
                # Ensure we clean up this temporary session if there was an error
                if temp_session_id is not None and temp_session_id in self.sandbox.sessions:
                    del self.sandbox.sessions[temp_session_id]
        # If no successful executions, return None
        if not successful_executions:
            return None
        
        # If using majority voting, pick the most common successful result
        if self.config.error_recovery.majority_voting and len(successful_executions) > 1:
            # Count by stdout value to find the most common successful result
            counts = Counter(execution['stdout'] for _, _, _, execution in successful_executions)
            most_common_stdout = counts.most_common(1)[0][0]
            # Find the first execution dict with the most common stdout
            for generated_code, _, _, execution_dict in successful_executions:
                if execution_dict['stdout'] == most_common_stdout:
                    self._apply_successful_recovery(
                        session_id, 
                        original_session_state,
                        generated_code, 
                        execution_dict,
                        code_execution_futures
                    )
                    return generated_code, execution_dict
        
        # Otherwise use the first successful execution
        best_code, _, _, best_execution = successful_executions[0]
        
        self._apply_successful_recovery(
            session_id, 
            original_session_state,
            best_code, 
            best_execution,
            code_execution_futures
        )
        
        return best_code, best_execution
    
    def _apply_successful_recovery(
        self, 
        session_id: Optional[str], 
        original_session_state: Optional[list],
        successful_code: str, 
        execution_dict: Dict[str, Any],
        code_execution_futures: list,
        current_future=None
    ):
        """Apply a successful recovery solution by updating the session and cleaning up resources."""
        # Restore original session state and add the successful code
        if session_id is not None and original_session_state is not None:
            self.sandbox.sessions[session_id] = original_session_state.copy()
            self.sandbox.sessions[session_id].append(successful_code)
        
        # Cancel other futures to save resources if we have a reference to the current future
        if current_future is not None:
            for _, _, _, other_future in code_execution_futures:
                if other_future != current_future and not other_future.done():
                    other_future.cancel()
        
        # Clean up all temporary sessions
        self._cleanup_temp_sessions([s_id for _, _, s_id, _ in code_execution_futures])

    def _cleanup_temp_sessions(self, session_ids):
        """Clean up temporary sessions created for recovery attempts."""
        for session_id in session_ids:
            if session_id is not None and session_id in self.sandbox.sessions:
                del self.sandbox.sessions[session_id]

    def generate_async(
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
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
        top_logprobs: int | list[int] | None = None,
        is_recovery: bool = False,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        # TODO: currently nemo server would get separate 1-batch requests, which is likely really inefficient
        #       but the alternative is to have a fully separate implementation, which is also not nice
        #       If we find ourselves needing to use nemo with code execution often, we should fix this
        if top_logprobs is not None:  # TODO: add this
            raise NotImplementedError("top_logprobs is not supported yet.")
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
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases,
            'is_recovery': is_recovery,
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

        gen_ids = []
        for request_idx in range(len(prompts)):
            request = {key: value[request_idx] for key, value in kwargs.items()}
            request['prompt'] = prompts[request_idx]
            self.model.preprocess_request(request)
            future = self.executor.submit(self._generate_single, **request)
            gen_id = str(uuid.uuid4())
            self.gen_id_to_future[gen_id] = future
            self.gen_id_to_params[gen_id] = (request['stop_phrases'], remove_stop_phrases)
            gen_ids.append(gen_id)

        return gen_ids

    def get_generations(
        self,
        generation_ids: list[str],
    ) -> list[dict]:

        generations = []
        for generation_id in generation_ids:
            if generation_id not in self.gen_id_to_future:
                raise ValueError(f"Generation id {generation_id} not found.")

            stop_phrases, remove_stop_phrases = self.gen_id_to_params[generation_id]
            future = self.gen_id_to_future[generation_id]
            if not future.done():
                output = {'generation': None}
            else:
                output = future.result()
                del self.gen_id_to_future[generation_id]
                del self.gen_id_to_params[generation_id]

            if remove_stop_phrases:
                if output['generation'] is not None:
                    output['generation'] = trim_after_stop_phrases(output['generation'], stop_phrases)

            generations.append(output)

        return generations

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
        min_p: float | list[float] = 0.0,
        repetition_penalty: float | list[float] = 1.0,
        random_seed: int | list[int] = 0,
        stop_phrases: list[str] | list[list[str]] | None = None,
        remove_stop_phrases: bool = True,
        is_recovery: bool = False,
    ) -> list[dict]:
        """For any generation parameter you can specify a list of values that needs to match the number of prompts.

        Not every server supports that, so make sure to override this method directly if that's not the case.
        """
        generation_ids = self.generate_async(
            prompts=prompts,
            code_begin=code_begin,
            code_end=code_end,
            code_output_begin=code_output_begin,
            code_output_end=code_output_end,
            code_output_format=code_output_format,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            remove_stop_phrases=remove_stop_phrases,
            is_recovery=is_recovery,
        )
        all_generations = [None] * len(prompts)
        while True:
            remaining_ids = [generation_id for generation_id in generation_ids if generation_id is not None]
            if len(remaining_ids) == 0:
                break
            remaining_positions = [
                idx for idx, generation_id in enumerate(generation_ids) if generation_id is not None
            ]
            generations = self.get_generations(remaining_ids)
            for gen_pos, gen_dict in zip(remaining_positions, generations):
                if gen_dict['generation'] is not None:  # will be None until done
                    generation_ids[gen_pos] = None
                    all_generations[gen_pos] = gen_dict

            time.sleep(1)

        return all_generations
    
    def is_code_error(self, execution_dict: Dict[str, Any]):
        return execution_dict['stderr'] or "Traceback (most recent call last)" in execution_dict['stdout'] or "SyntaxError" in execution_dict['stdout']


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
    code_execution_config = CodeExecutionConfig(_init_nested=True, **code_execution)
    return CodeExecutionWrapper(model=model, sandbox=sandbox, config=code_execution_config)

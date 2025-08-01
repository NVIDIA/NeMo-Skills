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


import asyncio
import copy
import logging
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from nemo_skills.code_execution import extract_code_to_execute, format_code_output
from nemo_skills.code_execution.sandbox import Sandbox
from nemo_skills.inference.model.base import BaseModel
from nemo_skills.inference.model.utils import trim_after_stop_phrases
from nemo_skills.utils import get_logger_name, nested_dataclass

from .base import BaseModel
from .utils import trim_after_stop_phrases

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class CodeExecutionConfig:
    max_code_output_characters: int = 1000
    code_execution_timeout: float = 10.0
    code_execution_language: str = 'ipython'
    code_execution_headers: list[str] = field(default_factory=lambda: [])
    max_code_executions: int = 8
    sandbox_traceback_verbosity: str = 'plain'  # could be plain, context, verbose, or minimal
    add_remaining_code_executions: bool = False


class CodeExecutionWrapper:
    def __init__(self, model: BaseModel, sandbox: Sandbox, config: CodeExecutionConfig):
        self.model = model
        self.sandbox = sandbox
        self.config = config

        self.gen_id_to_params = {}
        self.gen_id_to_future = {}
        self.cancelled_gen_ids = set()  # Track cancelled generation IDs

        self.executor = ThreadPoolExecutor(max_workers=1024)  # is this too much?

        if hasattr(model, '_generate_single_async') and hasattr(model, 'cancel_generations'):
            self._can_cancel_generations = True
        else:
            self._can_cancel_generations = False

    def _is_generation_cancelled(self, gen_id):
        """Check if a generation has been requested to be cancelled."""
        return gen_id in self.cancelled_gen_ids

    def _generate_single(
        self,
        prompt: str | list,
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
        gen_id: str = None,  # used for cancelling requests if supported
        timeout: int | None = None,
        max_code_executions: int | None = None,  # if not None, will override self.config.max_code_executions
        stream: bool = False,
        extra_body: dict = None,
    ):
        # Handle OpenAI-style dictionary prompts
        is_openai_format = not isinstance(prompt, str)

        if top_logprobs is not None:  # TODO: add this
            raise NotImplementedError("top_logprobs is not supported yet.")

        if stream:
            return self._stream_single(
                prompt=prompt,
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
                timeout=timeout,
                max_code_executions=max_code_executions,
                extra_body=extra_body,
            )

        effective_max_code_executions = self.config.max_code_executions
        if max_code_executions is not None:
            effective_max_code_executions = max_code_executions

        # making a copy of prompts to not corrupt original data
        if is_openai_format:
            new_prompt = copy.deepcopy(prompt)
        else:
            new_prompt = copy.deepcopy(prompt)

        start_time = int(time.time())

        stop_phrases = stop_phrases or []

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
            "timeout": timeout,
            "extra_body": extra_body,
        }
        session_id = None
        code_rounds_executed = 0
        total_num_generated_tokens = 0
        generation_time = 0
        code_execution_time = 0
        stopped_on_repetition = False
        # adding plus one to make sure there is always some completion after the last requested code block
        for generation_index in range(effective_max_code_executions + 1):

            generation_time_start = time.time()
            if timeout is not None:
                # updating timeout to account for the time already spent
                new_timeout = int(timeout - (time.time() - start_time))
                request["timeout"] = new_timeout
                if request['timeout'] <= 0:
                    break

            # Check if generation has been cancelled before proceeding
            if gen_id is not None and self._is_generation_cancelled(gen_id):
                break

            if self._can_cancel_generations:
                # Wait for generation to finish while periodically checking for cancellation
                # TODO: clean up the interface to always use public method, not just in this case
                request["prompts"] = [request.pop("prompt")]
                async_gen_id = self.model.generate_async(**request, remove_stop_phrases=False)[0]
                while True:
                    time.sleep(0.5)
                    # Check periodically if generation should be cancelled
                    if gen_id is not None and self._is_generation_cancelled(gen_id):
                        self.model.cancel_generations([async_gen_id])
                        break

                    output_dict = self.model.get_generations([async_gen_id])[0]
                    if output_dict['generation'] is not None:
                        break

                if gen_id is not None and self._is_generation_cancelled(gen_id):
                    break
                request["prompt"] = request.pop("prompts")[0]
            else:
                output_dict = self.model._generate_single(**request)

            output, num_generated_tokens = output_dict['generation'], output_dict.get('num_generated_tokens', 0)
            # no need to do anything with this as the code below should just exit, so that's only for logging
            stopped_on_repetition = output_dict.get('stopped_on_repetition', False)

            # openai don't show what stop word was triggered, so we assume that it was `code_end`
            # if there's an unfinished code block
            if is_openai_format and output_dict.get('finish_reason') == 'stop':
                if output.count(code_end) + 1 == output.count(code_begin):
                    output += code_end
            # Update the prompt based on format
            if is_openai_format:
                request['prompt'].append({'role': 'assistant', 'content': output})
                request['prompt'].append({'role': 'user', 'content': "continue"})
            else:
                request['prompt'] += output

            # if it's the extra iteration, we don't execute the code block and just finish

            if generation_index == effective_max_code_executions:
                break
            # adjusting requested tokens to account for what has been generated already
            request['tokens_to_generate'] -= num_generated_tokens
            total_num_generated_tokens += num_generated_tokens
            generation_time += int(time.time() - generation_time_start)
            # TODO: currently we don't account for tokens in the code output that we add to the prompt
            #       in most cases the output should be small though
            if request['tokens_to_generate'] <= 0:
                break
            # .rfind(code_end, 0, -1) searches for the second-to-last occurrence of code_end and checks
            # that the last code_begin is not closed to ensure that we are inside the code block
            if output.endswith(code_end) and output.rfind(code_begin) > output.rfind(code_end, 0, -1):
                code_execution_time_start, execution_dict = self.execute_generated_code(
                    prompt, code_begin, code_end, output, session_id
                )
                remaining_code_executions = None
                if self.config.add_remaining_code_executions:
                    remaining_code_executions = effective_max_code_executions - generation_index - 1
                # adding code output to the prompt
                code_output = format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format, remaining_code_executions
                )

                if is_openai_format:
                    request['prompt'][-2]['content'] += code_output
                else:
                    request['prompt'] += code_output

                code_execution_time += int(time.time() - code_execution_time_start)
                code_rounds_executed += 1
            else:  # if no code was generated, we need to finish
                break

        # removing original prompt and returning the generation
        if is_openai_format:
            generation = "\n".join(msg['content'] for msg in request['prompt'] if msg['role'] == 'assistant')
        else:
            generation = request['prompt'][len(prompt) :]

        return {
            'generation': generation,
            'code_rounds_executed': code_rounds_executed,
            'num_generated_tokens': total_num_generated_tokens,
            'generation_time': generation_time,
            'code_execution_time': code_execution_time,
            'stopped_on_repetition': stopped_on_repetition,
        }

    def execute_generated_code(self, input_prompt, code_begin, code_end, output, session_id):
        code_execution_time_start = time.time()
        header = '\n'.join(self.config.code_execution_headers)
        code_block = extract_code_to_execute(output, code_begin, code_end)
        extracted_code = f'{header}{code_block}'
        execution_dict, session_id = self.sandbox.execute_code(
            generated_code=extracted_code,
            language=self.config.code_execution_language,
            timeout=self.config.code_execution_timeout,
            max_output_characters=self.config.max_code_output_characters,
            session_id=session_id,
            traceback_verbosity=self.config.sandbox_traceback_verbosity,
        )

        return code_execution_time_start, execution_dict

    # TODO: is there a way to reuse this with BaseModel?
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
        timeout: int | list[int] | None = None,
        max_code_executions: int | list[int] | None = None,
        stream: bool = False,
        extra_body: dict = None,
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
            "timeout": timeout,
            "max_code_executions": max_code_executions,
            "stream": stream,
            "extra_body": extra_body,
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
            gen_id = str(uuid.uuid4())
            # Pass the gen_id to _generate_single
            future = self.executor.submit(self._generate_single, gen_id=gen_id, **request)
            self.gen_id_to_future[gen_id] = future
            self.gen_id_to_params[gen_id] = (request['stop_phrases'], remove_stop_phrases)
            gen_ids.append(gen_id)

        return gen_ids

    def cancel_generations(self, generation_ids: list[str]) -> list[str]:
        if not self._can_cancel_generations:
            raise NotImplementedError("This model does not support cancelling generations.")

        statuses = []
        for generation_id in generation_ids:
            if generation_id not in self.gen_id_to_future:
                raise ValueError(f"Generation id {generation_id} not found.")

            # Mark this generation as cancelled - it will be actually cancelled in the generate_single loop
            self.cancelled_gen_ids.add(generation_id)
            statuses.append("canceled")

            # TODO: more checks?

        return statuses

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
                output = {
                    'generation': None,
                    'code_rounds_executed': None,
                    'num_generated_tokens': None,
                    'generation_time': None,
                    'code_execution_time': None,
                    'stopped_on_repetition': None,
                }
            else:
                output = future.result()
                del self.gen_id_to_future[generation_id]
                del self.gen_id_to_params[generation_id]

            if remove_stop_phrases:
                if isinstance(output, dict) and output['generation'] is not None:
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
        top_logprobs: int | list[int] | None = None,
        timeout: int | list[int] | None = None,
        max_code_executions: int | list[int] | None = None,
        stream: bool = False,
        extra_body: dict = None,
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
            top_logprobs=top_logprobs,
            timeout=timeout,
            max_code_executions=max_code_executions,
            stream=stream,
            extra_body=extra_body,
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
                if isinstance(gen_dict, Generator) or gen_dict['generation'] is not None:  # will be None until done
                    generation_ids[gen_pos] = None
                    all_generations[gen_pos] = gen_dict

            time.sleep(1)

        return all_generations

    async def generate_asyncio(self, *args, **kwargs) -> dict:
        result = await asyncio.to_thread(self.generate, *args, **kwargs)
        assert len(result) == 1, "generate_asyncio should return a single result"
        return result[0]

    def _stream_single(
        self,
        prompt: str,
        code_begin: str,
        code_end: str,
        code_output_begin: str,
        code_output_end: str,
        code_output_format: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        max_code_executions: int | None = None,
        extra_body: dict = None,
    ):
        """
        Helper method, that implements streaming generation.
        """
        # Handle OpenAI-style dictionary prompts
        is_openai_format = not isinstance(prompt, str)

        effective_max_code_executions = self.config.max_code_executions
        if max_code_executions is not None:
            effective_max_code_executions = max_code_executions

        stop_phrases = stop_phrases or []

        request = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'random_seed': random_seed,
            'stop_phrases': stop_phrases + [code_end],
            'timeout': timeout,
            'tokens_to_generate': tokens_to_generate,
            'stream': True,
            'extra_body': extra_body,
        }

        current_full_prompt = copy.deepcopy(prompt)
        session_id = None  # For sandbox state continuity
        for generation_index in range(effective_max_code_executions + 1):
            model_token_iterator = self.model._generate_single(prompt=current_full_prompt, **request)

            current_output_segment = ""
            num_generated_tokens = 0
            for chunk in model_token_iterator:
                yield chunk
                current_output_segment += chunk['generation']
                num_generated_tokens += 1

            request['tokens_to_generate'] -= num_generated_tokens
            if request['tokens_to_generate'] <= 0:
                break
            if not current_output_segment:
                break

            # openai don't show what stop word was triggered, so we assume that it was `code_end`
            # if there's an unfinished code block
            if is_openai_format and chunk.get('finish_reason') == 'stop':
                if current_output_segment.count(code_end) + 1 == current_output_segment.count(code_begin):
                    current_output_segment += code_end

            # Update the prompt based on format
            if is_openai_format:
                current_full_prompt.append({'role': 'assistant', 'content': current_output_segment})
                current_full_prompt.append({'role': 'user', 'content': "continue"})
            else:
                current_full_prompt += current_output_segment

            if generation_index == effective_max_code_executions:
                # This was the last iteration, intended for final text generation after all code executions.
                break

            if current_output_segment.endswith(code_end) and current_output_segment.rfind(
                code_begin
            ) > current_output_segment.rfind(code_end, 0, -1):
                execution_dict, session_id = self.sandbox.execute_code(
                    generated_code=extract_code_to_execute(current_output_segment, code_begin, code_end),
                    language=self.config.code_execution_language,
                    timeout=self.config.code_execution_timeout,
                    max_output_characters=self.config.max_code_output_characters,
                    session_id=session_id,
                    traceback_verbosity=self.config.sandbox_traceback_verbosity,
                )

                remaining_code_executions = None
                if self.config.add_remaining_code_executions:
                    remaining_code_executions = effective_max_code_executions - generation_index - 1

                formatted_code_output = format_code_output(
                    execution_dict, code_output_begin, code_output_end, code_output_format, remaining_code_executions
                )

                yield {'generation': formatted_code_output}  # Yield the entire formatted code output as one chunk

                # Append executed code's output to the prompt
                if is_openai_format:
                    current_full_prompt[-2]['content'] += formatted_code_output
                else:
                    current_full_prompt += formatted_code_output
            else:
                break

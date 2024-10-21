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

from typing import Dict, List

import torch
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy

from nemo_skills.code_execution.sandbox import get_sandbox


class CodeExecutionStrategy(GPTModelTextGenerationStrategy):
    def __init__(
        self, sandbox_cfg: Dict, timeout=10.0, max_code_output_characters=1000, stop_on_code_error=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.sandbox = get_sandbox(**sandbox_cfg)
        self.execution_state = None
        self.timeout = timeout
        self.max_code_output_characters = max_code_output_characters
        self.stop_on_code_error = stop_on_code_error

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the single step inference, post process the inference results
        Args:
            tokens  (torch.Tensor): the context tokens
            new_token (torch.Tensor): sampled new token id
            context_length (int): the new token position in the tokens
        """
        from nemo_skills.code_execution import (
            CODE_OUTPUT_SEPARATORS,
            CODE_SEPARATORS,
            extract_code_to_execute,
            format_code_output,
        )

        if self.execution_state is None:
            self.execution_state = [
                {
                    'session_id': None,
                    'execution_output': None,
                    'output_len': None,  # need this to broadcast from multiple ranks
                    'output_token_id': None,
                    'should_stop': torch.tensor(False, dtype=torch.bool, device='cuda'),
                }
                for _ in range(len(tokens))
            ]
        for idx, elem_tokens in enumerate(tokens):
            text_end = self.model.tokenizer.ids_to_text(elem_tokens[context_length - 20 : context_length + 1].tolist())
            if text_end.endswith(CODE_SEPARATORS[-1]) and not self.execution_state[idx]['should_stop']:
                generated_text = self.model.tokenizer.ids_to_text(elem_tokens[: context_length + 1].tolist())
                code_to_execute = extract_code_to_execute(generated_text)
                if torch.distributed.get_rank() == parallel_state.get_tensor_model_parallel_src_rank():
                    execution_dict, self.execution_state[idx]['session_id'] = self.sandbox.execute_code(
                        code_to_execute,
                        timeout=self.timeout,
                        max_output_characters=self.max_code_output_characters,
                        session_id=self.execution_state[idx]['session_id'],
                    )

                    if self.stop_on_code_error and execution_dict['stderr']:
                        self.execution_state[idx]['should_stop'] = torch.tensor(True, dtype=torch.bool, device='cuda')
                    else:
                        self.execution_state[idx]['should_stop'] = torch.tensor(False, dtype=torch.bool, device='cuda')

                    # adding [1:] to skip the first " " token that's somehow always added
                    self.execution_state[idx]['execution_output'] = torch.tensor(
                        self.model.tokenizer.text_to_ids(format_code_output(execution_dict))[1:],
                        dtype=torch.int32,
                        device='cuda',
                    )
                    self.execution_state[idx]['output_len'] = torch.tensor(
                        len(self.execution_state[idx]['execution_output']),
                        dtype=torch.int32,
                        device='cuda',
                    )
                else:
                    self.execution_state[idx]['should_stop'] = torch.tensor(False, dtype=torch.bool, device='cuda')
                    self.execution_state[idx]['output_len'] = torch.tensor(0, dtype=torch.int32, device='cuda')

                torch.distributed.broadcast(
                    self.execution_state[idx]['output_len'],
                    src=parallel_state.get_tensor_model_parallel_src_rank(),
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                torch.distributed.broadcast(
                    self.execution_state[idx]['should_stop'],
                    src=parallel_state.get_tensor_model_parallel_src_rank(),
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                # broadcasting the execution output as well
                if (
                    torch.distributed.get_rank() != parallel_state.get_tensor_model_parallel_src_rank()
                ):  # creating an empty tensor to receive the broadcast
                    self.execution_state[idx]['execution_output'] = torch.empty(
                        self.execution_state[idx]['output_len'],
                        dtype=torch.int32,
                        device='cuda',
                    )
                torch.distributed.broadcast(
                    self.execution_state[idx]['execution_output'],
                    src=parallel_state.get_tensor_model_parallel_src_rank(),
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                self.execution_state[idx]['output_token_id'] = 0

            # if we have any execution output that we didn't fully include yet, doing so (including the current one)
            output = self.execution_state[idx]['execution_output']
            if output is not None:
                if self.execution_state[idx]['output_token_id'] < len(output):
                    new_tokens[idx] = output[self.execution_state[idx]['output_token_id']]
                    self.execution_state[idx]['output_token_id'] += 1
                else:
                    self.execution_state[idx]['execution_output'] = None
                    self.execution_state[idx]['output_token_id'] = None
                    self.execution_state[idx]['output_len'] = None

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        is_done = super().end_of_generation_condition(tokens, prev, eod_id, end_strings)
        # stopping generation for and code errors that are fully written in the output already
        for idx in range(len(tokens)):
            if (
                self.execution_state[idx]['should_stop'].item()
                and self.execution_state[idx]['execution_output'] is None
            ):
                is_done[idx] = True
        return is_done

    def post_generation_process(self, output):
        # cleaning up all sessions and results
        if self.execution_state is None:  # if pipeline parallel is used
            return output
        for exec_state in self.execution_state:
            if (
                torch.distributed.get_rank() == parallel_state.get_tensor_model_parallel_src_rank()
                and exec_state['session_id'] is not None
            ):
                self.sandbox.clear_session(exec_state['session_id'])
        self.execution_state = None
        return output

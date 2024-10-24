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
import json
import logging
import re
import sys
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from mpi4py import MPI
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp
from transformers import AutoTokenizer, T5Tokenizer


# keeping it here to make this file self-contained. This is duplicated from model.py
def trim_after_stop_phrases(text: str, stop_phrases: List[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


class CustomSentencePieceTokenizer(T5Tokenizer):
    """
    Adapted from https://github.com/NVIDIA/Megatron-LM/blob/db3a3f79d1cda60ea4b3db0ceffcf20c5760e11d/examples/inference/trtllm_text_generation.py
    """

    def __init__(self, model):
        super().__init__(model, extra_ids=0, bos_token="<s>", pad_token="<pad>")

    def encode(self, text, add_special_tokens: bool = True, **kwargs):
        return torch.Tensor(self.sp_model.encode_as_ids(text))

    def batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens: bool = True, **kwargs):
        return {'input_ids': self.sp_model.encode_as_ids(batch_text_or_text_pairs)}

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **kwargs):
        if isinstance(sequences, np.ndarray) or torch.is_tensor(sequences):
            sequences = sequences.tolist()
        return self.sp_model.decode(sequences)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        return self.sp_model.decode([token_ids])[0]


class TrtStartGeneration(Resource):
    def __init__(self, model):
        self.model = model
        self.comm = MPI.COMM_WORLD

    def start_generation(
        self,
        prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        return self.model.start_generation(
            prompt,
            max_output_token=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_words_list=stop_words_list,
        )

    def put(self):
        logging.debug("generate async request")
        logging.debug("request IP: %s", str(request.remote_addr))
        input_request = request.get_json()
        logging.debug("request content: %s", json.dumps(input_request))

        top_k = input_request.get("top_k")
        if top_k == 0:
            top_k = None
        data = dict(
            prompt=input_request["prompt"],
            max_new_tokens=input_request.get("tokens_to_generate", 64),
            temperature=input_request.get("temperature", 1.0),
            top_k=top_k,
            top_p=input_request.get("top_p", 1.0),
            repetition_penalty=input_request.get("repetition_penalty", 1.2),
            random_seed=input_request.get("random_seed", 0),
            stop_words_list=input_request.get("stop_words_list"),
        )
        self.comm.Barrier()
        data = self.comm.bcast(data, root=0)

        out = self.start_generation(**data)
        return jsonify(out)


class TrtGetResult(Resource):
    def __init__(self, model):
        self.model = model

    def get_result(self, idx):
        return self.model.get_result(idx)

    def put(self):
        logging.debug("get result request")
        logging.debug("request IP: %s", str(request.remote_addr))
        input_request = request.get_json()
        logging.debug("request content: %s", json.dumps(input_request))
        return jsonify(self.get_result(input_request['generation_id']))


def parse_input(input_texts: str, tokenizer):
    batch_input_ids = [
        tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )
        for input_text in input_texts
    ]
    batch_input_ids = [torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids]
    input_lengths = [x.size(1) for x in batch_input_ids]

    return batch_input_ids, input_lengths


def get_output_single(output_ids, input_length, max_output_len, tokenizer, eos_token):
    output_begin = input_length
    output_end = input_length + max_output_len
    outputs = output_ids[output_begin:output_end]
    eos_ids = (outputs == eos_token).nonzero(as_tuple=True)[-1]
    if len(eos_ids) > 0:
        outputs = outputs[: eos_ids[0]]
    outputs = outputs.tolist()
    return tokenizer.decode(outputs)


def get_output(output_ids, input_lengths, max_output_len, tokenizer, eos_token):
    num_beams = output_ids.size(1)
    assert num_beams == 1
    output_texts = []
    for idx, input_len in enumerate(input_lengths):
        output_texts.append(get_output_single(output_ids[idx, 0], input_len, max_output_len, tokenizer, eos_token))
    return output_texts


def load_tokenizer(tokenizer_dir: str, model_name: str):
    if model_name == 'gpt-next':
        tokenizer = CustomSentencePieceTokenizer(str(Path(tokenizer_dir) / 'tokenizer.model'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


def read_model_name(config):
    name = config['pretrained_config']['architecture'].lower()
    name_map = {
        'GPTForCausalLM'.lower(): 'gpt-next',
    }
    return name_map.get(name, None)


def generate(
    runner,
    batch_input_ids: List[torch.Tensor],
    *,
    encoder_input_ids: List[torch.Tensor] = None,
    sampling_config=None,
    lora_uids: Optional[list] = None,
    streaming: bool = False,
    stopping_criteria=None,
    logits_processor=None,
    max_new_tokens: int = 1,
    end_id: int | None = None,
    pad_id: int | None = None,
    bad_words_list: list[list[int]] | None = None,
    stop_words_list: list[list[int]] | None = None,
    return_dict: bool = False,
    output_sequence_lengths: bool = False,
    output_log_probs: bool = False,
    output_cum_log_probs: bool = False,
    prompt_table=None,
    prompt_tasks: Optional[str] = None,
    return_all_generated_tokens: bool = False,
    input_lengths=None,
    tokenizer=None,
    **kwargs,
):
    """
    Generates sequences of token ids.
    The generation-controlling parameters are set in the sampling_config; it will be set to a default one if not passed.
    You can override any sampling_config's attributes by passing corresponding parameters.

    Args:
        batch_input_ids (List[torch.Tensor]):
            A list of input id tensors. Each tensor is of shape (sequence_length, ).
        sampling_config (SamplingConfig):
            The sampling configuration to be used as base parametrization for the generation call.
            The passed **kwargs matching the sampling_config's attributes will override them.
            If the sampling_config is not provided, a default will be used.
        prompt_table (str or torch.Tensor):
            The file path of prompt table (.npy format, exported by nemo_prompt_convert.py) or the prompt table itself.
        prompt_tasks (str):
            The prompt tuning task ids for the input batch, in format of comma-separated list (e.g., 0,3,1,0).
        lora_uids (list):
            The uids of LoRA weights for the input batch. Use -1 to disable the LoRA module.
        streaming (bool):
            Whether or not to use streaming mode for generation.
        stopping_criteria (StoppingCriteria):
            Custom stopping criteria.
        logits_processor (LogitsProcessor):
            Custom logits processors.
        return_all_generated_tokens (bool):
            Whether the full output is returned at each streaming step
        kwargs (Dict[str, Any]:
            Ad hoc parametrization of sampling_config.
            The passed **kwargs matching the sampling_config's attributes will override them.
    Returns:
        torch.Tensor or dict:
            If return_dict=False, the method returns generated output_ids.
            If return_dict=True, the method returns a dict of output_ids,
            sequence_lengths (if sampling_config.output_sequence_lengths=True),
            context_logits and generation_logits (if self.gather_context_logits=True and
            self.gather_generation_logits=True, respectively).
    """
    assert streaming
    # TODO: Check if these can be supported now and support them
    if lora_uids is not None:
        raise RuntimeError("LoRA is not supported in C++ session.")
    if stopping_criteria is not None:
        raise RuntimeError("Stopping criteria is not supported in C++ session.")
    if logits_processor is not None:
        raise RuntimeError("Logits processor is not supported in C++ session.")

    # If we are in a multi-gpu scenario, only rank 0 continues
    if not runner.session.can_enqueue_requests():
        return []

    # Convert tensor input to plain lists
    batch_input_ids_list = [a.tolist() for a in batch_input_ids]
    encoder_input_ids_list = [a.tolist() for a in encoder_input_ids] if encoder_input_ids else None

    if sampling_config is None:
        # Convert from old API of SamplingConfig
        # Note: Due to a Python3.10 bug one cannot use inspect on it currently
        accepted_parameters = [
            "num_beams",
            "top_k",
            "top_p",
            "top_p_min",
            "top_p_reset_ids",
            "top_p_decay",
            "random_seed",
            "temperature",
            "min_length",
            "beam_search_diversity_rate",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
        ]
        rename_params = {"num_beams": "beam_width", "random_seed": "seed"}
        sampling_params = {k: v for k, v in kwargs.items() if k in accepted_parameters}
        for k, v in rename_params.items():
            if k in sampling_params:
                sampling_params[v] = sampling_params.pop(k)
        if "top_p" in sampling_params and sampling_params["top_p"] == 0.0:
            sampling_params["top_p"] = None

        sampling_config = trtllm.SamplingConfig(**sampling_params)
    else:
        sampling_config = copy.deepcopy(sampling_config)

    runner._check_inputs(
        encoder_input_ids_list if encoder_input_ids else batch_input_ids_list, sampling_config, max_new_tokens
    )

    output_config = trtllm.OutputConfig(
        return_context_logits=runner.gather_context_logits,
        return_generation_logits=runner.gather_generation_logits,
        return_log_probs=output_log_probs,
    )

    prompt_tuning_configs = runner._prepare_ptuning_executor(batch_input_ids_list, prompt_table, prompt_tasks)

    # not letting trtllm handle stop words as this is only supported on a token-level
    stop_words_list_none = runner._prepare_words_list(None, len(batch_input_ids_list))
    bad_words_list = runner._prepare_words_list(bad_words_list, len(batch_input_ids_list))

    requests = [
        trtllm.Request(
            input_token_ids=input_ids,
            encoder_input_token_ids=encoder_input_ids_list[i] if encoder_input_ids is not None else None,
            max_tokens=max_new_tokens,
            pad_id=pad_id,
            end_id=end_id,
            stop_words=stop_words,
            bad_words=bad_words,
            sampling_config=sampling_config,
            streaming=streaming,
            output_config=output_config,
            prompt_tuning_config=prompt_tuning_config,
            return_all_generated_tokens=return_all_generated_tokens,
        )
        for i, (input_ids, stop_words, bad_words, prompt_tuning_config) in enumerate(
            zip(batch_input_ids_list, stop_words_list_none, bad_words_list, prompt_tuning_configs)
        )
    ]

    request_ids = runner.session.enqueue_requests(requests)

    return _stream(
        runner,
        request_ids,
        end_id,
        return_dict,
        output_sequence_lengths,
        output_log_probs,
        output_cum_log_probs,
        batch_input_ids,
        streaming,
        batch_input_ids_list,
        return_all_generated_tokens,
        stop_words_list,
        tokenizer,
        input_lengths,
        max_new_tokens=max_new_tokens,
    )


def _stream(
    runner,
    request_ids,
    end_id,
    return_dict,
    output_sequence_lengths,
    output_log_probs,
    output_cum_log_probs,
    batch_input_ids,
    streaming,
    batch_input_ids_list,
    return_all_generated_tokens,
    stop_words_list,
    tokenizer,
    input_lengths,
    max_new_tokens: int,
    num_return_sequences: int = 1,
):
    if stop_words_list is None:
        stop_words_list = []
    assert len(request_ids) == 1
    output_ids = [[] for _ in range(len(request_ids))]
    for reqid_pos in range(len(request_ids)):
        output_ids[reqid_pos] = [copy.deepcopy(batch_input_ids_list[reqid_pos]) for _ in range(runner.max_beam_width)]

    # checking the last 20 tokens for stop words
    num_tokens_to_check = 20

    idx = 0
    finished_reqs = 0
    while finished_reqs < len(request_ids):
        multi_responses = runner.session.await_responses(request_ids)
        responses = [response for responses in multi_responses for response in responses]
        for response in responses:
            if response.result.is_final:
                finished_reqs += 1

        output = runner._fill_output(
            responses,
            output_ids,
            end_id,
            return_dict,
            output_sequence_lengths,
            output_log_probs,
            output_cum_log_probs,
            batch_input_ids,
            batch_input_ids_list,
            streaming,
            request_ids,
            return_all_generated_tokens,
            max_new_tokens,
            num_return_sequences,
        )

        matching_stop_word = None
        # checking every half of the required tokens to have overlapping checks
        if idx < num_tokens_to_check - 1 or idx % (num_tokens_to_check // 2) != 0:
            idx += 1
            continue

        seq_length = output['sequence_lengths']
        generation_suffix = output['output_ids'][0, 0, seq_length[0] - num_tokens_to_check : seq_length[0]]
        out_string = get_output_single(generation_suffix, 0, num_tokens_to_check, tokenizer, end_id)
        for stop_word in stop_words_list:
            if stop_word in out_string:
                matching_stop_word = stop_word
                break

        if matching_stop_word is not None:
            runner.session.cancel_request(request_ids[0])
            break
        idx += 1

    out_string = get_output(output['output_ids'], input_lengths, output['sequence_lengths'][0], tokenizer, end_id)[0]
    for stop_word in stop_words_list:
        if stop_word in out_string:
            matching_stop_word = stop_word
            break
    if matching_stop_word is not None:
        out_string = trim_after_stop_phrases(out_string, stop_words_list)
        # adding it back, since we only need to remove what's *after* the stop phrase
        out_string += matching_stop_word
    else:
        # trtllm removes end id if it was the stop reason
        # this is a hack to add it back, but we are going to include it even when
        # it was not generated by the model e.g. if we stopped due to max tokens
        out_string += tokenizer.decode(end_id)
    return out_string


class TensorRTLLM:
    def __init__(self, model_path: str):
        with open(Path(model_path) / "config.json", 'r') as f:
            config = json.load(f)
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=model_path, model_name=read_model_name(config)
        )
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=model_path,
            rank=tensorrt_llm.mpi_rank(),
            enable_chunked_context=True,
            kv_cache_enable_block_reuse=True,
        )
        # might need to adjust in the future
        self.executor = ThreadPoolExecutor(max_workers=1024)
        self.requests = {}  # id to future

    def get_output(
        self,
        batch_input_ids,
        input_lengths,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        # TODO: return dictionary with a proper error reporting

        try:
            output = generate(
                self.runner,
                batch_input_ids[0],
                max_new_tokens=max_output_token,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                # stop words in trtllm are supported on the token-level only and this representation is not unique
                # so instead of passing in all tokenizations (is that even possible?) of each phrase, we will
                # instead stream outputs and detokenize them to check for stop words - this is done inside
                # overriden generate/stream functions above
                tokenizer=self.tokenizer,
                stop_words_list=stop_words_list,
                input_lengths=input_lengths,
                return_dict=True,
                output_sequence_lengths=True,
                streaming=True,
            )
        except RuntimeError as e:
            logging.error("RuntimeError: %s", e)
            output = f"RuntimeError: {e}"

        return output

    def get_result(self, idx):
        if self.requests[idx].done():
            result = self.requests.pop(idx).result()
            return result
        return None

    @torch.no_grad()
    def start_generation(
        self,
        input_text,
        max_output_token,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
    ):
        # TODO: remove batch dimension since it's not needed anymore?
        idx = str(uuid.uuid4())
        batch_input_ids, input_lengths = parse_input([input_text], self.tokenizer)
        self.requests[idx] = self.executor.submit(
            self.get_output,
            batch_input_ids,
            input_lengths,
            max_output_token,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            random_seed,
            stop_words_list,
        )

        return idx


class WrapperServer:
    def __init__(self, model_path: str):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.model = TensorRTLLM(model_path=model_path)

        if self.rank == 0:
            self.app = Flask(__file__, static_url_path="")
            api = Api(self.app)
            api.add_resource(TrtStartGeneration, "/start_generation", resource_class_args=[self.model])
            api.add_resource(TrtGetResult, "/get_result", resource_class_args=[self.model])

    def run(self, url, port=5000):
        if self.rank == 0:
            self.app.run(url, threaded=True, port=port, debug=False)
        else:
            self.worker_loop()

    def worker_loop(self):
        server = TrtStartGeneration(self.model)
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            server.start_generation(**data)


if __name__ == "__main__":

    class LogFilter(logging.Filter):
        def filter(self, record):
            filter_strings = ("\"PUT /get_result HTTP/1.1\" 200", "\"PUT /start_generation HTTP/1.1\" 200")
            return all(filter_string not in record.getMessage() for filter_string in filter_strings)

    log = logging.getLogger('werkzeug')
    log.addFilter(LogFilter())
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    server = WrapperServer(model_path=args.model_path)
    server.run(args.host, args.port)

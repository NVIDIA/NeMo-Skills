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
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import httpx
import openai
import requests
from openai import BadRequestError, DefaultHttpxClient, OpenAI

LOG = logging.getLogger(__file__)


class BaseModel(abc.ABC):
    """Base model class for handling requests to the reward model inference server.

    Args:
        model_type: Reward model type
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        model_type: str,
        host: str = '127.0.0.1',
        port: str = '5000',
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
    ):
        self.model_type = model_type
        self.server_host = host
        self.server_port = port
        self.ssh_server = ssh_server
        self.ssh_key_path = ssh_key_path
        if ssh_server is None:
            self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
        if ssh_key_path is None:
            self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            self.requests_lib = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
        else:
            self.requests_lib = requests

    def score(self, prompts: list[str]) -> list[dict]:
        pass


class RequestException(RuntimeError):
    pass


class NemoRewardModel(BaseModel):
    def score(self, prompts: list[str]) -> list[float]:
        request = {
            "prompts": prompts,
        }
        response = self.requests_lib.post(f"http://{self.server_host}:{self.server_port}/score", json=request)

        if response.status_code != 200:
            raise RequestException(f"Failed to score prompts: {response.text}")

        scores = response.json()

        outputs = [{"reward_model_score": score} for score in scores["rewards"]]
        return outputs


class VLLMRewardModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ssh_server and self.ssh_key_path:
            raise NotImplementedError("SSH tunnelling is not implemented for vLLM model.")

        http_client = DefaultHttpxClient(
            limits=httpx.Limits(max_keepalive_connections=1500, max_connections=1500),
            transport=httpx.HTTPTransport(retries=3),
        )

        self.oai_client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"http://{self.server_host}:{self.server_port}/v1",
            timeout=None,
            http_client=http_client,
        )

        # Reward models are accessed via the "pooling" interface
        # https://docs.vllm.ai/en/latest/models/pooling_models.html
        self.request_url = f"http://{self.server_host}:{self.server_port}/pooling"

        model_list = self.oai_client.models.list()
        self.model = model_list.data[0].id

    def _score_single_prompt(self, prompt):
        """Score a single prompt"""

        per_token_scores = None
        try:
            response = requests.post(self.request_url, json={"input": prompt, "model": self.model})
            output = response.json()
            per_token_scores = output['data'][0]['data']
        except requests.exceptions.HTTPError as err:
            print(f"Request failed: {err}")
        except ValueError as ve:
            # Could be that the sequence exceeds the maximum context length
            print(f"Tokenization error: {ve}")
        except KeyError as ke:
            # Returned output is not adhering to the expected output format
            print(f"Output fmt error: {ke}")
            print(output)

        if per_token_scores is None:
            # Return a trivial reward model score
            return {"reward_model_score": 0.0}

        last_token_score = per_token_scores[-1]
        score = None
        if self.model_type == "orm":
            # Last token's score
            if isinstance(last_token_score, list):
                logit_score = last_token_score[0]
            else:
                logit_score = last_token_score
            # Normalize the score
            score = 1 / (1 + math.exp(-logit_score))
        elif self.model_type == "prm":
            # Last token's score, a 2-entry array where the second entry is the probability of being correct
            score = last_token_score[1]

        return {"reward_model_score": score}

    def score(self, prompts: list[str]) -> list[float]:
        outputs = [None] * len(prompts)  # Pre-allocate a list to store results in correct order
        futures = {}

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            for idx, prompt in enumerate(prompts):
                futures[executor.submit(self._score_single_prompt, prompt)] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except BadRequestError as e:
                    error_details = e.body
                    error_message = error_details.get("message", "No message found")
                    error_code = error_details.get("code", "No code found")
                    if error_code == 400 and 'maximum context length' in error_message:
                        outputs[idx] = {
                            "reward_model_score": 0
                        }  # Default value set as 0 if we have request over maximum context length
                        LOG.warning("Maximum context length exceeded, setting reward score as 0")
                    else:
                        raise
        return outputs


models = {
    'nemo': NemoRewardModel,
    'vllm': VLLMRewardModel,
}


def get_reward_model(server_type, model_type, **kwargs):
    """A helper function to make it easier to set server through cmd."""
    model_class = models[server_type.lower()]
    return model_class(model_type=model_type, **kwargs)

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

# adapted from https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py

import datetime
import logging
import os

import torch
from megatron.core import parallel_state
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer

# explicitly adding a strategy to avoid failing imports
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.collections.nlp.modules.common.text_generation_utils import generate
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

"""
This is the script to run GPT text generation.

Usage:
         python megatron_gpt_eval.py \
            gpt_model_file=PATH_TO_MODEL \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            server=True

        To send a request to the server, here is one example code:
        ```python
        import json
        import requests

        batch_size = 8
        port_num = 5555
        headers = {"Content-Type": "application/json"}


        def request_data(data):
            resp = requests.put('http://localhost:{}/generate'.format(port_num),
                                data=json.dumps(data),
                                headers=headers)
            sentences = resp.json()['sentences']
            return sentences


        data = {
            "sentences": [""] * batch_size,
            "tokens_to_generate": 300,
            "temperature": 1.0,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }

        sentences = request_data(data)
        ```
"""

if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


@hydra_runner(config_path=".", config_name="nemo_inference")
def main(cfg) -> None:
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=[CustomProgressBar()],
    )

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.gpt_model_file):
        save_restore_connector.model_extracted_dir = cfg.gpt_model_file

    pretrained_cfg = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        return_config=True,
        save_restore_connector=save_restore_connector,
    )
    OmegaConf.set_struct(pretrained_cfg, True)
    with open_dict(pretrained_cfg):
        pretrained_cfg.sequence_parallel = False
        pretrained_cfg.activations_checkpoint_granularity = None
        pretrained_cfg.activations_checkpoint_method = None
        pretrained_cfg.precision = trainer.precision
        pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        if trainer.precision == "16":
            pretrained_cfg.megatron_amp_O2 = False
        elif trainer.precision in ['bf16', 'bf16-mixed'] and cfg.get('megatron_amp_O2', False):
            pretrained_cfg.megatron_amp_O2 = True
            
        if 'attn_scaling_during_inference_type' in cfg:
            pretrained_cfg.attn_scaling_during_inference_type = cfg.attn_scaling_during_inference_type
            logging.info(f"setting attn_scaling_during_inference_type to {pretrained_cfg.attn_scaling_during_inference_type}")

        if 'attn_scaling_during_inference_value' in cfg:
            pretrained_cfg.attn_scaling_during_inference_value = cfg.attn_scaling_during_inference_value
            logging.info(f"setting attn_scaling_during_inference_value to {pretrained_cfg.attn_scaling_during_inference_value}")
            
    model = MegatronGPTModel.restore_from(
        restore_path=cfg.gpt_model_file,
        trainer=trainer,
        override_config_path=pretrained_cfg,
        save_restore_connector=save_restore_connector,
        map_location=f'cuda:{trainer.local_rank}',  # map_location is needed for converted models
    )
    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    inference_strategy = GPTModelTextGenerationStrategy(model=model)

    if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model.cuda(), inference_strategy=inference_strategy)
        server.run("0.0.0.0", port=cfg.port)

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            generate(model.cuda(), strategy=inference_strategy)


if __name__ == '__main__':
    # setting nemo logging to warning as there is too much info otherwise
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)
    main()  # noqa pylint: disable=no-value-for-parameter

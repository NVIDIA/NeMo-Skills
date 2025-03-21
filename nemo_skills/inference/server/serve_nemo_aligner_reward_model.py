# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# copied from: https://github.com/NVIDIA/NeMo-Aligner/blob/main/examples/nlp/gpt/serve_reward_model.py

import torch

try:
    from lightning.pytorch.trainer.trainer import Trainer
except ModuleNotFoundError:
    from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.core.config import hydra_runner
from nemo_aligner.algorithms.reward_server import RewardModelServer
from nemo_aligner.models.nlp.gpt.reward_model_classes import REWARD_MODEL_CLASS_DICT, RewardModelType
from nemo_aligner.utils.text_generation_utils import tokenize_batch
from nemo_aligner.utils.train_script_utils import init_distributed
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, set_autocast_gpu_dtype

"""PyTriton Based Inference Server for the Reward Model"""

ENDPOINT_BIND_ADDRESS = "0.0.0.0"


@hydra_runner(config_path=".", config_name="nemo_inference_rm")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.rm_model_file, cfg.model)
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    # needed for autocasting BF16
    set_autocast_gpu_dtype(cfg.trainer.precision)
    if trainer.precision == "16":
        cfg.model.megatron_amp_O2 = False
    elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
        cfg.model.megatron_amp_O2 = True

    reward_model_type = RewardModelType(cfg.model.get("reward_model_type", "binary_ranking"))
    reward_model_cls = REWARD_MODEL_CLASS_DICT[reward_model_type]

    ptl_model = load_from_nemo(
        reward_model_cls,
        cfg.model,
        trainer,
        strict=True,
        restore_path=cfg.rm_model_file,
    )
    ptl_model.freeze()

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    ptl_model = ptl_model.cuda()
    ptl_model.prepare_for_inference()

    def tokenize_func(sentences):
        return tokenize_batch(
            sentences=sentences,
            tokenizer=ptl_model.tokenizer,
            max_len=ptl_model.cfg.encoder_seq_length,
            add_BOS=False,
            add_EOS=False,
        )

    inference_cfg = cfg.inference

    server = RewardModelServer(
        infer_fn=ptl_model.infer,
        tokenize_func=tokenize_func,
        model_name=inference_cfg.get("model_name", "reward_model"),
        port=inference_cfg.get("port", 5001),
        inference_micro_batch_size=inference_cfg.get("inference_micro_batch_size", 2),
        model_forward_micro_batch_size=cfg.model.get("forward_micro_batch_size", cfg.model.micro_batch_size),
        strip_sequence_length_to_multiple=inference_cfg.get("strip_sequence_length_to_multiple", None),
        max_queue_delay_microseconds=inference_cfg.get("max_queue_delay_microseconds", 2000),
    )
    server.run_server()


if __name__ == "__main__":
    with torch.no_grad():
        main()

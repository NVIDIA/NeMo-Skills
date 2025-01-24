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

import os

import nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset as gpt_sft_chat_dataset
import torch.multiprocessing as mp
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import get_prompt_template_example
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.nlp.builders import build_dataloader, build_sft_dataset
from nemo_aligner.models.nlp.gpt.gpt_sft_model import GPTSFTModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo
from omegaconf.omegaconf import OmegaConf, open_dict

"""Script to start SFT training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path=".", config_name="sft_config")
def main(cfg) -> None:
    cfg.model = load_and_override_model_config(cfg.model.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # updating a few parameters based on num_checkpoints_to_save arg
    if cfg.trainer.sft.get("num_checkpoints_to_save", None) is not None:
        # if steps are > 0 using that
        if cfg.trainer.sft.max_steps > 0:
            num_steps = cfg.trainer.sft.max_steps
        else:
            # counting the steps per epoch
            # using wc -l since sft file might be large and we want to use optimized util
            data_size = int(os.popen(f'wc -l "{cfg.model.data.train_ds.file_path}"').read().split()[0])
            assert cfg.trainer.sft.max_epochs > 0
            num_steps = (data_size * cfg.trainer.sft.max_epochs) // cfg.model.data.train_ds.global_batch_size
        num_checkpoints = cfg.trainer.sft.num_checkpoints_to_save
        with open_dict(cfg):
            cfg.trainer.sft.max_epochs = 10000  # always using steps internally
            # rounding steps to make sure last checkpoint is not repeated
            cfg.trainer.sft.max_steps = (num_steps // num_checkpoints) * num_checkpoints
            cfg.trainer.sft.save_interval = num_steps // num_checkpoints
            cfg.trainer.sft.val_check_interval = num_steps // num_checkpoints
        logging.info(
            (
                "Adjusting config parameters in the following way:\n"
                "max_epochs: %d\nmax_steps: %d\nsave_interval: %d\nval_check_interval: %d"
            ),
            cfg.trainer.sft.max_epochs,
            cfg.trainer.sft.max_steps,
            cfg.trainer.sft.save_interval,
            cfg.trainer.sft.val_check_interval,
        )

    trainer = resolve_and_create_trainer(cfg, "sft")
    exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    ptl_model = load_from_nemo(
        GPTSFTModel,
        cfg,
        trainer,
        strict=True,
        restore_path=cfg.model.restore_from_path,
        return_updated_cfg=False,
    )

    init_peft(ptl_model, cfg.model)

    with open_dict(cfg):
        # overwrite the model config with the config from the checkpoint
        cfg.model.encoder_seq_length = ptl_model.cfg.encoder_seq_length

    # monkey-patching the system token to allow training with llama format
    # TODO: remove when this is properly supported in nemo
    if cfg.model.data.chat:
        # not using default to avoid accidental errors by mistyping or misplacing this value in the config
        gpt_sft_chat_dataset.SYSTEM_TOKEN = cfg.model.data["system_token"]

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path

    # TODO: log this restore path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_data_cfg = cfg.model.data.train_ds
    val_data_cfg = cfg.model.data.validation_ds

    if cfg.model.data.get("sample", False):
        # if it is negative, num_samples is None
        if cfg.trainer.sft.max_steps < 0:
            num_samples = None
        else:
            num_samples = cfg.trainer.sft.max_steps * train_data_cfg.global_batch_size
    else:
        num_samples = None
    train_ds = build_sft_dataset(
        train_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )
    if cfg.model.data.get("sample", False):
        num_samples = cfg.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
    else:
        num_samples = None
    validation_ds = build_sft_dataset(
        val_data_cfg,
        ptl_model.tokenizer,
        num_samples,
        answer_only_loss=True,
        is_chat=cfg.model.data.chat,
        special_tokens=cfg.model.data.chat_prompt_tokens,
    )

    train_dataloader = build_dataloader(
        cfg=cfg,
        dataset=train_ds,
        consumed_samples=consumed_samples,
        mbs=train_data_cfg.micro_batch_size,
        gbs=train_data_cfg.global_batch_size,
        collate_fn=train_ds.collate_fn,
        drop_last=train_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not train_data_cfg.drop_last,
        load_gbs=True,
    )

    val_dataloader = build_dataloader(
        cfg=cfg,
        dataset=validation_ds,
        consumed_samples=0,
        mbs=val_data_cfg.micro_batch_size,
        gbs=val_data_cfg.global_batch_size,
        collate_fn=validation_ds.collate_fn,
        drop_last=val_data_cfg.drop_last,
        pad_samples_to_global_batch_size=not val_data_cfg.drop_last,
        load_gbs=True,
        use_random_sampler=False,
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_ds)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))
    timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)

    sft_trainer = SupervisedTrainer(
        cfg=cfg.trainer.sft,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    if custom_trainer_state_dict is not None:
        sft_trainer.load_state_dict(custom_trainer_state_dict)

    sft_trainer.fit()


if __name__ == "__main__":
    main()

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from nemo_skills.inference.chat_interface.core import AppConfig
from nemo_skills.inference.chat_interface.chat_service import AppContext
from nemo_skills.inference.chat_interface.ui import ChatUI
from nemo_skills.utils import setup_logging


cs = ConfigStore.instance()
cs.store(name="base_chat_interface_config", node=AppConfig)


@hydra.main(version_base=None, config_name="base_chat_interface_config")
def launch(cfg: AppConfig):
    setup_logging(disable_hydra_logs=True)
    logging.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    cfg_obj = AppConfig(**cfg)

    ctx = AppContext(cfg_obj)
    ui = ChatUI(ctx)

    app = ui.launch()
    app.queue().launch(server_port=7885)


if __name__ == "__main__":
    launch()

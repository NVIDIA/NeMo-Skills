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

import argparse
import logging

from nemo_skills.inference.chat_interface.core import AppConfig
from nemo_skills.inference.chat_interface.chat_service import AppContext
from nemo_skills.inference.chat_interface.ui import ChatUI


def launch():
    parser = argparse.ArgumentParser(description="NeMo Skills Chat Interface (refactored)")
    parser.add_argument("--mode", choices=["manual", "direct"], default="manual")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--server_type", default="vllm")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO)

    cfg = AppConfig(host=args.host, server_type=args.server_type, launch_mode=args.mode)
    ctx = AppContext(cfg)
    ui = ChatUI(ctx)

    app = ui.launch()
    app.queue().launch()


if __name__ == "__main__":
    launch() 
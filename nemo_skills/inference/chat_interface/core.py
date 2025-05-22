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
import time
import requests
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple, Dict

from nemo_skills.inference.server.model import get_model
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.prompt.utils import get_prompt, Prompt
from openai import APIConnectionError
from requests.exceptions import ConnectionError

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AppConfig:
    """Immutable application-wide configuration."""

    # Server / backend
    host: str = "localhost"
    server_type: str = "vllm"  # "vllm", "sglang", "openai"
    ssh_server: str | None = None
    ssh_key_path: str | None = None

    # Prompt configuration
    base_prompt_config: str = "generic/math"
    code_prompt_config: str = "openmath/tir"
    prompt_template: str = "openmath-instruct"

    # Code-execution related
    initial_code_execution_state: bool = False
    max_code_executions: int = 8
    add_remaining_code_executions: bool = True

    # Model capabilities: "cot", "tir", "both" (toggleable)
    supported_modes: str = "both"

class CodeExecStatus(Enum):
    """High-level availability of the Python code-execution toolchain."""

    NOT_REQUESTED = auto()
    DISABLED = auto()  # requested but unavailable (either model or sandbox down)
    ENABLED = auto()


# -----------------------------------------------------------------------------
# Prompt handling
# -----------------------------------------------------------------------------


class PromptManager:
    """Loads and caches prompt."""

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._cache: Dict[str, Prompt] = {}

    def get(self, use_code: bool) -> Prompt:
        """Initialize and cache the requested prompt object."""
        path = self._cfg.code_prompt_config if use_code else self._cfg.base_prompt_config
        if path in self._cache:
            return self._cache[path]

        logger.debug("Loading prompt config: %s", path)
        prompt = get_prompt(prompt_config=path, prompt_template=self._cfg.prompt_template)
        self._cache[path] = prompt
        return prompt


# -----------------------------------------------------------------------------
# Model + sandbox loader
# -----------------------------------------------------------------------------


class ModelLoader:
    """Responsible for fetching / caching model handles and sandbox."""

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._generic_llm: Any | None = None
        self._code_llm: Any | None = None
        self._sandbox: Any | None = None

    @property
    def generic_llm(self) -> Any | None:  # noqa: D401
        return self._generic_llm

    @property
    def code_llm(self) -> Any | None:  # noqa: D401
        return self._code_llm

    @property
    def sandbox(self):  # noqa: D401
        return self._sandbox

    @property
    def cfg(self):  # noqa: D401
        return self._cfg


    def load_generic(self) -> Tuple[bool, str]:
        """Attempt to load the generic/inference-only model."""
        if self._generic_llm:
            return True, "already-loaded"

        logger.info("Loading GENERIC LLM (%s, host=%s)…", self._cfg.server_type, self._cfg.host)
        try:
            self._generic_llm = get_model(
                server_type=self._cfg.server_type,
                host=self._cfg.host,
                ssh_server=self._cfg.ssh_server,
                ssh_key_path=self._cfg.ssh_key_path,
            )
            logger.info("Generic model loaded.")
            return True, "success"
        except (ConnectionError, APIConnectionError) as e:
            logger.error("Connection error loading generic model: %s", e)
            return False, str(e)
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error loading generic model")
            return False, str(e)

    def load_code_and_sandbox(self) -> Tuple[bool, str]:
        """Attempt to load code-exec model and sandbox."""
        if self._code_llm and self._sandbox:
            return True, "already-loaded"

        # Always (re)create sandbox first so we can attach it to the model.
        try:
            self._sandbox = get_sandbox(
                host=self._cfg.host,
                ssh_server=self._cfg.ssh_server,
                ssh_key_path=self._cfg.ssh_key_path,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to create sandbox client")
            self._sandbox = None
            return False, f"sandbox error: {e}"

        code_params = {
            "max_code_executions": self._cfg.max_code_executions,
            "add_remaining_code_executions": self._cfg.add_remaining_code_executions,
        }

        try:
            self._code_llm = get_code_execution_model(
                server_type=self._cfg.server_type,
                host=self._cfg.host,
                ssh_server=self._cfg.ssh_server,
                ssh_key_path=self._cfg.ssh_key_path,
                sandbox=self._sandbox,
                code_execution=code_params,
            )
            logger.info("Code-execution model loaded.")
            return True, "success"
        except (ConnectionError, APIConnectionError) as e:
            logger.warning("Connection error loading code model: %s", e)
            return False, str(e)
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error loading code model")
            return False, str(e)

    def get_code_execution_status(self, requested: bool) -> CodeExecStatus:
        """Return the CodeExecStatus based on requested flag and availability."""
        if not requested:
            return CodeExecStatus.NOT_REQUESTED

        if self._code_llm and self._sandbox and self._is_sandbox_alive():
            return CodeExecStatus.ENABLED

        return CodeExecStatus.DISABLED

    def _is_sandbox_alive(self) -> bool:
        if not self._sandbox or not hasattr(self._sandbox, "_get_execute_url"):
            return False

        try:
            resp = self._sandbox.execute_code("1")
            return int(resp[0]["stdout"].strip()) == 1
        except requests.RequestException as e:  # noqa: BLE001
            logger.warning("Sandbox health check failed: %s", e)
            return False

    def supports_code_toggle(self) -> bool:
        """Return True if the backend advertises support for both execution modes."""
        return self._cfg.supported_modes == "both"

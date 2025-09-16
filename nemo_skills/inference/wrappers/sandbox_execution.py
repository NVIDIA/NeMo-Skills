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

"""Wrapper that executes code from generation using sandbox."""

import re
from typing import Any, Dict

from nemo_skills.inference.model.base import BaseModel
from nemo_skills.inference.model.wrapper import ContextAwareModel, ContextAwareWrapper


class SandboxExecutionWrapper(ContextAwareWrapper):
    """Wrapper that executes code from generation using the GenerationTask's sandbox."""

    def default_config(self) -> Dict[str, Any]:
        return {
            "extract_code_blocks": True,
            "code_block_pattern": r"```python\n(.*?)\n```",
            "timeout_override": None,  # Can override sandbox timeout
        }

    def configure(self, overrides=None, context=None):
        super().configure(overrides, context)

        # Get sandbox from context (passed from GenerationTask)
        self.sandbox = self.context.get("sandbox") if self.context else None

        if self.sandbox is None:
            raise ValueError("SandboxExecutionWrapper requires sandbox in context")

    def wrap(self, model: BaseModel) -> BaseModel:
        return SandboxExecutionModel(model, self.config, self.sandbox)


class SandboxExecutionModel(ContextAwareModel):
    """Model that executes code from generation in sandbox."""

    def __init__(self, model: BaseModel, config: dict, sandbox):
        super().__init__(model, config)
        self.sandbox = sandbox

    async def post_process(self, result, data_point, all_data):
        """Extract and execute code from generation."""
        if self.config["extract_code_blocks"]:
            code = self._extract_code(result["generation"])
            if code:
                try:
                    # Use the sandbox from GenerationTask
                    execution_result = await self.sandbox.execute(code)
                    result["sandbox_execution"] = {"code": code, "result": execution_result, "success": True}
                except Exception as e:
                    result["sandbox_execution"] = {"code": code, "error": str(e), "success": False}
        return result

    def _extract_code(self, generation: str) -> str | None:
        """Extract code blocks from generation text."""
        pattern = self.config["code_block_pattern"]
        code_blocks = re.findall(pattern, generation, re.DOTALL)
        return "\n".join(code_blocks) if code_blocks else None

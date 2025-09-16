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

"""Model wrapper system for composable generation enhancements.

This provides a lightweight plugin system for model wrappers that can be supplied as
Python modules/classes, similar to the MCP tool system. Each wrapper module exposes
a concrete class implementing the ModelWrapper interface. The WrapperManager imports
and instantiates wrappers, applies per-wrapper overrides, and chains them together.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict

from nemo_skills.inference.model.base import BaseModel
from nemo_skills.mcp.utils import locate


class ModelWrapper(ABC):
    """Abstract base for module-based model wrappers.

    Conventions:
    - default_config() returns default configuration dict
    - configure() applies configuration overrides and context
    - wrap() wraps a model and returns the wrapped version
    """

    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Return default configuration for this wrapper."""
        pass

    @abstractmethod
    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        """Configure the wrapper with overrides and context."""
        pass

    @abstractmethod
    def wrap(self, model: BaseModel) -> BaseModel:
        """Wrap the given model and return the wrapped version."""
        pass


class WrapperManager:
    """Registry/Manager for module-based model wrappers.

    - Loads wrapper classes from module specs using nemo_skills.mcp.utils.locate.
    - Applies per-wrapper overrides based on the wrapper class name.
    - Applies wrappers in sequence to create a composed model.
    """

    def __init__(
        self,
        module_specs: list[str],
        overrides: Dict[str, Dict[str, Any]] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        self._wrappers: list[ModelWrapper] = []  # Preserve order for chaining

        overrides = overrides or {}
        context = context or {}

        for spec in module_specs or []:
            wrapper_cls_or_obj = locate(spec)
            wrapper_cls = wrapper_cls_or_obj
            if not inspect.isclass(wrapper_cls_or_obj):
                # Allow passing an already-instantiated object
                wrapper_cls = wrapper_cls_or_obj.__class__
            wrapper: ModelWrapper = wrapper_cls() if inspect.isclass(wrapper_cls_or_obj) else wrapper_cls_or_obj

            wrapper_key = wrapper.__class__.__name__

            wrapper.configure(overrides.get(wrapper_key), context)
            self._wrappers.append(wrapper)

    def wrap_model(self, model: BaseModel) -> BaseModel:
        """Apply all wrappers to the model in sequence."""
        wrapped_model = model
        for wrapper in self._wrappers:
            wrapped_model = wrapper.wrap(wrapped_model)
        return wrapped_model


class ContextAwareModel:
    """Model that handles context parameters and delegates to base model."""

    def __init__(self, model: BaseModel, config: dict):
        self.model = model
        self.config = config
        # Delegate all attributes to the wrapped model
        self._delegate_attributes()

    def _delegate_attributes(self):
        """Delegate common model attributes to the wrapped model."""
        # Copy important attributes from the wrapped model
        for attr in ["model_name_or_path", "tokenizer", "MODEL_PROVIDER"]:
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))

    async def generate_async(self, prompt, data_point=None, all_data=None, **kwargs):
        """Generate with context awareness."""
        # Store context for potential use in post-processing
        self.current_data_point = data_point
        self.current_all_data = all_data

        # Call base model without context params (for backward compatibility)
        result = await self.model.generate_async(prompt, **kwargs)

        # Allow post-processing with context
        return await self.post_process(result, data_point, all_data)

    async def post_process(self, result, data_point, all_data):
        """Override in subclasses to add post-processing logic."""
        return result

    def __getattr__(self, name):
        """Delegate any missing attributes to the wrapped model."""
        return getattr(self.model, name)


class ContextAwareWrapper(ModelWrapper):
    """Base wrapper that provides context-aware model interface."""

    def default_config(self):
        return {}

    def configure(self, overrides=None, context=None):
        self.config = {**self.default_config(), **(overrides or {})}
        self.context = context or {}

    def wrap(self, model: BaseModel) -> BaseModel:
        return ContextAwareModel(model, self.config)

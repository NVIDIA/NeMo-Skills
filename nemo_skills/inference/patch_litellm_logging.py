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

"""
Patch for litellm.litellm_core_utils.logging_worker.LoggingWorker
to disable its functionality and make all methods no-op.
"""

from typing import Coroutine


class NoOpLoggingWorker:
    """No-op implementation of LoggingWorker that disables all functionality."""

    def __init__(self, *args, **kwargs):
        """Initialize with no-op."""
        pass

    def _ensure_queue(self) -> None:
        """No-op queue initialization."""
        pass

    def start(self) -> None:
        """No-op start."""
        pass

    async def _worker_loop(self) -> None:
        """No-op worker loop."""
        pass

    def enqueue(self, coroutine: Coroutine) -> None:
        """No-op enqueue - drops all logging tasks."""
        if coroutine is not None:
            coroutine.close()

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine):
        """No-op ensure and enqueue."""
        if async_coroutine is not None:
            async_coroutine.close()

    async def stop(self) -> None:
        """No-op stop."""
        pass

    async def flush(self) -> None:
        """No-op flush."""
        pass

    async def clear_queue(self):
        """No-op clear queue."""
        pass


def patch_litellm_logging_worker():
    """
    Patches the litellm LoggingWorker to disable its functionality.
    This prevents any logging worker from keeping the server alive.
    """
    try:
        # Import the module
        import litellm.litellm_core_utils.logging_worker as logging_worker_module

        # Replace the LoggingWorker class with our no-op version
        logging_worker_module.LoggingWorker = NoOpLoggingWorker

        # Replace the global instance with a no-op instance
        logging_worker_module.GLOBAL_LOGGING_WORKER = NoOpLoggingWorker()
    except ModuleNotFoundError:
        # Ensure compatibility with different litellm versions
        pass

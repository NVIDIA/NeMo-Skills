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

import json
import logging
from dataclasses import asdict, field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import (
    GenerateSolutionsConfig,
    GenerationTask,
    InferenceConfig,
)
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MultiChallengeGenerationConfig(GenerateSolutionsConfig):
    """MultiChallenge benchmark generation."""

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    attempts: int = 1  # Number of attempts to generate for each conversation

    # Override defaults to use openai format (no prompt_config needed)
    prompt_format: str = "openai"
    prompt_config: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_multi_challenge_generation_config", node=MultiChallengeGenerationConfig)


class MultiChallengeGenerationTask(GenerationTask):
    """Generation task for MultiChallenge benchmark."""

    def __init__(self, cfg: MultiChallengeGenerationConfig):
        super().__init__(cfg)
        self.attempts = cfg.attempts

    def log_example_prompt(self, data):
        """MultiChallenge is a conversational benchmark."""
        LOG.info("Example conversation:")
        if data and len(data) > 0:
            conversation = data[0].get("conversation", [])
            for turn in conversation[:2]:  # Show first 2 turns
                LOG.info(f"  {turn['role']}: {turn['content'][:100]}...")

    def setup_prompt(self):
        """No special prompt setup needed for MultiChallenge."""
        return None

    def preprocess_data(self, data):
        """Preprocess data to wrap conversations in messages key for openai format."""
        for data_point in data:
            # Wrap conversation in messages key for openai format
            if "conversation" in data_point and "messages" not in data_point:
                data_point["messages"] = data_point["conversation"]
        return data

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single data point and generate responses.

        Args:
            data_point: Dictionary containing:
                - messages: List of message dicts with 'role' and 'content'
                - question_id, axis, target_question, pass_criteria
            all_data: Full dataset (not used in MultiChallenge)

        Returns:
            Dictionary with generated responses
        """
        from dataclasses import asdict as dc_asdict, is_dataclass

        # Handle inference config - check if it's a dataclass or already a dict
        if is_dataclass(self.cfg.inference):
            inference_params = dc_asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)

        # Generate multiple attempts
        responses = []
        for attempt_idx in range(self.attempts):
            try:
                # Get the prompt using the base class method
                prompt = self.fill_prompt(data_point, all_data)

                # Generate using the LLM
                output_dict = await self.generate_with_semaphore(
                    prompt=prompt,
                    **inference_params,
                )

                generation = output_dict.get("generation", "")
                responses.append(generation)

                LOG.debug(f"Attempt {attempt_idx + 1}/{self.attempts} completed for question {data_point.get('question_id', 'unknown')}")

            except Exception as e:
                LOG.error(f"Error in attempt {attempt_idx + 1} for question {data_point.get('question_id', 'unknown')}: {str(e)}")
                responses.append(f"Error: {str(e)}")

        # Return result with all attempts
        result = {
            "responses": responses,
            "generation": responses[0] if responses else "",  # First attempt as main generation
        }

        return result


@hydra.main(version_base=None, config_name="base_multi_challenge_generation_config")
def main(cfg: MultiChallengeGenerationConfig):
    """Main entry point for MultiChallenge generation."""
    cfg = MultiChallengeGenerationConfig(**cfg)
    setup_logging(disable_hydra_logs=False)

    LOG.info("="*60)
    LOG.info("MultiChallenge Generation")
    LOG.info("="*60)
    LOG.info(f"Input: {cfg.input_file}")
    LOG.info(f"Output: {cfg.output_file}")
    LOG.info(f"Model: {cfg.server.get('model', 'N/A')}")
    LOG.info(f"Attempts: {cfg.attempts}")
    LOG.info(f"Max samples: {cfg.max_samples}")
    LOG.info("="*60)

    if cfg.dry_run:
        LOG.info("Dry run mode - loading data only")
        task = MultiChallengeGenerationTask(cfg)
        task.load_data()
        task.log_example_prompt(task.data)
        return

    # Run generation
    task = MultiChallengeGenerationTask(cfg)
    task.generate()

    LOG.info("="*60)
    LOG.info("Generation completed successfully!")
    LOG.info(f"Results saved to: {cfg.output_file}")
    LOG.info("="*60)


if __name__ == "__main__":
    main()

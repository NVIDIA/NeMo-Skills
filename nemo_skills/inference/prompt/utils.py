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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import MISSING, OmegaConf

from nemo_skills.inference.prompt.few_shot_examples import examples_map
from nemo_skills.utils import nested_dataclass

# listing all available configs here
prompt_types = [cfg.stem for cfg in Path(__file__).parent.glob("*.yaml")]

# listing all dataset folders available - note this will not be available
# if using from installed package but you need to have data files available anyway
datasets = [
    d.name for d in (Path(__file__).parents[3] / 'datasets').glob("*") if d.is_dir() and d.name != "__pycache__"
]


@nested_dataclass
class FewShotExamples:
    template: str = MISSING
    examples_type: Optional[str] = None
    num_few_shots: int = 5


@nested_dataclass
class PromptConfig:
    few_shot_examples: FewShotExamples = field(default_factory=FewShotExamples)
    prompt_template: str = MISSING
    user: str = MISSING
    system: str = MISSING
    context_type: str = "empty"
    stop_phrases: List[str] = field(default_factory=list)


@nested_dataclass
class Prompt:
    config: PromptConfig
    input_dict: Dict[str, Any]
    example_dicts: Optional[List[Dict[str, Any]]] = None
    context_template: Optional[str] = None
    generated_solution: str = ""

    def __post_init__(self):
        """Initialize example_dicts/context_template if not provided."""
        if self.example_dicts is None:
            self.example_dicts = examples_map.get(self.config.few_shot_examples.examples_type, [])[
                : self.config.few_shot_examples.num_few_shots
            ]
        if self.context_template is None:
            self.context_template = context_templates.get(self.config.context_type, "")

    def build_context(self, example_dict: Dict[str, Any]) -> str:
        """Builds the context string based on the example dictionary."""
        context = self.context_template.format(**example_dict)
        return context

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""
        context = self.build_context(example_dict)
        return self.config.few_shot_examples.template.format(context=context, **example_dict)

    def build_examples(self) -> str:
        """Builds all examples string concatenated by delimiter."""
        filled_examples = [self.build_filled_example(example) for example in self.example_dicts]
        examples = "".join(filled_examples)
        context = self.build_context(self.input_dict)
        user = self.config.user.format(examples=examples, context=context, **self.input_dict)
        return user

    def build_chat_prompt(self) -> List[Dict[str, str]]:
        """Builds a structured representation of the prompt."""
        structured_prompt = [{"role": "system", "content": self.config.system}] if self.config.system else []
        structured_prompt.append({"role": "user", "content": self.build_examples()})
        if self.generated_solution:
            structured_prompt.append({"role": "assistant", "content": self.generated_solution})
        return structured_prompt

    def __str__(self) -> str:
        """Returns the complete prompt string representation."""
        prompt = self.config.prompt_template.format(
            system=self.config.system,
            user=self.build_examples(),
            generated_solution=self.generated_solution,
        )
        return prompt


def get_prompt_config(prompt_type: str) -> PromptConfig:
    # reading prompt format from the yaml file, if not running through hydra
    config_path = Path(__file__).parent / f"{prompt_type}.yaml"
    with open(config_path, "rt", encoding="utf-8") as fin:
        # Load the YAML file using OmegaConf
        loaded_config = OmegaConf.load(fin)

        # Create a default PromptConfig and convert it to a DictConfig
        default_config = OmegaConf.create(asdict(PromptConfig()))

        # Merge the default config with the loaded config
        merged_config = OmegaConf.merge(default_config, loaded_config)

        prompt_config = OmegaConf.structured(PromptConfig(_init_nested=True, **merged_config))
        return prompt_config


# this does not come from the config as it's coupled with few shot examples structure
# and so most often will require changes to the code anyway
context_templates = {
    "empty": "",
    "reference_solution": "Reference solution (do not copy it):\n{reference_solution}\n\n",
    "masked_solution": "Reference solution:\n{reference_masked_solution}\n\n",
}

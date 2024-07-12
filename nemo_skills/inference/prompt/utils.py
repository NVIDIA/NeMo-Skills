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
import random
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from omegaconf import MISSING
except ImportError:
    MISSING = "???"

from nemo_skills.inference.prompt.few_shot_examples import examples_map
from nemo_skills.utils import nested_dataclass

LOG = logging.getLogger(__file__)

# listing all available configs here
prompt_types = [str(cfg).split('nemo_skills/inference/prompt/')[1] for cfg in Path(__file__).parent.glob("**/*.yaml")]
# removing .yaml from the end
prompt_types = [cfg[:-5] for cfg in prompt_types]

# listing all dataset folders available - note this will not be available
# if using from installed package but you need to have data files available anyway
datasets = [
    d.name for d in (Path(__file__).parents[3] / 'datasets').glob("*") if d.is_dir() and d.name != "__pycache__"
]


class BM25Retriever:
    def __init__(self, data_path: str, field: str):
        from rank_bm25 import BM25Okapi

        with open(data_path, "rt", encoding="utf-8") as fin:
            self.entries = [json.loads(x) for x in fin]

        corpus = [entry[field] for entry in self.entries]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 1):
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.entries, n=top_k)


@nested_dataclass
class FewShotExamplesConfig:
    template: str = ""
    num_few_shots: int = 0

    examples_type: Optional[str] = None
    example_dicts: Optional[List[Dict[str, Any]]] = None

    retrieval_field: Optional[str] = None  # e.g. question, reference_solution, etc.
    retrieval_file: Optional[str] = None  # needs to be provided if retrieval_field is not None
    retrieved_entries: int = 0
    randomize_retrieved_entries: bool = False
    max_retrieved_chars: int = 100000000  # no limit by default
    max_retrieved_chars_field: str = "reference_solution"
    retriever: Optional[Any] = None

    def __post_init__(self):
        """Error checks + building example_dicts and retriever if needed."""
        if self.examples_type is not None:  # building example_dicts
            self.example_dicts = examples_map[self.examples_type][: self.num_few_shots]

        if self.example_dicts is not None and self.num_few_shots > len(self.example_dicts):
            raise ValueError(
                f"There are not enough few shot examples in {self.examples_type}. "
                f"Max number is {len(self.example_dicts)}"
            )

        if self.retrieved_entries == 0:
            self.retrieved_entries = 2 * self.num_few_shots

        if self.example_dicts is not None and self.retriever is not None:
            raise ValueError("example_dicts and retriever cannot be used together")

        if self.retriever is not None:
            return

        if self.retrieval_field is not None:  # building retriever
            if self.retrieval_file is None:
                raise ValueError("retrieval_file must be provided if retrieval_field is not None")
            self.retriever = BM25Retriever(self.retrieval_file, field=self.retrieval_field)
        else:
            if self.retrieval_file is not None:
                raise ValueError("retrieval_field must be provided if retrieval_file is not None")

        if self.example_dicts is None and self.retriever is None and self.num_few_shots > 0:
            raise ValueError("You need to construct either example_dicts or retriever if num_few_shots > 0")


@nested_dataclass
class PromptConfig:
    few_shot_examples: FewShotExamplesConfig = field(default_factory=FewShotExamplesConfig)
    prompt_template: str = MISSING
    user: str = MISSING
    system: str = MISSING
    context_type: str = "empty"
    _context_template: Optional[str] = None  # cannot be set directly for now!
    stop_phrases: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize context_template if not provided."""
        self._context_template = context_templates[self.context_type]


class Prompt:
    def __init__(self, config):
        # rebuilding prompt config to make sure post init is called again in
        # case some parameters were manually changed after the config was created
        self.config = PromptConfig(_init_nested=True, **asdict(config))

    def build_context(self, example_dict: Dict[str, Any]) -> str:
        """Builds the context string based on the example dictionary."""
        context = self.config._context_template.format(**example_dict)
        return context

    def build_filled_example(self, example_dict: Dict[str, Any]) -> str:
        """Builds a filled example string based on the example dictionary."""
        context = self.build_context(example_dict)
        return self.config.few_shot_examples.template.format(context=context, **example_dict)

    def build_examples_dict(self, input_dict):
        if self.config.few_shot_examples.num_few_shots == 0:
            return []

        if self.config.few_shot_examples.example_dicts:
            return self.config.few_shot_examples.example_dicts[: self.config.few_shot_examples.num_few_shots]

        example_dicts = self.config.few_shot_examples.retriever.retrieve(
            query=input_dict[self.config.few_shot_examples.retrieval_field],
            top_k=self.config.few_shot_examples.retrieved_entries,
        )
        reference = input_dict[self.config.few_shot_examples.retrieval_field]
        # filtering exact match if it's there
        while example_dicts and example_dicts[0][self.config.few_shot_examples.retrieval_field] == reference:
            example_dicts = example_dicts[1:]

        # removing too long solutions
        example_dicts = [
            example_dict
            for example_dict in example_dicts
            if len(example_dict[self.config.few_shot_examples.max_retrieved_chars_field])
            < self.config.few_shot_examples.max_retrieved_chars
        ]

        if len(example_dicts) < self.config.few_shot_examples.num_few_shots:
            LOG.warning(
                'Too little examples (%d) found for the query "%s"',
                len(example_dicts),
                input_dict[self.config.few_shot_examples.retrieval_field],
            )

        # let's reverse the order to show the most relevant last
        examples = example_dicts[: self.config.few_shot_examples.num_few_shots][::-1]
        if self.config.few_shot_examples.randomize_retrieved_entries:
            random.shuffle(examples)

        return examples

    def build_user_message(self, input_dict: Dict[str, str]) -> str:
        """Builds all examples string concatenated by delimiter."""
        example_dicts = self.build_examples_dict(input_dict)

        filled_examples = [self.build_filled_example(example) for example in example_dicts]
        examples = "".join(filled_examples)
        context = self.build_context(input_dict)
        user = self.config.user.format(examples=examples, context=context, **input_dict)
        return user

    def build_string(self, input_dict: Dict[str, str]) -> str:
        """Returns the complete prompt string representation."""
        generation = input_dict.get("generation", "")

        prompt = self.config.prompt_template.format(
            system=self.config.system,
            user=self.build_user_message(input_dict),
            generation=generation,
        )
        return prompt


def get_prompt_config(prompt_type: str) -> PromptConfig:
    """
    Reads the prompt config from the yaml file.

    Args:
        prompt_type: The name of the prompt config file. Can be the path to a yaml file or one of the available configs.

    Returns:
        The prompt config object.
    """
    # reading prompt format from the yaml file, if not running through hydra
    internal_config_path = Path(__file__).parent / f"{prompt_type}.yaml"
    if internal_config_path.exists():
        config_path = internal_config_path

    elif ".yaml" in prompt_type:
        # Assume prompt type is a str to a filepath
        config_path = Path(prompt_type).absolute()

    else:
        raise ValueError(f"Prompt config not found for {prompt_type}")

    with open(config_path, "rt", encoding="utf-8") as fin:
        prompt_config = PromptConfig(_init_nested=True, **yaml.safe_load(fin))
        return prompt_config


# this does not come from the config as it's coupled with few shot examples structure
# and so most often will require changes to the code anyway
context_templates = {
    "empty": "",
    "reference_solution": "\n\nReference solution (do not copy it):\n{reference_solution}",
    "masked_solution": "\n\nReference solution:\n{masked_reference_solution}",
    "table": "\n\nUse the following table to answer the question:\n{table}",
    "table_solution": (
        "\n\nUse the following table to answer the question:\n{table}\n"
        "Reference solution (do not copy it):\n{reference_solution}"
    ),
}

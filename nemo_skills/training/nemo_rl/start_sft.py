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

# copied from https://github.com/NVIDIA/NeMo-RL/blob/main/examples/run_sft.py

import argparse
import json
import os
import pprint
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, cast

from datasets import Dataset, load_dataset, load_from_disk
from nemo_rl.algorithms.sft import MasterConfig, setup, sft_train
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig, hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec

# from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TokenizerType = PreTrainedTokenizerBase


class PromptResponseDataset:
    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str,
        input_key: str = "input",
        output_key: str = "output",
        num_proc: int | None = None,
        force_reprocess: bool = False,  # Only keep this to control overwriting
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.force_reprocess = force_reprocess

        # Auto-determine number of processes
        if num_proc is None:
            cpu_count = os.cpu_count() or 2
            self.num_proc = min(8, cpu_count)
        else:
            self.num_proc = num_proc

        # Train and validation set processing
        self.formatted_ds = {
            "train": self.load_or_process_split(train_ds_path, "train"),
            "validation": self.load_or_process_split(val_ds_path, "val"),
        }

        self.task_spec = TaskDataSpec("json_dataset")

    def load_or_process_split(self, path: str, split_name: str) -> Dataset:
        data_path = Path(path)
        cache_dir = data_path.parent / ".cache" / f"{split_name}_{data_path.stem}"
        sig_file = cache_dir / "signature.json"
        file_size = str(data_path.stat().st_size)
        if cache_dir.exists() and sig_file.exists() and not self.force_reprocess:
            with open(sig_file) as f:
                old_sig = json.load(f)["size"]
            if old_sig == file_size:
                print(f"[Cache] Loading {split_name} dataset from: {cache_dir}")
                return load_from_disk(str(cache_dir))
            else:
                print(f"[Cache] Invalidated (file size changed): {path}")

        # Re-process dataset
        print(f"[Map] Processing {split_name} dataset from: {path}")
        raw_dataset = load_dataset("json", data_files=str(path))["train"]

        mapped_dataset = raw_dataset.map(
            self.add_messages_key,
            batched=True,
            num_proc=self.num_proc,
        )
        # Save dataset + new size signature
        cache_dir.mkdir(parents=True, exist_ok=True)
        mapped_dataset.save_to_disk(str(cache_dir))
        with open(sig_file, "w") as f:
            json.dump({"size": file_size}, f)

        print(f"[Cache] Saved {split_name} dataset to: {cache_dir}")
        return mapped_dataset

    def add_messages_key(self, examples: dict[str, list[Any]]) -> dict[str, list[list[dict[str, Any]]]]:
        return {
            "messages": [
                [
                    {"role": "user", "content": input_},
                    {"role": "assistant", "content": output},
                ]
                for input_, output in zip(examples[self.input_key], examples[self.output_key])
            ]
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SFT training with configuration")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# TODO: remove these two functions and import after bug fix is merged
def get_first_index_that_differs(str1: str, str2: str) -> int:
    """Get the first index that differs between two strings."""
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            return i
    return min(len(str1), len(str2))


def get_formatted_message_log(
    message_log: LLMMessageLogType,
    tokenizer: TokenizerType,
    task_data_spec: TaskDataSpec,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    add_generation_prompt: bool = False,
) -> LLMMessageLogType:
    """Format and tokenize chat messages using the specified template.

    Args:
        message_log: List of message dicts with 'role' and 'content' keys
        tokenizer: Tokenizer for converting text to token IDs
        task_data_spec: Task spec for this dataset.
        add_bos_token: Whether to add bos token to first message if it is not already present. Default: True
        add_eos_token: Whether to add eos token to last message if it is not already present. Default: True
        add_generation_prompt: Whether to include assistant's generation prompt in user messages. Default: False

    Returns:
        The message log with updated 'token_ids' and 'content' fields.
    """
    new_message_log: LLMMessageLogType = []
    prev_formatted_message = ""
    message_log_strs: list[dict[str, str]] = cast(
        list[dict[str, str]], message_log
    )  # we just use the str:str parts here

    if task_data_spec.prompt:
        message_log_strs = [
            {
                "role": "user",
                "content": task_data_spec.prompt.format(message_log_strs[0]["content"]),
            }
        ] + message_log_strs[1:]

    for i, message in enumerate(message_log_strs):
        # If enabled, add_generation_prompt is only used on user messages to include
        # the assistant's generation prompt as part of the user message.
        formatted_message: str = tokenizer.apply_chat_template(  # type: ignore
            message_log_strs[: i + 1],
            add_generation_prompt=add_generation_prompt and message["role"] == "user",
            tokenize=False,
            add_special_tokens=False,
        )

        ## get the length of the previous message, excluding the eos token (if present)
        prev_message_len_no_eos: int = get_first_index_that_differs(
            prev_formatted_message,
            formatted_message,
        )

        ## pull out the chunk corresponding to the current message
        message_chunk = formatted_message[prev_message_len_no_eos:]

        if i == 0:
            if add_bos_token:
                if tokenizer.bos_token is None:
                    warnings.warn(
                        "add_bos_token is True but the tokenizer does not have a BOS token. Skipping BOS token addition."
                    )
                elif not message_chunk.startswith(tokenizer.bos_token):
                    message_chunk = tokenizer.bos_token + message_chunk

        if i == len(message_log_strs) - 1:
            if add_eos_token:
                if tokenizer.eos_token is None:
                    warnings.warn(
                        "add_eos_token is True but the tokenizer does not have an EOS token. Skipping EOS token addition."
                    )
                elif not message_chunk.endswith(tokenizer.eos_token):
                    message_chunk += tokenizer.eos_token

        new_message = message.copy()
        new_message["token_ids"] = tokenizer(message_chunk, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0]
        if len(new_message["token_ids"]) == 0:
            # if there is an empty message, the empty `token_ids` tensor ends up being in fp32,
            # which causes `_validate_tensor_consistency` to fail. To fix this, we convert the
            # empty tensor to int64.
            new_message["token_ids"] = new_message["token_ids"].to(torch.int64)  # type: ignore

        new_message["content"] = message_chunk
        new_message_log.append(new_message)

        prev_formatted_message = formatted_message

    return new_message_log


# =======================================================
# Data Processing
# =======================================================
def sft_preprocessor(
    datum_dict: Dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
    add_bos: bool = True,
    add_eos: bool = True,
    add_generation_prompt: bool = False,
) -> DatumSpec:
    """Process a datum dictionary for SFT training."""
    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][: min(4, max_seq_length // len(message_log))]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    return output


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig):
    print("\n▶ Setting up data...")
    assert data_config["dataset_name"] == 'prompt_response_dataset'
    data = PromptResponseDataset(
        data_config["train_data_path"],
        data_config["val_data_path"],
        data_config["input_key"],
        data_config["output_key"],
        force_reprocess=data_config.get("force_reprocess", False),
    )
    print(
        f"  ✓ Training and validation datasets loaded with {len(data.formatted_ds['train'])} and "
        f"{len(data.formatted_ds['validation'])} samples, respectively."
    )

    train_dataset = data.formatted_ds["train"]
    val_dataset = data.formatted_ds["validation"]
    sft_task_spec = data.task_spec

    train_dataset = AllTaskProcessedDataset(
        train_dataset,
        tokenizer,
        sft_task_spec,
        partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset = AllTaskProcessedDataset(
        val_dataset,
        tokenizer,
        sft_task_spec,
        partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        ),
        max_seq_length=data_config["max_input_seq_length"],
    )

    return train_dataset, val_dataset, sft_task_spec


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "sft.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # setup data
    (
        dataset,
        val_dataset,
        sft_task_spec,
    ) = setup_data(tokenizer, config["data"])

    (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)
    sft_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        sft_task_spec,
        checkpointer,
        sft_save_state,
    )


if __name__ == "__main__":
    main()

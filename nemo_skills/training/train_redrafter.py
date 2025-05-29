#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


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


"""
Script adapted from:
https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/cmd/train.py
Training arguments:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""

import os
import math
import multiprocessing
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import datasets
import numpy as np
import torch
import transformers
from transformers import Trainer

from recurrent_drafting.configuration_drafter import DrafterConfig
from recurrent_drafting.modeling_drafter import Drafter
from recurrent_drafting.train import data
from recurrent_drafting.train.loss import drafter_loss
from recurrent_drafting.train.model import ReDrafter

local_rank = int(os.getenv("LOCAL_RANK", "0"))

class ReDrafterTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_logged_step = -1
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]:
                The computed loss, optionally with model outputs.
        """

        next_n = self.args.drafter_predict_n_tokens
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], next_n=next_n
        )
        loss, log, eval_log = drafter_loss(
            logits, inputs["labels"], next_n, self.args.drafter_top_k
        )
        
        # Only log once per global step
        should_log = (local_rank == 0) and \
                    (self.state.global_step % self.args.logging_steps == 0) and \
                    (self.state.global_step != self.last_logged_step)
        
        if should_log:
            self.last_logged_step = self.state.global_step
            rounded_log = {k: round(v, 4) if isinstance(v, (float, np.float32, np.float64)) else v 
                           for k, v in log.items()}
            self.log(rounded_log)
        
        return (loss, eval_log) if return_outputs else loss



@dataclass
class ModelArguments:
    llm_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    drafter_name_or_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. "
            "Sequences will be right padded (and possibly truncated)."
        },
    )
    drafter_predict_n_tokens: int = field(
        default=5,
        metadata={"help": "Drafter predicts k extra tokens."},
    )
    drafter_top_k: int = field(
        default=5,
        metadata={"help": "Drafter top k accuracy for each token."},
    )
    drafter_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for the drafter."},
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={"help": "Include inputs for metrics."},
    )
    phase: str = field(
        default="train",
        metadata={"help": "train or eval"},
    )
    rnn: bool = field(
        default=False,
        metadata={"help": "Include rnn in drafter."},
    )
    dataset: str = field(
        default="nvidia/OpenMathReasoning",
        metadata={"help": "Huggingface dataset."},
    )
    dataset_split: str = field(
        default="tir",
        metadata={"help": "Huggingface dataset split."},
    )
    dataset_nrows: int = field(
        default=100000,
        metadata={"help": "Number of training rows."},
    )
    dataset_eval: str = field(
        default="nvidia/OpenMathReasoning",
        metadata={"help": "Huggingface dataset."},
    )
    dataset_eval_split: str = field(
        default="tir",
        metadata={"help": "Huggingface dataset split."},
    )
    report_to: str = field(  # Add this line
        default="none",  # Disables W&B and other logging integrations
        metadata={"help": "Disable logging (e.g., 'none', 'all', 'wandb', 'tensorboard')"},
    )
    logging_steps: int = field(
        default=30,
        metadata={"help": "Number of logging steps."},
    )


def get_tokenizer(model_args, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,  # Try with False
    )
    return tokenizer


def generate_drafter_config_from_base(llm, training_args):
    return DrafterConfig(
        vocab_size=llm.lm_head.weight.shape[0],
        hidden_size=llm.lm_head.weight.shape[-1],
        exit_dim=2 * llm.lm_head.weight.shape[-1],
        num_draft_layers=training_args.drafter_num_layers,
        rnn=training_args.rnn,
    )

def get_compute_metrics(training_args):
    predict_n_tokens = training_args.drafter_predict_n_tokens

    def compute_metrics(all_preds):
        return_val = {}
        for i in range(predict_n_tokens):
            for k in range(1, training_args.drafter_top_k + 1):
                metric_value = np.mean(all_preds.predictions[i * predict_n_tokens + k - 1])
                return_val[f"redrafter{i}_top{k}"] = metric_value
        return return_val

    return compute_metrics


def record_to_training_instance(
    record: Dict[str, Any],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """Given a conversation, outputs a training instance into which the conversation is
    converted.

    Returns:

      A dictionary of tokenized inputs, labels, and attention mask.

    """
    problem = record["problem"]
    solution = record["generated_solution"]
    IGNORE_TOKEN_ID = -100

    messages = [
            {"role": "system", "content": f""},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}]

    input_text = tokenizer.apply_chat_template(messages,
                                                   tokenize = False,
                                                   add_generation_prompt = False)

    input_dict = tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    input_dict = {k:v.flatten() for k,v in input_dict.items()}
    input_dict['labels'] = input_dict['input_ids'].clone()
    input_dict['labels'][~input_dict['attention_mask'].bool()] = IGNORE_TOKEN_ID

    return input_dict

def train(model_args, training_args):
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    # Load data in streaming mode
    train_dataset = datasets.load_dataset(
        training_args.dataset, 
        split=training_args.dataset_split,
        streaming=True
    )
    
    # Take only the required number of samples if specified
    if hasattr(training_args, 'dataset_nrows') and training_args.dataset_nrows:
        train_dataset = train_dataset.take(training_args.dataset_nrows)
    
    # Convert to regular dataset and map
    train_dataset = datasets.Dataset.from_generator(
        lambda: train_dataset, 
        features=train_dataset.features
    ).map(
        lambda x: record_to_training_instance(x, tokenizer),
        num_proc=min(32, multiprocessing.cpu_count()),
    )
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(model_args.llm_name_or_path)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    # Load and freeze the base model
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.llm_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    any((setattr(param, "requires_grad", False) for param in llm.base_model.parameters()))
    drafter_config = generate_drafter_config_from_base(llm, training_args)
    drafter = Drafter(drafter_config)
    drafter = drafter.to(llm.dtype)
    redrafter = ReDrafter(llm, drafter)
    # Format output dir
    training_args.output_dir = (
        f"{training_args.output_dir}"
        f"_redrafter_{model_args.llm_name_or_path.split('/')[-1]}"
        f"_n_{training_args.drafter_predict_n_tokens}"
        f"_lr_{training_args.learning_rate}"
        f"_layers_{training_args.drafter_num_layers}"
    )
    print(training_args)
    trainer = ReDrafterTrainer(
        model=redrafter,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
    )
    trainer.train(
        resume_from_checkpoint=bool(
            list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        )
    )
    # Save ReDrafter
    drafter.save_pretrained(training_args.output_dir)


def eval(model_args, training_args):
    tokenizer = get_tokenizer(model_args, training_args)
    compute_metrics = get_compute_metrics(training_args)
    # Load data
    eval_dataset = (
        datasets.load_dataset(training_args.dataset, 
                              num_proc=multiprocessing.cpu_count(),
                              split=training_args.dataset_split)
        .map(
            lambda x: record_to_training_instance(x, tokenizer),
            num_proc=min(32, multiprocessing.cpu_count()),
        )
    )
    # Load ReDrafter
    redrafter = ReDrafter.from_pretrained(
        model_args.llm_name_or_path,
        model_args.drafter_name_or_path,
        torch_dtype=torch.float16,
    )
    # Start trainer
    trainer = ReDrafterTrainer(
        model=redrafter,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset,
    )
    trainer.evaluate()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    assert training_args.phase in ["train", "eval"]
    run = train if training_args.phase == "train" else eval
    run(model_args, training_args)

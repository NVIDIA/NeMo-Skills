#!/usr/bin/env python3
#
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

import argparse
import glob
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Copies the last checkpoint from the training folder.")
    parser.add_argument(
        "--name_prefix",
        required=True,
        help="Name of the final checkpoint. Will append '-last' automatically.",
    )
    parser.add_argument(
        "--untarred_nemo_dir",
        required=True,
        help="Path to the untarred nemo checkpoint to get config and tokenizers",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing the checkpoints (must include one ending with '-last')",
    )
    args = parser.parse_args()

    # Find the checkpoint directory that ends with '-last'
    last_checkpoint = None
    for ckpt in os.listdir(args.checkpoint_dir):
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt)
        if os.path.isdir(ckpt_path) and ckpt.endswith("-last"):
            last_checkpoint = ckpt_path
            logging.info(f"Found checkpoint: {ckpt}")
            break

    if last_checkpoint is None:
        logging.error("No checkpoint directory ending with '-last' found.")
        return

    if "model_weights" in os.listdir(last_checkpoint):
        src_model_weights = os.path.join(last_checkpoint, "model_weights")
    else:
        src_model_weights = last_checkpoint

    dest_ckpt_base = os.path.join(args.checkpoint_dir, args.name_prefix + "-last")
    dest_model_weights = os.path.join(dest_ckpt_base, "model_weights")
    os.makedirs(dest_model_weights, exist_ok=True)

    copy_items = []
    for item in os.listdir(src_model_weights):
        src_item_path = os.path.join(src_model_weights, item)
        # if item is not a directory, skip it
        if not os.path.isdir(src_item_path):
            copy_items.append(src_item_path)
            continue

        # transformer engine states, leave them out
        if item.endswith("._extra_state"):
            copy_items.append(src_item_path)
            continue

        # skipping optimizer states
        if item.startswith("optimizer."):
            logging.info(f"Skipping optimizer state: {item}")
            continue

        logging.info(f"Processing weight directory: {item}")
        copy_items.append(src_item_path)

    for item in copy_items:
        item_name = os.path.basename(item)
        dest_item_path = os.path.join(dest_model_weights, item_name)
        if os.path.isfile(item):
            logging.info(f"Copying file {item} to {dest_item_path}")
            shutil.copy(item, dest_item_path)
        else:
            logging.info(f"Copying directory {item} to {dest_item_path}")
            shutil.copytree(item, dest_item_path, dirs_exist_ok=True)

    shutil.copy(
        os.path.join(args.untarred_nemo_dir, "model_config.yaml"),
        os.path.join(dest_ckpt_base, "model_config.yaml"),
    )
    logging.info("Copied model_config.yaml")

    for pattern in ["*.model", "*.json"]:
        for file in glob.glob(os.path.join(args.untarred_nemo_dir, pattern)):
            dest_file = os.path.join(dest_ckpt_base, os.path.basename(file))
            shutil.copy(file, dest_file)
            logging.info(f"Copied {file} to {dest_file}")

    logging.info(f"Checkpoint successfully copied to: {dest_ckpt_base}")


if __name__ == "__main__":
    main()

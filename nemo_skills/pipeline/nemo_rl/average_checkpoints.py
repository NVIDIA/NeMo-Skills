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
import json
import logging
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)


def list_candidate_model_dirs(checkpoint_dir, steps):
    """List subfolders matching specific step numbers."""
    out = []
    for name in os.listdir(checkpoint_dir):
        if not os.path.isdir(os.path.join(checkpoint_dir, name)):
            continue
        if any(f"step_{s}" in name for s in steps):
            out.append(name)
    out.sort()
    return out


def find_index_json(model_dir):
    """Find model.safetensors.index.json if exists."""
    for f in os.listdir(model_dir):
        if f.endswith(".safetensors.index.json"):
            return os.path.join(model_dir, f)
    return None


def build_key_to_shard_map(model_dir):
    """Return (key->file map, list of shards)."""
    idx_path = find_index_json(model_dir)
    if idx_path:
        with open(idx_path, "r") as fr:
            idx = json.load(fr)
        weight_map = idx.get("weight_map", {})
        shards = sorted(list(set(weight_map.values())))
        return weight_map, shards
    # fallback: scan shards
    shards = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    key2file = {}
    for shard in shards:
        spath = os.path.join(model_dir, shard)
        with safe_open(spath, framework="pt") as f:
            for k in f.keys():
                key2file[k] = shard
    return key2file


def copy_side_files(src_model_dir, dst_dir):
    """Copy all non-weight files from the first model directory."""
    for fname in os.listdir(src_model_dir):
        if fname.endswith(".safetensors") or fname.endswith(".safetensors.index.json"):
            continue  # skip weight shards and index
        src = os.path.join(src_model_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Root directory containing multiple model subfolders.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        required=True,
        help="List of step numbers to include (e.g. --steps 100 200 300).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="If set, will delete the original step directories after averaging.",
    )

    args = parser.parse_args()

    model_dirs = list_candidate_model_dirs(args.checkpoint_dir, args.steps)
    if not model_dirs:
        raise SystemExit("No valid model subdirectories found.")

    logging.info("Selected model dirs:")
    for d in model_dirs:
        logging.info("  - %s", d)

    n = len(model_dirs)
    logging.info("Averaging %d checkpoints", n)

    first_dirname = model_dirs[0]
    first_dir = os.path.join(args.checkpoint_dir, first_dirname)
    key2file_first = build_key_to_shard_map(first_dir)
    keys = sorted(key2file_first.keys())
    logging.info("Total parameter tensors: %d", len(keys))

    # Build per-directory key->file maps and check consistency
    per_dir_key2file = {first_dirname: key2file_first}
    for dirname in model_dirs[1:]:
        md = os.path.join(args.checkpoint_dir, dirname)
        k2f, _ = build_key_to_shard_map(md)
        if set(k2f.keys()) != set(keys):
            raise SystemExit("[Strict] Key sets differ between %s and first model." % dirname)
        per_dir_key2file[dirname] = k2f

    # Output directory
    out_dir = os.path.join(args.checkpoint_dir, "final_hf_model")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Group keys by shard based on first model
    shard_to_keys = {}
    for k in keys:
        shard_to_keys.setdefault(key2file_first[k], []).append(k)

    # Average each shard separately
    for shard_name, shard_keys in shard_to_keys.items():
        logging.info("Averaging shard %s with %d tensors", shard_name, len(shard_keys))
        out_state = {}
        for i, k in enumerate(shard_keys):
            # read tensor from first model
            with safe_open(os.path.join(first_dir, shard_name), framework="pt") as f0:
                t0 = f0.get_tensor(k)
                ref_shape = tuple(t0.shape)
                ref_dtype = t0.dtype
                acc = t0.to(dtype=torch.float32)

            # add from other models
            for dirname in model_dirs[1:]:
                md = os.path.join(args.checkpoint_dir, dirname)
                shard_k = per_dir_key2file[dirname][k]
                with safe_open(os.path.join(md, shard_k), framework="pt") as fh:
                    tk = fh.get_tensor(k)
                    if tuple(tk.shape) != ref_shape:
                        raise SystemExit("[Strict] Shape mismatch for %s in %s" % (k, dirname))
                    if tk.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                        raise SystemExit("Int tensor not supported for averaging: %s (%s)" % (k, tk.dtype))
                    acc.add_(tk.to(dtype=torch.float32))

            acc.div_(float(n))
            out_state[k] = acc.to(dtype=ref_dtype)
            if (i % 4000) == 0:
                logging.info("  ... %d / %d tensors", i, len(shard_keys))

        out_path = os.path.join(out_dir, shard_name)
        save_file(out_state, out_path)
        logging.info("Saved shard: %s", out_path)
        out_state.clear()

    # Copy or create index.json
    idx_path = find_index_json(first_dir)
    if idx_path:
        with open(idx_path, "r") as fr:
            idx = json.load(fr)
        meta = idx.setdefault("metadata", {})
        meta["averaged_from"] = model_dirs
        with open(os.path.join(out_dir, os.path.basename(idx_path)), "w") as fw:
            json.dump(idx, fw, indent=2)
        logging.info("Copied index json.")
    else:
        weight_map = {k: key2file_first[k] for k in keys}
        idx = {"metadata": {"format": "pt", "averaged_from": model_dirs}, "weight_map": weight_map}
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fw:
            json.dump(idx, fw, indent=2)
        logging.info("Generated simple index json.")

    copy_side_files(first_dir, out_dir)
    logging.info("Averaged (sharded) checkpoint saved at: %s", out_dir)
    logging.info("Done.")

    if args.cleanup:
        logging.info("Cleaning up original step directories...")
        for model_dir in model_dirs:
            full_path = os.path.join(args.checkpoint_dir, model_dir)
            try:
                shutil.rmtree(full_path)
                logging.info("Deleted directory: %s", full_path)
            except Exception as e:
                logging.warning("Failed to delete %s: %s", full_path, e)


if __name__ == "__main__":
    main()

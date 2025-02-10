import argparse
import os

import numpy as np
from tqdm import tqdm


def load_packed_arrays(prefix):
    # Load the .npy files corresponding to the prefix
    input_ids = np.load(f"{prefix}.input_ids.npy")
    loss_mask = np.load(f"{prefix}.loss_mask.npy")
    seq_start_id = np.load(f"{prefix}.seq_start_id.npy")
    return input_ids, loss_mask, seq_start_id


def merge_packed_arrays(prefixes, output_prefix):
    total_samples = 0
    max_input_len = 0
    max_seq_starts = 0

    # First pass: compute total samples and global maximum lengths
    arrays_list = []
    for prefix in prefixes:
        input_ids, loss_mask, seq_start_id = load_packed_arrays(prefix)
        arrays_list.append((input_ids, loss_mask, seq_start_id))
        total_samples += input_ids.shape[0]
        max_input_len = max(max_input_len, input_ids.shape[1])
        max_seq_starts = max(max_seq_starts, seq_start_id.shape[1])

    # Allocate merged arrays
    merged_input_ids = -np.ones((total_samples, max_input_len), dtype=np.int32)
    merged_loss_mask = np.ones((total_samples, max_input_len), dtype=np.bool_)
    merged_seq_start_id = -np.ones((total_samples, max_seq_starts), dtype=np.int32)

    # Second pass: copy into preallocated arrays
    sample_idx = 0
    for (input_ids, loss_mask, seq_start_id), prefix in zip(arrays_list, prefixes):
        n_samples = input_ids.shape[0]
        print("Processing prefix:", prefix)
        for i in tqdm(range(n_samples)):
            curr_input = input_ids[i]
            curr_loss = loss_mask[i]
            curr_seq_start = seq_start_id[i]
            merged_input_ids[sample_idx, : len(curr_input)] = curr_input
            merged_loss_mask[sample_idx, : len(curr_loss)] = curr_loss
            merged_seq_start_id[sample_idx, : len(curr_seq_start)] = curr_seq_start
            sample_idx += 1

    # Save merged files
    np.save(f"{output_prefix}.input_ids.npy", merged_input_ids)
    np.save(f"{output_prefix}.loss_mask.npy", merged_loss_mask)
    np.save(f"{output_prefix}.seq_start_id.npy", merged_seq_start_id)
    print(f"Merged arrays saved with prefix '{output_prefix}'.")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple packed npy files.")
    parser.add_argument(
        "--input_prefixes",
        nargs="+",
        required=True,
        help="List of file prefixes to merge (each prefix should have .input_ids.npy, .loss_mask.npy, .seq_start_id.npy)",
    )
    parser.add_argument("--output_prefix", required=True, help="Output file prefix for the merged arrays")
    args = parser.parse_args()

    merge_packed_arrays(args.input_prefixes, args.output_prefix)


if __name__ == "__main__":
    main()

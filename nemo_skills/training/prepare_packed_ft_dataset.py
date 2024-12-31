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

import collections
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from tqdm import tqdm

PACKING_ALGOS = ['first_fit_decreasing', 'first_fit_shuffle']


def find_first_bin_that_fits(bins: List[List[int]], s: int, bin_size: int) -> int:
    """
    Finds the first bin in a list of bins that has enough space to fit a sequence of size 's'.

    Args:
      bins: A list of lists, where each inner list represents a bin and contains the current elements in that bin.
      s: The size of the sequence to be placed in a bin.
      bin_size: The maximum capacity of each bin.

    Returns:
      The index of the first bin that can fit the sequence 's', or -1 if no such bin exists.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the indices of the sequences assigned to that bin.
    """
    res = []
    for s in seqlens:
        first_bin = find_first_bin_that_fits(res, s, pack_size)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit Decreasing algorithm.

    This is a variation of the First-Fit algorithm where the sequences are sorted by decreasing length before packing.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    sorted_seqlens = sorted(seqlens, reverse=True)
    return first_fit(sorted_seqlens, pack_size)


def first_fit_shuffle(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit with Shuffling algorithm.

    This variation shuffles the order of the sequences before applying the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, similar to the output of the 'first_fit' function.
    """
    shuffled_seqlens = seqlens[:]
    np.random.shuffle(shuffled_seqlens)
    return first_fit(shuffled_seqlens, pack_size)


def create_hist(dataset: np.array, truncate_seq_len: int):
    """
    Creates a histogram of sequence lengths from a tokenized dataset.

    This function analyzes the tokenized dataset and creates a histogram showing the distribution of sequence lengths.

    Args:
      dataset: A NumPy array containing the tokenized sequences. Each element is a dictionary that contains at minimum
               the key `input_ids`.
      truncate_seq_len: The maximum sequence length to consider in the histogram.

    Returns:
      sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences from the dataset.
      histogram: A list representing the histogram data (number of sequences for each length).
    """
    logging.info("Creating histogram from tokenized dataset...")

    sequences = collections.defaultdict(list)
    counts = [0] * (truncate_seq_len + 1)

    for item_dict in dataset:
        # Minus 1 here to account for the fact that transformer input and label have one less token than the full sequence
        # Input is missing the last token and label is missing the first token (this way the tokens are aligned for next token prediction).
        # We want pack size to be the length of the actual input and label, hence minus 1.
        seq_len = len(item_dict['input_ids']) - 1
        sequences[seq_len].append(item_dict)
        counts[seq_len] += 1

    logging.debug("Histogram of sequence lengths")
    logging.debug(counts)

    histogram = []
    for seq_len in range(truncate_seq_len + 1):
        histogram.append(len(sequences[seq_len]))

    return sequences, histogram


def create_packing_strategy(
    histogram: List[int], pack_size: int, packing_algorithm: str = 'first_fit'
) -> List[List[int]]:
    """
    Packs sequences into bins using the specified packing algorithm.

    This function takes the histogram of sequence lengths, desired pack size, and a string representing the packing
    algorithm to use. It then calls the corresponding function (e.g., 'first_fit_decreasing') and performs the
    packing process using only sequence lengths as input (without the actual sequences).

    Args:
          histogram: A list representing the histogram data (number of sequences for each length).
          pack_size: The maximum capacity of each bin.
          packing_algorithm: One of the supported packing algorithms from ['first_fit_decreasing', 'first_fit_shuffle']

    Returns:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin.
          pack_metadata: A dict that records packing metadata, for instance the max number of samples per bin.
    """

    logging.info(f"Packing sequences to length {pack_size}...")

    all_seq_lens = []
    for i, count in enumerate(histogram):
        all_seq_lens.extend([i] * count)

    packing_fn = globals()[packing_algorithm]
    assignments = packing_fn(all_seq_lens, pack_size)
    packed_seq_lens = [sum(x) for x in assignments]
    packing_factor = len(all_seq_lens) / len(packed_seq_lens)

    max_seqlen = max(all_seq_lens)
    max_samples_per_bin = max([len(b) for b in assignments])
    packing_metadata = {'dataset_max_seqlen': max_seqlen, 'max_samples_per_bin': max_samples_per_bin}

    logging.debug("Packed sequence lengths:")
    logging.debug(packed_seq_lens)
    logging.info(f"Packing is {sum(packed_seq_lens)/len(packed_seq_lens)/pack_size*100:.2f}% efficient")
    logging.info(
        f">>>>> For pack size {pack_size}, average number of sequences per pack is n = {packing_factor:.3f} <<<<<"
    )
    return assignments, packing_metadata


def fill_packing_strategy(
    assignments: List[List[int]], sequences: Dict[int, List[Dict]], pack_size: int, pad_id: int
) -> List[Dict]:
    """
    Fills the packing strategy with actual sequence data based on assignments and sequence information.

    This function takes the assignments generated by the packing algorithm (containing sequence length indices),
    the original sequences data, and the pack size. It iterates through the assignments, retrieves the corresponding
    sequences from the sequences dictionary, and constructs the final output data structure with input IDs, loss masks
    (if available), and starting indices for each sequence in a packed sequence.

    Args:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin (output of 'create_packing_strategy').
          sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences
                      from the dataset (output of 'create_hist').
          pack_size: The maximum capacity of each bin.
          pad_id: The tokenizer's padding token.

    Returns:
          output_data: A list of dictionaries, where each dictionary represents a packed sequence with its input IDs,
                        loss mask (if available), and starting indices.
    """
    ifile_handles = dict()
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            perm = np.random.permutation(len(per_seq_data))
            input_ids = np.array([x['input_ids'] for x in per_seq_data])[perm].tolist()
            try:
                loss_mask = np.array(
                    [
                        [
                            idx >= x['answer_start_idx'] and x['input_ids'][idx] != pad_id
                            for idx in range(len(x['input_ids']))
                        ]
                        for x in per_seq_data
                    ]
                )[perm].tolist()
            except KeyError:
                loss_mask = None
            ifile_handles[seq_len] = (input_ids, loss_mask)

    input_ids, loss_mask, seq_start_id = {}, {}, {}

    for oindex, assignment in tqdm(enumerate(assignments), total=len(assignments)):
        _input_ids, _loss_mask, _seq_start_id = [], [], [0]

        for seq_length in assignment:
            _input_ids.extend(ifile_handles[seq_length][0].pop())
            _loss_mask.extend(ifile_handles[seq_length][1].pop())
            _seq_start_id.append(len(_input_ids))

        input_ids[oindex] = _input_ids
        loss_mask[oindex] = _loss_mask
        seq_start_id[oindex] = _seq_start_id[:-1]

    output_data = []
    for i in range(len(input_ids)):
        item_dict = {'input_ids': input_ids[i], 'loss_mask': loss_mask[i], 'seq_start_id': seq_start_id[i]}
        output_data.append(item_dict)

    assert all(not seq[0] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    assert all(not seq[1] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    return output_data


if TYPE_CHECKING:
    from omegaconf import DictConfig

"""
Script to prepare packed dataset from a SFT/PEFT dataset in the jsonl format.
Two main steps are run in this script:
1. The online processing code in GPTSFTDataset is run (including prompt template manipulation,
sequence length truncation, tokenization, etc) and the result is an array of tokenized sequences,
represented by indices).
2. The sequences are grouped by length, and a packing algorithm is run. (https://en.wikipedia.org/wiki/Bin_packing_problem#Offline_algorithms)
Currently, two variants of "first fit" are supported.
"first_fit_decreasing" sorts the sequences in decreasing order before applying first-fit.
It generates a more optimal packing, but it tends to keep all short sequences together, which may affect convergence.
"first_fit_shuffle" runs first-fit in a random order. Packing is less optimal but it keeps the dataset order random.
The recommendation is to run "first_fit_shuffle" and check the packed sequence lengths in the printout.
If they are similar to the target length (i.e. packing is efficient), then use shuffle. Otherwise try first_fit_decreasing.

Example usage:

python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
   model.data.train_ds.file_names=[/path/to/training.jsonl] \
   model.data.train_ds.max_seq_length=2048 \
   +tokenizer_path=<see note 1 below> \
   +output_dir=/path/to/output_folder \
   +pack_sizes=[2048,4096,8192]

when using context parallelism (CP) with packed dataset, CP size needs to be set in the command:

python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
    model.data.train_ds.file_names=[/path/to/training.jsonl] \
    model.data.train_ds.max_seq_length=4096 \
    ++model.context_parallel_size=2 \
    +tokenizer_path=<see note 1 below> \
    +output_dir=/path/to/output_folder \
    +pack_sizes=[4096]

Note:
  - Tokenizer path supports SentencePiece tokenizer and HF tokenizer.
    For SentencePiece tokenizer, specify the file /path/to/tokenizer.model
    For HF tokenizer, specify a folder /path/to/hf_folder which contains tokenizer.json, tokenizer_config.json
    and special_tokens_map.json

  - If your model or dataset requires non-default configs for conventional SFT/PEFT training in NeMo, you will
    need to pass in the same configs to ``model.data.train_ds`` as you would for training with unpacked dataset.

  - ``model.data.train_ds.max_seq_length`` is the length to truncate each sequence before packing multiple sequences
    to the size of packed sequence (``pack_size``). ``max_seq_length`` should be set to the same value as unpacked data,
    and can be determined by examining the distribution of sequence lengths in the dataset.

  - ``model.context_parallel_size`` is the CP size the model uses in SFT. The default value is 1 (no context parallelism)
    if not specified. This argument is necessary to make each individual sequence length in a packed sequence a multiple of CP*2
    when CP is enabled in SFT.

  - ``pack_sizes`` is a list of packed sequence lengths. In this example, there will be three output files, one for
    each pack size. The output files are named ``<output_folder>/packed_{pack_size}_seed{seed}.npy``.
    This argument is a list because you will likely want to experiment with a few ``pack_sizes`` to find out which length
    can fill the GPU memory without exceeding it. Adjusting ``pack_size`` is analogous to adjusting the micro batch size in
    the unpacked case.
"""


def tokenize_dataset(cfg: 'DictConfig'):
    """
    Tokenizes a dataset using the same configuration file as finetuninng with GPTSFTDataset.

    This function reads a dataset and tokenizes it using SentencePiece tokenizer based on the provided configuration.

    Args:
      cfg: A Hydra configuration object containing parameters for tokenization.

    Returns:
      A NumPy array containing the tokenized sequences from the dataset.
    """

    logging.info("Tokenizing dataset...")
    # using the same template as SFT/PEFT script. This may be overkill but guarantees the preprocess settings
    # are identical to normal SFT training
    data_cfg = cfg.model.data.train_ds
    pad_seq_length_to_mult = 16
    cp_size = cfg.model.get("context_parallel_size", 1)

    # if context parallel is used, each individual data length in one packed dataset sample
    # needs to be a multiple of (cp_size * 2): https://github.com/NVIDIA/TransformerEngine/pull/641
    if cp_size > 1:
        pad_seq_length_to_mult = max(pad_seq_length_to_mult, cp_size * 2)

    if os.path.isdir(cfg.tokenizer_path):
        # pass in a Hugging Face folder which contains tokenizer.json
        tokenizer = get_nmt_tokenizer(library="huggingface", model_name=cfg.tokenizer_path, use_fast=True)
    else:
        tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=cfg.tokenizer_path)

    dataset = GPTSFTDataset(
        file_path=data_cfg.file_names[0],
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        pad_seq_length_to_mult=pad_seq_length_to_mult,
        add_bos=data_cfg.get('add_bos', False),
        add_eos=data_cfg.get('add_eos', True),
        add_sep=data_cfg.get('add_sep', False),
        sep_id=cfg.get('sep_id', 49704),
        max_num_samples=None,
        seed=data_cfg.get('seed', 1234),
        label_key=data_cfg.get('label_key', 'answer'),
        answer_only_loss=cfg.get('answer_only_loss', True),
        truncation_field=data_cfg.get('truncation_field', 'text'),
        pad_to_max_length=data_cfg.get('pad_to_max_length', False),
        index_mapping_dir=data_cfg.get('index_mapping_dir', None),
        prompt_template=data_cfg.get('prompt_template', None),
        virtual_tokens=0,
        tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
        memmap_workers=data_cfg.get('memmap_workers', None),
        hf_dataset=data_cfg.get('hf_dataset', False),
        truncation_method=data_cfg.get('truncation_method', 'right'),
        special_tokens=data_cfg.get('chat_prompt_tokens', None),
        is_test=True,
    )

    max_seq_length = dataset.max_seq_length
    pad_id = dataset.tokenizer.eos_id
    tokenizer = dataset.tokenizer
    pad_seq_length_to_mult = dataset.pad_seq_length_to_mult
    dataset = np.array([dataset[i] for i in range(len(dataset))])
    if cp_size > 1:

        def pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id):
            '''
            pad each individual data point to the length of max_length
            '''
            assert max_seq_length >= max_length_to_pad
            for key, val in data.items():
                if key in {'input_ids', 'context_ids'}:
                    if len(val) <= max_length_to_pad:
                        # because input_ids are truncated by 1 for inputs and labels,
                        # we add 1 extra padding here to make sure padded inputs and labels
                        # are is a multiple of (cp_size * 2)
                        val = val + [pad_id] * (max_length_to_pad - len(val) + 1)
                        data[key] = val
                    elif len(val) > max_seq_length:
                        logging.info(
                            f"""The current sequence length {len(val)} for packing is
                                        larger than the max_seq_length specified ({max_seq_length}).
                                        The current seqquence length is truncated to the size of max_seq_length.
                                        Please consider increase the sequence packing size"""
                        )
                        data[key] = val[:max_seq_length]
            return

        ceil_to_nearest = lambda n, m: (n + m - 1) // m * m
        for data in dataset:
            max_length_to_pad = min(max_seq_length, ceil_to_nearest(len(data['input_ids']), pad_seq_length_to_mult))
            pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id)
    return dataset, tokenizer


@dataclass
class PackingArgs:
    output_dir: str = "output"
    pack_sizes: Tuple[int] = (2048,)
    packing_algorithm: str = "first_fit_shuffle"
    seed: int = 0

    def from_config(self, cfg: 'DictConfig'):
        for required_arg in ('output_dir', 'pack_sizes'):
            assert cfg.get(required_arg, None), f"Please specify +{required_arg}=..."
        self.output_dir = cfg.output_dir
        self.pack_sizes = cfg.pack_sizes
        self.packing_algorithm = cfg.get("packing_algorithm", "first_fit_shuffle")
        self.seed = cfg.get("seed", 0)
        return self


@hydra_runner(config_path=".", config_name="pack_config")
def main(cfg: 'DictConfig') -> None:
    args = PackingArgs().from_config(cfg)
    dataset, tokenizer = tokenize_dataset(cfg)
    sequences, histogram = create_hist(dataset, cfg.model.data.train_ds.max_seq_length)
    for pack_size in args.pack_sizes:
        assignments, metadata = create_packing_strategy(histogram, pack_size, args.packing_algorithm)
        output_data = fill_packing_strategy(assignments, sequences, pack_size, tokenizer.eos_id)

        # save output data
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'packed_{pack_size}_seed{args.seed}.npy')
        np.save(output_path, output_data)
        logging.info(f"Done, output written to {output_path}")

    logging.info(
        f"""
✅ Packed datasets with pack sizes {args.pack_sizes} are prepared successfully.
To train with packed sequences, you need to make changes to the SFT/PEFT config file. See NeMo Documentation
for more details: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/throughput_optimizations.html#sequence-packing-for-sft-peft>
"""
    )


if __name__ == '__main__':
    main()

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_partitions(file_path: str, num_workers: int):
    size = os.path.getsize(file_path)
    if size == 0:
        return []
    # Split file into roughly equal byte ranges
    step = max(1, math.ceil(size / max(1, num_workers)))
    partitions = []
    start = 0
    while start < size:
        end = min(size, start + step)
        partitions.append((start, end))
        start = end
    return partitions


def process_partition(file_path: str, start: int, end: int):
    results = []
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            if start != 0:
                # Align to next line start to avoid partial JSON line
                f.readline()
            while True:
                pos = f.tell()
                if pos >= end:
                    break
                line = f.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8", errors="ignore").strip()
                    if not s:
                        continue
                    d = json.loads(s)
                except Exception:
                    continue
                try:
                    if d.get("generation", {}).get("success"):
                        rdl = d["generation"]["results_dict_list"]
                        if "promt_turn_list_list" in rdl:
                            results.append(
                                {
                                    "problem": d["problem"],
                                    "promt_turn_list_list": rdl["promt_turn_list_list"],
                                    "prompt_turn_list": rdl.get("prompt_turn_list"),
                                    "full_prompt_turn_list": rdl.get("full_prompt_turn_list"),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "problem": d["problem"],
                                    "prompt_turn_list": rdl["prompt_turn_list"],
                                }
                            )
                except Exception:
                    continue
    except Exception:
        pass
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    data = []
    base_dir = args.data_path
    # Build list of files (supports directory or single file path)
    if os.path.isdir(base_dir):
        file_paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".jsonl")]
    else:
        file_paths = [base_dir] if base_dir.endswith(".jsonl") and os.path.isfile(base_dir) else []
    total_tiles = len(file_paths)

    # Parallelize across partitions of each file
    partitions = []
    for fp in file_paths:
        parts = compute_partitions(fp, args.num_workers or 1)
        for start, end in parts:
            partitions.append((fp, start, end))

    if partitions:
        with ProcessPoolExecutor(max_workers=args.num_workers or 1) as ex:
            futures = [ex.submit(process_partition, fp, s, e) for (fp, s, e) in partitions]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                part = fut.result()
                if part:
                    data.extend(part)

    print(f"Total tiles: {total_tiles}")
    # save as jsonl
    print(f"Saving to {args.output_path}")
    with open(args.output_path + "_raw.jsonl", "w") as f:
        for d in tqdm(data):
            f.write(json.dumps(d) + "\n")

    # postprocessing
    df = pd.DataFrame(data)
    df_grouped = df["promt_turn_list_list"].groupby(df["problem"]).apply(list)
    df_filtered = df_grouped  # df_grouped.apply(temp_func)   #filter out easy problems with high pass rate
    df_filtered_explode = df_filtered.explode().dropna().to_frame()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dict_list = []
    # iterate rows as dict
    df.reset_index(drop=True, inplace=True)
    for i in range(len(df_filtered_explode)):
        d = df_filtered_explode.iloc[i]
        for prompt_turn_list in d["promt_turn_list_list"]:
            # for dd in prompt_turn_list:
            #     dd['content']=dd['content'].replace('<|im_start|>assistant','').replace('<|im_start|>','')
            splits = tokenizer.apply_chat_template(prompt_turn_list, tokenize=False).split(
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            inputs = "<|im_end|>\n<|im_start|>assistant\n".join(splits[:-1]) + "<|im_end|>\n<|im_start|>assistant\n"
            outputs = splits[-1].replace("<|im_start|>assistant", "").replace("<|im_start|>", "")
            dict_list.append({"input": inputs, "output": outputs})

    with open(args.output_path + "_sft.jsonl", "w") as f:
        for d in dict_list:
            f.write(json.dumps(d) + "\n")

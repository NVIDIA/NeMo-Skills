#!/usr/bin/env python3

import sys
import os
import subprocess

# Check if at least one file is provided
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} output_file input_file1 [input_file2 ...]")
    sys.exit(1)

# Get the output file name
output_file = sys.argv[1]
input_files = sys.argv[2:]

# Check for .done files
for file_ in input_files:
    done_file = f"{file_}.done"
    if not os.path.isfile(done_file):
        print(f"Info: {done_file} not found. Skipping the rest of the script.")
        sys.exit(0)
    if not os.path.isfile(file_):
        print(f"Info: {file_} not found. Exiting.")
        sys.exit(0)

# Concatenate the files using subprocess
try:
    subprocess.run(["cat"] + input_files, stdout=open(output_file, "w"), check=True)
    print(f"Successfully concatenated {len(input_files)} files to {output_file}")

    # Create .done file and delete input files
    open(f"{output_file}.done", "w").close()
    for file_ in input_files:
        os.remove(file_)
except subprocess.CalledProcessError:
    print("An error occurred while concatenating files")
    sys.exit(1)

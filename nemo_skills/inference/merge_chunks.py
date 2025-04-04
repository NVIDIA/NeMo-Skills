#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys


def unescape_shell_command(command: str) -> str:
    """Unescape special shell characters so they are correctly interpreted before execution."""
    command = command.strip() if command else ""
    return shlex.split(command)


# Check if at least one input file and an output file are provided
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} output_file input_file1 [input_file2 ...] [-- command_to_run]")
    sys.exit(1)

# Separate file arguments from optional command
if "--" in sys.argv:
    sep_index = sys.argv.index("--")
    output_file = sys.argv[1]
    input_files = sys.argv[2:sep_index]
    post_merge_command = " ".join(sys.argv[sep_index + 1 :])
    post_merge_command = unescape_shell_command(post_merge_command)
else:
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    post_merge_command = None

# Check for .done files and actual input files
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
    with open(output_file, "w") as out_f:
        subprocess.run(["cat"] + input_files, stdout=out_f, check=True)
    print(f"Successfully concatenated {len(input_files)} files to {output_file}")

    # Create .done file and delete input files
    open(f"{output_file}.done", "w").close()
    for file_ in input_files:
        os.remove(file_)

    # Execute the post-merge command, if provided
    if post_merge_command:
        print(f"Executing post-merge command: {''.join(post_merge_command)}")
        print(post_merge_command)
        subprocess.run(post_merge_command, shell=True, check=True)
        print("Done")

except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

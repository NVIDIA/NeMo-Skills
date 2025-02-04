#!/bin/bash

# Check if at least one file is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 output_file input_file1 [input_file2 ...]"
    exit 1
fi

# Get the output file name
output_file="$1"
shift

# Check for .done files
for file in "$@"; do
    done_file="${file}.done"
    if [ ! -f "$done_file" ]; then
        echo "Error: $done_file not found. Skipping the rest of the script."
        exit 1
    fi
done

# Concatenate the files
cat "$@" > "$output_file"

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo "Successfully concatenated $# files to $output_file"
else
    echo "An error occurred while concatenating files"
    exit 1
fi

#!/bin/bash

# Ensure at least one argument is provided.
if [ "$#" -lt 1 ]; then
    echo "This script converts the latest DeepSpeed ZeRO checkpoint to a universal checkpoint."
    echo "Usage: $0 <CKPT_PATH> [additional arguments for deepspeed.checkpoint.ds_to_universal]"
    exit 1
fi

# Set CKPT_PATH to the first argument and shift it out so that "$@" contains the extra arguments.
CKPT_PATH="$1"
shift
EXTRA_ARGS="$@"

# Function to process a given directory.
process_dir() {
    local path="$1"
    echo "Processing checkpoint: $path"
    
    # Check if the latest tag exists.
    if [ ! -f "$path/latest" ]; then
        echo "latest tag file not found in $path, ensure the directory contains a valid DeepSpeed ZeRO checkpoint."
        return 1
    fi

    # Read the latest tag.
    LATEST_TAG=$(cat "$path/latest")
    LATEST_UNI_TAG="${LATEST_TAG}_uni"
    
    # Write the universal tag.
    echo "$LATEST_UNI_TAG" > "$path/latest_universal"
    
    # Run the python command with any additional arguments.
    python -m deepspeed.checkpoint.ds_to_universal --inject_missing_state \
        --input_folder "$path/$LATEST_TAG" \
        --output_folder "$path/$LATEST_UNI_TAG" \
        $EXTRA_ARGS
}

# Flag to check if at least one of the specific subdirectories exists.
found_subdir=0

## For PPO, checkpoints for each model are stored under "_actor" and "_critic" separately.
# Check for the subdirectory named exactly "_actor".
if [ -d "$CKPT_PATH/_actor" ]; then
    process_dir "$CKPT_PATH/_actor"
    found_subdir=1
fi

# Check for the subdirectory named exactly "_critic".
if [ -d "$CKPT_PATH/_critic" ]; then
    process_dir "$CKPT_PATH/_critic"
    found_subdir=1
fi

# If neither subdirectory exists, process the main CKPT_PATH.
if [ "$found_subdir" -eq 0 ]; then
    process_dir "$CKPT_PATH"
fi

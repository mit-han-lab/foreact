#!/bin/bash

source scripts/setup.sh

# First arg: yaml filename inside configs/ (default: finetune.yaml).
# Second arg: run_name (default: yaml basename without extension).
CONFIG_FILE="${1:-finetune.yaml}"
RUN_NAME="${2:-$(basename "$CONFIG_FILE" .yaml)}"

torchrun $TORCHRUN_ARGS train.py \
    --run_name "$RUN_NAME" \
    --config_file "$CONFIG_FILE"
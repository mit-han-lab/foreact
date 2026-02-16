#!/bin/bash

source scripts/setup.sh

torchrun $TORCHRUN_ARGS train.py \
    --run_name run_finetune \
    --config_file finetune.yaml
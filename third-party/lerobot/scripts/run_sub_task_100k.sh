#!/bin/bash

source scripts/setup.sh

JOB_NAME="pi05_realworld_sub_task_100k"
JOB_DIR="./$JOB_NAME"

DATASET_ROOT="./ForeAct_VLA_Dataset"

if [ -d "$JOB_DIR/checkpoints/last" ]; then
    torchrun $TORCHRUN_ARGS -m lerobot.scripts.lerobot_train \
        --config_path=$JOB_DIR/checkpoints/last/pretrained_model/train_config.json \
        --resume=true
    exit 0
fi

torchrun $TORCHRUN_ARGS -m lerobot.scripts.lerobot_train \
    --policy.type=pi05 \
    --job_name=$JOB_NAME \
    --output_dir=$JOB_DIR \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.use_amp=false \
    --dataset.repo_id="+" \
    --dataset.root=$DATASET_ROOT \
    --dataset.ignore_features='["observation.images.head_right_rgb"]' \
    --use_policy_training_preset=false \
    --seed=42 \
    --optimizer.type=adamw \
    --optimizer.lr=5e-5 \
    --optimizer.betas=[0.9,0.95] \
    --optimizer.weight_decay=0 \
    --optimizer.grad_clip_norm=1.0 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=1000 \
    --scheduler.num_decay_steps=100000 \
    --scheduler.peak_lr=5e-5 \
    --scheduler.decay_lr=5e-6 \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=100000 \
    --log_freq=100 \
    --save_freq=2500 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    
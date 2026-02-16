#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD
from lerobot.datasets.compute_stats import aggregate_stats
import os

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            if key in cfg.goal_condition_features:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps



def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
        if "+" in cfg.dataset.repo_id:
            repo_ids = ['20251108_Sort_Blocks', '20251105_Sort_Mask', '20251106_Pick_Tool', '20251104_Pick_Rubbish_Left', '20251104_Stack_Bowls', '20251109_Pick_Flower', '20251108_Pen_Drawer', '20251106_Toolkit', '20251105_Pick_Rubbish_Dual', '20251108_Sort_Study_Desk', '20251108_Pick_Pen', '20251102_Pick_Veg', '20251104_Pick_Rubbish', '20251105_Same_Color_Bowl_Plate']
            ds_meta = LeRobotDatasetMetadata(
                repo_ids[0], root=f"{cfg.dataset.root}/{repo_ids[0]}", revision=cfg.dataset.revision, ignore_features=cfg.dataset.ignore_features
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            datasets = [
                LeRobotDataset(
                    repo_id,
                    root=f"{cfg.dataset.root}/{repo_id}",
                    episodes=cfg.dataset.episodes,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    revision=cfg.dataset.revision,
                    video_backend=cfg.dataset.video_backend,
                    ignore_features=cfg.dataset.ignore_features,
                )
                for repo_id in repo_ids
            ]
            # Aggregate stats first before subsampling
            aggregated_stats = aggregate_stats([dataset.meta.stats for dataset in datasets])
            
            # Calculate num_frames and num_episodes based on data_percentage
            total_num_frames = 0
            total_num_episodes = 0
            subsampled_datasets = []
            
            for dataset in datasets:
                num_samples = int(dataset.num_frames * cfg.dataset.data_percentage)
                total_num_frames += num_samples
                total_num_episodes += dataset.num_episodes
                
                # Update stats on the original dataset before wrapping
                dataset.meta.stats = aggregated_stats
                
                # Create Subset to limit dataset size
                indices = list(range(num_samples))
                subsampled_dataset = torch.utils.data.Subset(dataset, indices)
                subsampled_datasets.append(subsampled_dataset)
            # Apply ImageNet stats if needed
            if cfg.dataset.use_imagenet_stats:
                for dataset in datasets:
                    for key in dataset.meta.camera_keys:
                        if key not in dataset.meta.stats:
                            dataset.meta.stats[key] = {}
                        for stats_type, stats in IMAGENET_STATS.items():
                            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
            
            # Concatenate the subsampled datasets
            dataset = torch.utils.data.ConcatDataset(subsampled_datasets)
            dataset.num_frames = total_num_frames
            dataset.num_episodes = total_num_episodes
            dataset.meta = datasets[0].meta
        else:
            ds_meta = LeRobotDatasetMetadata(
                cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision, ignore_features=cfg.dataset.ignore_features
            )
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
                ignore_features=cfg.dataset.ignore_features,
            )
            if cfg.dataset.use_imagenet_stats:
                for key in dataset.meta.camera_keys:
                    if key not in dataset.meta.stats:
                        dataset.meta.stats[key] = {}
                    for stats_type, stats in IMAGENET_STATS.items():
                        dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    # if cfg.dataset.use_imagenet_stats:
    #     for key in dataset.meta.camera_keys:
    #         if key not in dataset.meta.stats:
    #             dataset.meta.stats[key] = {}
    #         for stats_type, stats in IMAGENET_STATS.items():
    #             dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
This script augments existing LeRobot datasets with quantile statistics.

Most datasets created before the quantile feature was added do not contain
quantile statistics (q01, q10, q50, q90, q99) in their metadata. This script:

1. Loads an existing LeRobot dataset in v3.0 format
2. Checks if it already contains quantile statistics
3. If missing, computes quantile statistics for all features
4. Updates the dataset metadata with the new quantile statistics

Usage:

```bash
# Basic usage (auto-detect number of workers based on CPU count)
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht

# With explicit number of parallel workers
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
    --repo-id=lerobot/pusht \
    --num-workers=8
```
"""

import argparse
import concurrent.futures
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from requests import HTTPError
from tqdm import tqdm

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import write_stats
from lerobot.utils.utils import init_logging


def has_quantile_stats(stats: dict[str, dict] | None, quantile_list_keys: list[str] | None = None) -> bool:
    """Check if dataset statistics already contain quantile information.

    Args:
        stats: Dataset statistics dictionary

    Returns:
        True if quantile statistics are present, False otherwise
    """
    if quantile_list_keys is None:
        quantile_list_keys = [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]

    if stats is None:
        return False

    for feature_stats in stats.values():
        if any(q_key in feature_stats for q_key in quantile_list_keys):
            return True

    return False


def _process_episode_in_worker(
    episode_idx: int,
    repo_id: str,
    root: str | Path | None,
) -> tuple[int, dict]:
    """Process a single episode in a separate worker process.

    This function is designed to be called from ProcessPoolExecutor.
    It loads the dataset independently in each worker process to avoid
    serialization issues and ensure thread-safety for video decoding.

    Args:
        episode_idx: Index of the episode to process
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset

    Returns:
        Tuple of (episode_idx, episode_stats)
    """
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    ep_stats = process_single_episode(dataset, episode_idx)
    return episode_idx, ep_stats


def process_single_episode(dataset: LeRobotDataset, episode_idx: int) -> dict:
    """Process a single episode and return its statistics.

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to process

    Returns:
        Dictionary containing episode statistics
    """
    logging.debug(f"Computing stats for episode {episode_idx}")

    start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    collected_data: dict[str, list] = {}
    for idx in range(start_idx, end_idx):
        item = dataset[idx]
        for key, value in item.items():
            if key not in dataset.features:
                continue

            if key not in collected_data:
                collected_data[key] = []
            collected_data[key].append(value)

    ep_stats = {}
    for key, data_list in collected_data.items():
        if dataset.features[key]["dtype"] == "string":
            continue

        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0

            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
        )

        if dataset.features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def compute_quantile_stats_for_dataset(
    dataset: LeRobotDataset,
    repo_id: str,
    root: str | Path | None = None,
    num_workers: int | None = None,
) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Args:
        dataset: The LeRobot dataset to compute statistics for
        repo_id: Repository ID of the dataset (needed for multiprocessing)
        root: Local root directory for the dataset (needed for multiprocessing)
        num_workers: Number of parallel workers. Defaults to min(num_episodes, cpu_count)

    Returns:
        Dictionary containing aggregated statistics with quantiles

    Note:
        Uses multiprocessing (ProcessPoolExecutor) to parallelize episode processing.
        Each worker process loads its own copy of the dataset, which ensures thread-safety
        for video decoding operations.
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    num_episodes = dataset.num_episodes

    if num_workers is None:
        num_workers = min(num_episodes, os.cpu_count() or 4)
    num_workers = max(1, min(num_workers, num_episodes))

    episode_stats_list = []

    if num_workers == 1:
        logging.info("Using single worker (sequential processing)")
        for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
            ep_stats = process_single_episode(dataset, episode_idx)
            episode_stats_list.append(ep_stats)
    else:
        logging.info(f"Using {num_workers} parallel workers (multiprocessing)")

        worker_fn = partial(_process_episode_in_worker, repo_id=repo_id, root=root)

        episode_results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_episode = {
                executor.submit(worker_fn, episode_idx): episode_idx
                for episode_idx in range(num_episodes)
            }

            with tqdm(total=num_episodes, desc="Processing episodes") as pbar:
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_idx, ep_stats = future.result()
                    episode_results[episode_idx] = ep_stats
                    pbar.update(1)

        for episode_idx in range(num_episodes):
            episode_stats_list.append(episode_results[episode_idx])

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Aggregating statistics from {len(episode_stats_list)} episodes")
    return aggregate_stats(episode_stats_list)


def augment_dataset_with_quantile_stats(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    num_workers: int | None = None,
) -> None:
    """Augment a dataset with quantile statistics if they are missing.

    Args:
        repo_id: Repository ID of the dataset
        root: Local root directory for the dataset
        overwrite: Overwrite existing quantile statistics if they already exist
        num_workers: Number of parallel workers for computing statistics
    """
    logging.info(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )

    if not overwrite and has_quantile_stats(dataset.meta.stats):
        logging.info("Dataset already contains quantile statistics. No action needed.")
        return

    logging.info("Dataset does not contain quantile statistics. Computing them now...")

    new_stats = compute_quantile_stats_for_dataset(
        dataset, repo_id=repo_id, root=root, num_workers=num_workers
    )

    logging.info("Updating dataset metadata with new quantile statistics")
    dataset.meta.stats = new_stats

    write_stats(new_stats, dataset.meta.root)

    logging.info("Successfully updated dataset with quantile statistics")
    dataset.push_to_hub()

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        logging.info(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        pass
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=None, repo_type="dataset")


def main():
    """Main function to run the augmentation script."""
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with quantile statistics")

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset (e.g., 'lerobot/pusht')",
    )

    parser.add_argument(
        "--root",
        type=str,
        help="Local root directory for the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile statistics if they already exist",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for computing statistics. Defaults to min(num_episodes, cpu_count)",
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else None

    init_logging()

    augment_dataset_with_quantile_stats(
        repo_id=args.repo_id,
        root=root,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

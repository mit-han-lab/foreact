import json
import time
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import v2
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from utils.video_utils import decode_video_frames, get_safe_default_codec


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        camera_key: str = "observation.images.head_left_rgb",
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.camera_key = camera_key
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        print(f"video_backend: {self.video_backend}")

        # Load metadata
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing info.json at: {info_path}")
        if not episodes_path.is_file():
            raise FileNotFoundError(f"Missing episodes.jsonl at: {episodes_path}")

        with open(info_path, "r", encoding="utf-8") as f:
            self.info: Dict[str, Any] = json.load(f)

        self.fps: int = int(self.info.get("fps", 30))
        self.chunks_size: int = int(self.info.get("chunks_size", 1000))
        self.video_path_template: Optional[str] = self.info.get("video_path")
        self.features: Dict[str, Dict[str, Any]] = self.info.get("features", {})

        # Validate camera key
        if self.camera_key not in self.features:
            raise KeyError(
                f"Camera key '{self.camera_key}' not found in features. Available: {list(self.features.keys())}"
            )
        if self.features[self.camera_key].get("dtype") != "video":
            raise ValueError(
                f"Camera key '{self.camera_key}' is not stored as video. dtype={self.features[self.camera_key].get('dtype')}"
            )

        # Load episodes list
        self._episodes: List[Dict[str, Any]] = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                # Ensure required fields exist
                if "episode_index" not in ep or "length" not in ep:
                    continue
                if ep.get("tasks") == [""]:
                    continue
                self._episodes.append(ep)

        if len(self._episodes) == 0:
            raise RuntimeError("No episodes loaded. Check 'episodes' filter or dataset contents.")

        # Build global index -> (episode_idx_in_list, local_frame_idx)
        self._index: List[Tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            length = int(ep["length"])  # number of frames
            self._index.extend([(i, fi) for fi in range(length) if fi % self.fps == 0])

    def __len__(self) -> int:
        return len(self._index)

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def _video_path_for_episode(self, episode_index: int) -> Path:
        if not self.video_path_template:
            chunk = self._episode_chunk(episode_index)
            rel = f"videos/chunk-{chunk:03d}/{self.camera_key}/episode_{episode_index:06d}.mp4"
            return self.root / rel
        fpath = self.video_path_template.format(
            episode_chunk=self._episode_chunk(episode_index),
            video_key=self.camera_key,
            episode_index=episode_index,
        )
        return self.root / fpath

    # ---------- core decoding ----------
    def _decode_pair(
        self, video_path: Path, ts_source: float, ts_target: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = decode_video_frames(
            str(video_path),
            [ts_source, ts_target],
            self.tolerance_s,
            self.video_backend,
        )
        source, target = frames[0], frames[1]

        return source, target

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_pos, local_fi = self._index[idx]
        ep = self._episodes[ep_pos]
        episode_index: int = int(ep["episode_index"])
        ep_len: int = int(ep["length"])

        ts_source = local_fi / float(self.fps)
        # Always use the last frame as the target frame
        target_fi = ep_len - 1 
        ts_target = target_fi / float(self.fps)

        video_path = self._video_path_for_episode(episode_index)
        source_img, target_img = self._decode_pair(video_path, ts_source, ts_target)

        tasks = ep.get("tasks", [])
        if isinstance(tasks, list):
            caption = "; ".join([str(t) for t in tasks]) if len(tasks) > 0 else ""
        else:
            caption = str(tasks)

        to_pil = v2.ToPILImage()

        return {
            "source_image": to_pil(source_img),
            "target_image": to_pil(target_img),
            "caption": caption,
        }


def _collate_fn_imagepair(batch, tokenize_func, tokenizer, source_transform, target_transform):
    captions = [example["caption"] for example in batch]
    source_images = [example["source_image"] for example in batch]
    target_images = [example["target_image"] for example in batch]

    rand_probs = torch.rand((len(source_images), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)

    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    source_images = [
        (
            Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(source_images)
    ]

    sources = [source_transform(image) for image in source_images]
    targets = [target_transform(image) for image in target_images]

    return_dict = {"source": torch.stack(sources), "target": torch.stack(targets)}
    (
        return_dict["input_ids"],
        return_dict["attention_mask"],
    ) = tokenize_func(tokenizer, captions)

    return return_dict


def _load_single_dataset(repo_id, root, camera_key):
    try:
        dataset = ImagePairDataset(
            root=root,
            camera_key=camera_key,
        )
        print(f"✓ Loaded dataset: {repo_id}")
        return repo_id, dataset
    except Exception as e:
        print(f"✗ Failed to load dataset {repo_id}: {e}")
        return repo_id, None


def get_train_datasets(data_args, tokenize_func, tokenizer):
    train_datasets = {}
    
    # Prepare dataset loading tasks
    dataset_tasks = []
    
    import os
    data_path = data_args.data_path
    camera_key = data_args.camera_key
    for dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir)):
            dataset_tasks.append((
                dir,
                os.path.join(data_path, dir),
                camera_key,
            ))
    # Load datasets in parallel using ThreadPoolExecutor
    print(f"Loading {len(dataset_tasks)} datasets in parallel...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(32, len(dataset_tasks))) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_load_single_dataset, repo_id, root, camera_key): (repo_id, root, camera_key)
            for repo_id, root, camera_key in dataset_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            repo_id, dataset = future.result()
            if dataset is not None:
                train_datasets[repo_id] = dataset
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(train_datasets)}/{len(dataset_tasks)} datasets in {elapsed_time:.2f} seconds")

    source_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    
    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    # Use a custom collate function for the torch Dataset
    collate_fn = partial(
        _collate_fn_imagepair,
        tokenize_func=tokenize_func,
        tokenizer=tokenizer,
        source_transform=source_transform,
        target_transform=target_transform,
    )

    train_dataset = ConcatDataset(list(train_datasets.values()))

    return train_dataset, collate_fn
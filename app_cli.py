import argparse
import numpy as np
import os
import random
import torch

from datetime import datetime
from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint
from PIL import Image


def process(
    pipeline,
    prompt,
    output_dir,
    input_image_path=None,
    negative_prompt="",
    seed=None,
    guidance_scale=4.5,
    image_guidance_scale=1.5,
    num_inference_steps=8,
    num_images_per_prompt=1,
):
    """Generate images using the VisualForesight pipeline"""

    # Set random seed
    if seed is None:
        seed = random.randint(0, np.iinfo(np.int32).max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\nGenerating images with seed: {seed}")
    print(f"Prompt: {prompt}")
    if negative_prompt:
        print(f"Negative prompt: {negative_prompt}")

    # Load input image if provided
    input_image = None
    if input_image_path:
        input_image = Image.open(input_image_path).convert('RGB')
        print(f"Loaded input image: {input_image_path}")

    # Generate images
    print("\nGenerating...")
    images = pipeline(
        image=input_image,
        caption=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        generator=torch.Generator().manual_seed(seed),
        enable_progress_bar=True,
    ).images

    # Save generated images
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    for idx, img in enumerate(images):
        filename = f"generated_{timestamp}_seed{seed}_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")

    print(f"\nGenerated {len(saved_paths)} image(s)")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using VisualForesight pipeline")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images (default: ./outputs)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (random if not specified)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Guidance scale (default: 4.5)"
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.5,
        help="Image guidance scale (default: 1.5)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=8,
        help="Number of inference steps (default: 30)"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )

    args = parser.parse_args()

    print("Loading model...")
    pipeline = VisualForesightPipeline.from_pretrained(
        find_newest_checkpoint(args.checkpoint_path),
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    print("Model loaded successfully!\n")

    process(
        pipeline=pipeline,
        prompt=args.prompt,
        output_dir=args.output_dir,
        input_image_path=args.input_image,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
    )

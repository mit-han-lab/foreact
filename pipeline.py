import numpy as np
import torch

from dataclasses import dataclass
from diffusers.pipelines.pipeline_utils import BaseOutput
from models.visualforesight import VisualForesight
from typing import List, Optional, Tuple, Union
from PIL import Image


@dataclass
class VisualForesightPipelineOutput(BaseOutput):
    images: Union[List[Image.Image], np.ndarray]


class VisualForesightPipeline(VisualForesight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        caption: str = "",
        image: Optional[Image.Image] = None,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 8,
        num_images_per_prompt: int = 1,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[VisualForesightPipelineOutput, Tuple]:

        samples = self.sample_images(
            caption=caption,
            input_image=image,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        # Resize generated samples to match input image size
        if image is not None and isinstance(samples, list):
            target_size = image.size  # (width, height)
            resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
            samples = [
                s.resize(target_size, resample=resample) if isinstance(s, Image.Image) and s.size != target_size else s
                for s in samples
            ]

        if not return_dict:
            return (samples,)
        return VisualForesightPipelineOutput(images=samples)

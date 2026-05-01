import numpy as np
import torch

from dataclasses import dataclass
from diffusers.pipelines.pipeline_utils import BaseOutput
from models.visualforesight import VisualForesight
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image


@dataclass
class VisualForesightPipelineOutput(BaseOutput):
    # Maps view_name -> list of PIL images (one per num_images_per_prompt).
    images: Union[Dict[str, List[Image.Image]], np.ndarray]


class VisualForesightPipeline(VisualForesight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        caption: str = "",
        images: Optional[Dict[str, Image.Image]] = None,
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
            input_images=images,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )

        # Resize generated samples per view to match its input image size.
        if isinstance(samples, dict) and images:
            resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
            resized: Dict[str, List[Image.Image]] = {}
            for view_name, view_samples in samples.items():
                ref = images.get(view_name)
                if ref is not None and isinstance(view_samples, list):
                    target_size = ref.size  # (width, height)
                    resized[view_name] = [
                        s.resize(target_size, resample=resample)
                        if isinstance(s, Image.Image) and s.size != target_size
                        else s
                        for s in view_samples
                    ]
                else:
                    resized[view_name] = view_samples
            samples = resized

        if not return_dict:
            return (samples,)
        return VisualForesightPipelineOutput(images=samples)

import numpy as np
import PIL
import torch

from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Optional, Union, List

from diffusers.models import AutoencoderDC
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import randn_tensor

from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM

from .sana import SanaTransformer2DModel


class VisualForesightConfig(PretrainedConfig):
    model_type = "visualforesight"

    def __init__(
        self,
        mllm_id: str = "google/gemma-2-2b-it",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        max_input_text_tokens: int = 256,
        vae_downsample_f: int = 32,
        input_size: tuple[int, int] = (15, 20),
        in_channels: int = 32,
        system_prompt: str = "You are a robot and should focus on your actions. Generate a new image that meets the user's instruction while maintaining consistency with the original input where appropriate.",
        _gradient_checkpointing: bool = True,
        modules_to_freeze: tuple[str] = (),
        modules_to_unfreeze: tuple[str] = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.vae_id = vae_id
        self.noise_scheduler_id = noise_scheduler_id
        self.scheduler_id = scheduler_id

        self.max_input_text_tokens = max_input_text_tokens
        self.vae_downsample_f = vae_downsample_f
        self.input_size = input_size
        self.in_channels = in_channels
        self.system_prompt = system_prompt

        self._gradient_checkpointing = _gradient_checkpointing
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_unfreeze = modules_to_unfreeze


class VisualForesight(PreTrainedModel):
    config_class = VisualForesightConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config

        # --- MLLM backbone ---
        self.mllm_backbone = AutoModelForCausalLM.from_pretrained(
            config.mllm_id, attn_implementation="sdpa", torch_dtype=torch.bfloat16
        )
        self.mllm_backbone.lm_head = nn.Identity()

        self.tokenizer = AutoTokenizer.from_pretrained(config.mllm_id)
        self.tokenizer.padding_side = "left"
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.system_prompt = config.system_prompt

        # --- Diffusion transformer ---
        if "Sana" in config.diffusion_model_id:
            self.transformer = SanaTransformer2DModel.from_pretrained(
                config.diffusion_model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
        else:
            raise ValueError(f"Unsupported diffusion model: {config.diffusion_model_id}")

        # --- VAE ---
        if "Sana" in config.vae_id:
            self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae", torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unsupported vae: {config.vae_id}")

        # --- Schedulers ---
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.noise_scheduler_id, subfolder="scheduler"
        )
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.scheduler_id, subfolder="scheduler"
        )

        # --- Gradient checkpointing ---
        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except Exception:
                pass
            self.transformer.enable_gradient_checkpointing()

        # --- Freeze / unfreeze modules ---
        for module_name in config.modules_to_freeze:
            self._set_module_requires_grad(module_name, False)

        for module_name in config.modules_to_unfreeze:
            self._set_module_requires_grad(module_name, True)

    def _set_module_requires_grad(self, module_name: str, requires_grad: bool):
        """Resolve a dotted module name and set requires_grad."""
        module = self
        for part in module_name.split("."):
            module = getattr(module, part, None)
            if module is None:
                return
        module.requires_grad_(requires_grad)

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    @staticmethod
    @torch.no_grad()
    def tokenize(tokenizer, caption):
        if not isinstance(caption, List):
            caption = [caption]

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{tokenizer.system_prompt}\n{cap}"}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for cap in caption
        ]

        tokens = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.max_input_text_tokens,
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def encode_condition(self, input_ids, attention_mask, **kwargs):
        prompt_embeds = self.mllm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        return prompt_embeds, attention_mask

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
        self, source, target, input_ids=None, attention_mask=None, **kwargs
    ):
        latents = self.vae.encode(target).latent
        source_latents = self.vae.encode(source).latent

        latents = latents * self.vae.config.scaling_factor
        source_latents = source_latents * self.vae.config.scaling_factor

        bsz = latents.shape[0]

        noise = torch.randn_like(latents, device=latents.device)

        weighting_scheme = "uniform"
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(
            device=latents.device
        )

        sigmas = self.get_sigmas(
            timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
        )
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

        prompt_embeds, attention_mask = self.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        model_pred = self.transformer(
            hidden_states=noisy_latents,
            source_hidden_states=source_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample

        target = noise - latents
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme, sigmas=sigmas
        )
        loss = torch.mean(
            (
                weighting.float() * (model_pred.float() - target.float()) ** 2
            ).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return {"loss": loss}

    @torch.no_grad()
    def decode_latents(self, latents, normalize=True, return_tensor=False):
        latents = latents / self.vae.config.scaling_factor
        samples = self.vae.decode(latents).sample
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def sample_images(
        self,
        caption: str = "",
        input_image: PIL.Image.Image = None,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 8,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        negative_prompt: str = "",
        enable_progress_bar=False,
        **kwargs,
    ):
        device = next(self.parameters()).device
        latent_size = self.config.input_size
        do_image_cfg = input_image is not None and image_guidance_scale > 1.0

        # --- Text conditioning (with classifier-free guidance) ---
        if do_image_cfg:
            captions = [negative_prompt, negative_prompt, caption]
        else:
            captions = [negative_prompt, caption]

        input_ids, attention_mask = self.tokenize(self.tokenizer, captions)
        input_ids = input_ids.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        attention_mask = attention_mask.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds, attention_mask = self.encode_condition(
            input_ids=input_ids, attention_mask=attention_mask,
        )

        # --- Source image → VAE latents ---
        source_transform = v2.Compose([
            v2.Resize((latent_size[0] * self.config.vae_downsample_f,
                        latent_size[1] * self.config.vae_downsample_f)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ])
        source = source_transform(input_image).unsqueeze(0).to(device, dtype=self.vae.dtype)
        source_latent = self.vae.encode(source).latent * self.vae.config.scaling_factor

        if do_image_cfg:
            black_latent = self.vae.encode(torch.zeros_like(source)).latent * self.vae.config.scaling_factor
            source_latents = torch.cat([black_latent, source_latent, source_latent])
        else:
            source_latents = source_latent.repeat(2, 1, 1, 1)
        source_latents = source_latents.repeat_interleave(num_images_per_prompt, dim=0)

        # --- Initial noise ---
        latents = randn_tensor(
            shape=(num_images_per_prompt, self.config.in_channels, latent_size[0], latent_size[1]),
            generator=generator, device=device, dtype=torch.float32,
        )

        # --- Scheduler ---
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        # --- Denoising loop ---
        n_cond = len(captions)
        for t in tqdm(self.scheduler.timesteps, desc="Sampling", disable=not enable_progress_bar):
            latent_input = latents.repeat(n_cond, 1, 1, 1).to(prompt_embeds.dtype)
            if hasattr(self.scheduler, "scale_model_input"):
                latent_input = self.scheduler.scale_model_input(latent_input, t)

            noise_pred = self.transformer(
                hidden_states=latent_input,
                source_hidden_states=source_latents.to(prompt_embeds.dtype),
                timestep=t.unsqueeze(0).expand(latent_input.shape[0]).to(device),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            ).sample

            if do_image_cfg:
                pred_uncond, pred_image, pred_full = noise_pred.chunk(3)
                noise_pred = (
                    pred_uncond
                    + image_guidance_scale * (pred_image - pred_uncond)
                    + guidance_scale * (pred_full - pred_image)
                )
            else:
                pred_uncond, pred_full = noise_pred.chunk(2)
                noise_pred = pred_uncond + guidance_scale * (pred_full - pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.decode_latents(
            latents.to(self.vae.dtype) if self.vae is not None else latents,
            return_tensor=return_tensor,
        )

import copy
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
        view_names: Optional[tuple] = None,
        view_latent_sizes: Optional[tuple] = None,
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
        self.input_size = tuple(input_size)
        self.in_channels = in_channels
        self.system_prompt = system_prompt

        # Backward compat: older checkpoints have neither field; treat as a
        # single-view "primary" model whose latent size is `input_size`.
        if view_names is None:
            view_names = ("primary",)
        if view_latent_sizes is None:
            view_latent_sizes = (self.input_size,)
        self.view_names = tuple(view_names)
        self.view_latent_sizes = tuple(tuple(s) for s in view_latent_sizes)
        assert len(self.view_names) == len(self.view_latent_sizes), (
            f"view_names ({len(self.view_names)}) and view_latent_sizes "
            f"({len(self.view_latent_sizes)}) must match."
        )

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

        # --- View-id embedding ---
        # Only create when there is more than one view: a single-view model
        # has nothing to disambiguate, and skipping the param keeps SV
        # checkpoints free of any view-related state (so loading a legacy SV
        # checkpoint produces no "newly initialized" warning).
        # Zero-init when present so a pretrained MV-architecture checkpoint
        # with this param missing produces byte-identical outputs at step 0.
        self.view_names = list(config.view_names)
        self.view_to_id = {name: i for i, name in enumerate(self.view_names)}
        if len(self.view_names) > 1:
            t_cfg = self.transformer.config
            inner_dim = t_cfg.num_attention_heads * t_cfg.attention_head_dim
            self.view_embedding = nn.Embedding(len(self.view_names), inner_dim)
            self.view_embedding.weight.data.zero_()
        else:
            self.view_embedding = None

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

    def _init_weights(self, module):
        # Called by HuggingFace from_pretrained's _fast_init for every module
        # whose weights are missing from the checkpoint. Submodules owned by
        # mllm_backbone / transformer / vae manage their own init; we only
        # need to handle our new view_embedding.
        if isinstance(module, nn.Embedding) and module is self.view_embedding:
            module.weight.data.zero_()

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

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Collect per-view source/target image tensors from the flattened
        # collate output (`source_{view}`, `target_{view}`) in the order the
        # model was configured with.
        source_views = [kwargs.pop(f"source_{n}") for n in self.view_names]
        target_views = [kwargs.pop(f"target_{n}") for n in self.view_names]

        # VAE-encode each view independently. Weights are shared; the VAE is
        # fully convolutional so different H/W per view is fine.
        target_latents_list = [
            self.vae.encode(t).latent * self.vae.config.scaling_factor
            for t in target_views
        ]
        source_latents_list = [
            self.vae.encode(s).latent * self.vae.config.scaling_factor
            for s in source_views
        ]

        bsz = target_latents_list[0].shape[0]
        device = target_latents_list[0].device

        # One shared timestep per batch element, independent noise per view.
        weighting_scheme = "uniform"
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=device)

        noisy_latents_list = []
        noise_list = []
        sigmas_list = []
        for lat in target_latents_list:
            noise = torch.randn_like(lat, device=lat.device)
            sigmas = self.get_sigmas(
                timesteps, lat.device, n_dim=lat.ndim, dtype=lat.dtype
            )
            noisy_latents_list.append((1.0 - sigmas) * lat + sigmas * noise)
            noise_list.append(noise)
            sigmas_list.append(sigmas)

        prompt_embeds, attention_mask = self.encode_condition(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = self.transformer(
            hidden_states=noisy_latents_list,
            source_hidden_states=source_latents_list,
            view_ids=[self.view_to_id[n] for n in self.view_names],
            view_embedding=self.view_embedding,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
        ).sample
        if torch.is_tensor(preds):
            preds = [preds]

        # Per-view flow-matching loss, averaged across views.
        per_view_losses = []
        for pred, tgt_lat, noise, sigmas in zip(
            preds, target_latents_list, noise_list, sigmas_list
        ):
            fm_target = noise - tgt_lat
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=weighting_scheme, sigmas=sigmas
            )
            view_loss = torch.mean(
                (
                    weighting.float() * (pred.float() - fm_target.float()) ** 2
                ).reshape(fm_target.shape[0], -1),
                1,
            )
            per_view_losses.append(view_loss.mean())

        loss = torch.stack(per_view_losses).mean()
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
        input_images: Optional[dict] = None,
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
        view_latent_sizes = list(self.config.view_latent_sizes)
        f = self.config.vae_downsample_f

        if input_images is None:
            input_images = {}
        # Fill missing views with blank images so image CFG still has a
        # well-defined "null" branch even when the caller only provides some.
        per_view_pil = {}
        for name, latent_size in zip(self.view_names, view_latent_sizes):
            img = input_images.get(name)
            if img is None:
                img = PIL.Image.new(
                    "RGB",
                    (latent_size[1] * f, latent_size[0] * f),
                )
            per_view_pil[name] = img

        any_real_image = any(input_images.get(n) is not None for n in self.view_names)
        do_image_cfg = any_real_image and image_guidance_scale > 1.0

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

        # --- Per-view source latents (replicated across CFG conditions) ---
        source_latents_list: List[torch.Tensor] = []
        for name, latent_size in zip(self.view_names, view_latent_sizes):
            transform = v2.Compose([
                v2.Resize((latent_size[0] * f, latent_size[1] * f)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),
            ])
            src = transform(per_view_pil[name]).unsqueeze(0).to(device, dtype=self.vae.dtype)
            src_lat = self.vae.encode(src).latent * self.vae.config.scaling_factor

            if do_image_cfg:
                black_lat = (
                    self.vae.encode(torch.zeros_like(src)).latent
                    * self.vae.config.scaling_factor
                )
                src_lat = torch.cat([black_lat, src_lat, src_lat])
            else:
                src_lat = src_lat.repeat(2, 1, 1, 1)
            src_lat = src_lat.repeat_interleave(num_images_per_prompt, dim=0)
            source_latents_list.append(src_lat)

        # --- Initial noise per view ---
        latents_list: List[torch.Tensor] = [
            randn_tensor(
                shape=(num_images_per_prompt, self.config.in_channels, h, w),
                generator=generator, device=device, dtype=torch.float32,
            )
            for (h, w) in view_latent_sizes
        ]

        # One scheduler instance per view — DPMSolverMultistepScheduler keeps
        # an internal history of previous model outputs, and interleaving
        # `step` calls for different views on a shared scheduler would
        # corrupt that state.
        schedulers = [copy.deepcopy(self.scheduler) for _ in self.view_names]
        for sch in schedulers:
            if isinstance(sch, FlowMatchEulerDiscreteScheduler):
                sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                sch.set_timesteps(num_inference_steps, sigmas=sigmas)
            else:
                sch.set_timesteps(num_inference_steps)

        n_cond = len(captions)
        view_ids = [self.view_to_id[n] for n in self.view_names]

        for t in tqdm(schedulers[0].timesteps, desc="Sampling", disable=not enable_progress_bar):
            latent_inputs = []
            for lat in latents_list:
                lat_in = lat.repeat(n_cond, 1, 1, 1).to(prompt_embeds.dtype)
                if hasattr(self.scheduler, "scale_model_input"):
                    lat_in = self.scheduler.scale_model_input(lat_in, t)
                latent_inputs.append(lat_in)

            noise_preds = self.transformer(
                hidden_states=latent_inputs,
                source_hidden_states=[s.to(prompt_embeds.dtype) for s in source_latents_list],
                view_ids=view_ids,
                view_embedding=self.view_embedding,
                timestep=t.unsqueeze(0).expand(latent_inputs[0].shape[0]).to(device),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            ).sample
            if torch.is_tensor(noise_preds):
                noise_preds = [noise_preds]

            new_latents_list = []
            for lat, noise_pred, sch in zip(latents_list, noise_preds, schedulers):
                if do_image_cfg:
                    pred_uncond, pred_image, pred_full = noise_pred.chunk(3)
                    combined = (
                        pred_uncond
                        + image_guidance_scale * (pred_image - pred_uncond)
                        + guidance_scale * (pred_full - pred_image)
                    )
                else:
                    pred_uncond, pred_full = noise_pred.chunk(2)
                    combined = pred_uncond + guidance_scale * (pred_full - pred_uncond)
                new_latents_list.append(
                    sch.step(combined, t, lat).prev_sample
                )
            latents_list = new_latents_list

        outputs: dict = {}
        for name, lat in zip(self.view_names, latents_list):
            outputs[name] = self.decode_latents(
                lat.to(self.vae.dtype) if self.vae is not None else lat,
                return_tensor=return_tensor,
            )
        return outputs

from typing import Any, Callable

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from steer import init_temp_idx, lambda_ladder, rescale, sort_by_rung, swap

SAMPLERS = ("vanilla", "tsr", "steer")


class StableDiffusion3SteerPipeline(StableDiffusion3Pipeline):
    r"""
    Stable Diffusion 3 with temporal score rescaling (TSR) and replica exchange (STEER).

    Each prompt is sampled with a ladder of `n_replicas` latents at temperatures linearly spaced between
    `lam_start` and `lam_end`. Returned images are ordered rung-major, so `images[0]` is the `lam_start`
    sample of the first prompt. The initial noise of every returned image, in the same order, is stored in
    `last_initial_latents`.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        prompt_3: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        negative_prompt_3: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        clip_skip: int | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 77,
        sampler: str = "vanilla",
        lam_start: float = 1.0,
        lam_end: float | None = None,
        n_replicas: int = 1,
    ):
        r"""
        Arguments not listed here behave as in [`StableDiffusion3Pipeline.__call__`]. `latents`, if given, must
        hold `n_replicas * batch_size * num_images_per_prompt` initial latents in replica-major order.

        Args:
            sampler (`str`, defaults to `"vanilla"`):
                `"vanilla"` for unmodified sampling, `"tsr"` to rescale each replica's score by its rung, and
                `"steer"` to additionally exchange rungs between neighbouring replicas at every step.
            lam_start (`float`, defaults to 1.0):
                Temperature of the first rung. Values below 1 sharpen the distribution.
            lam_end (`float`, *optional*):
                Temperature of the last rung. Defaults to `lam_start`.
            n_replicas (`int`, defaults to 1):
                Number of rungs in the ladder.
        """
        if sampler not in SAMPLERS:
            raise ValueError(f"`sampler` must be one of {SAMPLERS}, got {sampler!r}")
        lam_end = lam_start if lam_end is None else lam_end

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        n_chains = batch_size * num_images_per_prompt

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        prompt_embeds = prompt_embeds.repeat(n_replicas, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(n_replicas, 1)
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat(n_replicas, 1, 1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(n_replicas, 1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        latents = self.prepare_latents(
            n_replicas * n_chains,
            self.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        initial_latents = latents.clone()

        ladder = lambda_ladder(lam_start, lam_end, n_replicas, device=device)
        temp_idx = init_temp_idx(n_replicas, n_chains, device=device)
        swap_generator = generator if isinstance(generator, torch.Generator) else None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if sampler != "vanilla":
                    sigma = self.scheduler.sigmas[i].to(device=latents.device, dtype=latents.dtype)
                    alpha = 1 - sigma
                    alpha_safe = alpha.clamp(min=1e-6)
                    eps = latents + alpha * noise_pred

                    if sampler == "steer":
                        score = -eps / torch.sqrt(torch.clamp(1 - alpha_safe, min=1e-3))
                        swap(latents, score, alpha_safe, ladder, temp_idx, i, generator=swap_generator)

                    eps = rescale(eps, alpha, ladder, temp_idx)
                    noise_pred = (eps - latents) / alpha_safe

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = sort_by_rung(latents, temp_idx)
        self.last_initial_latents = sort_by_rung(initial_latents, temp_idx)

        if output_type == "latent":
            image = latents
        else:
            image = []
            for latent in latents.split(1):
                latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                decoded = self.vae.decode(latent, return_dict=False)[0]
                image.append(self.image_processor.postprocess(decoded, output_type=output_type))
            if output_type == "pil":
                image = [img for batch in image for img in batch]
            elif output_type == "np":
                image = np.concatenate(image)
            else:
                image = torch.cat(image)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

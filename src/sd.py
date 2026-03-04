import gc
from dataclasses import dataclass

import torch
import drjit as dr

from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel

@dataclass
class SDConfig:
    prompt: str
    negative_prompt: str = ""
    prompt_embeds: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    cn_cond_scale: float = 0.0
    render_size: int = 1024
    guidance_scale: float = 0.5
    min_time: float = 0.02
    max_time: float = 0.98

class StableDiffusion:

    def __init__(self, config: SDConfig, device: str = 'cuda', generator: torch.Generator = None, enable_offload: bool = True):
        self.config = config
        self.device = device
        self.generator = generator or torch.Generator(device=device).manual_seed(42)
        
        # Instantiate pipeline
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)

        if enable_offload:
            self.pipe.enable_model_cpu_offload()

        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(config.prompt, config.negative_prompt, do_cfg=config.guidance_scale > 1.0)
        self.config.prompt_embeds = prompt_embeds
        self.config.pooled_prompt_embeds = pooled_prompt_embeds

        # Cleanup text encoders
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.text_encoder_3
        gc.collect()
        torch.cuda.empty_cache()

        for param in self.pipe.vae.parameters():
            param.requires_grad_(False)

        for param in self.pipe.controlnet.parameters():
            param.requires_grad_(False)

        for param in self.pipe.transformer.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def predict_velocity(self, latents: torch.Tensor, depth: torch.Tensor, timestep):
        device = self.device
        dtype = self.pipe.transformer.dtype

        do_cfg = self.config.guidance_scale > 1.0
 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        t = timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep, device=device)
        t_expanded = t.expand(latent_model_input.shape[0])

        # Run controlnet if enabled and depth is provided
        control_block_samples = None
        if self.config.cn_cond_scale != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            controlnet_config = self.pipe.controlnet.config
            vae_shift_factor = 0 if controlnet_config.force_zeros_for_pooled_projection else self.pipe.vae.config.shift_factor
            control_image = self.encode_image(depth, vae_shift_factor)

            # Controlnet pooled projections (InstantX depth controlnet uses zero projections)
            if controlnet_config.force_zeros_for_pooled_projection:
                controlnet_pooled_projections = torch.zeros_like(self.config.pooled_prompt_embeds)
            else:
                controlnet_pooled_projections = self.config.pooled_prompt_embeds

            controlnet_encoder_hidden_states = (
                self.config.prompt_embeds if controlnet_config.joint_attention_dim is not None else None
            )

            # Apply controlnet conditioning
            control_block_samples = self.pipe.controlnet(
                hidden_states=latent_model_input,
                timestep=t_expanded,
                encoder_hidden_states=controlnet_encoder_hidden_states,
                pooled_projections=controlnet_pooled_projections,
                joint_attention_kwargs=None,
                controlnet_cond=control_image,
                conditioning_scale=self.config.cn_cond_scale,
                return_dict=False,
            )[0]

        # Transformer forward pass (single denoising step)
        velocity_pred = self.pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t_expanded,
            encoder_hidden_states=self.config.prompt_embeds,
            pooled_projections=self.config.pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Classifier-free guidance
        if do_cfg:
            velocity_pred_uncond, velocity_pred_text = velocity_pred.chunk(2)
            velocity_pred = velocity_pred_uncond + self.config.guidance_scale * (velocity_pred_text - velocity_pred_uncond)

        return velocity_pred

    @dr.wrap(source='drjit', target='torch')
    def compute_rdfs_loss(self, image: torch.FloatTensor, depth: torch.FloatTensor):
        image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert to (1, C, H, W)
        depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)

        latents = self.encode_image(image).float()

        time = torch.rand(1, generator=self.generator, device=self.device) * (self.config.max_time - self.config.min_time) + self.config.min_time
        noise = torch.randn_like(latents, generator=self.generator, device=self.device)
        latents_noisy = time * noise + (1.0 - time) * latents

        predicted_vel = self.predict_velocity(
            latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
        )
        predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        loss_rfds = torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")
        loss_rdfs = loss_rfds.mean()
        return loss_rdfs

    @dr.wrap(source='drjit', target='torch')
    def compute_rdfs_rev_loss(self, image: torch.Tensor, depth: torch.Tensor):
        image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert to (1, C, H, W)
        depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)

        latents = self.encode_image(image).float()

        with torch.no_grad():
            time = torch.rand(1, generator=self.generator, device=self.device) * (self.config.max_time - self.config.min_time) + self.config.min_time

            # Sample noise and do a one step optimization of the noise
            noise = torch.randn_like(latents, generator=self.generator, device=self.device)
            latents_noisy = time * noise + (1.0 - time) * latents

            predicted_vel = self.predict_velocity(
                latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
            )
            noise = noise + (1 - time) * (predicted_vel + latents - noise)

            # Back to the original RDFS loss with the optimized noise
            latents_noisy = time * noise + (1.0 - time) * latents
            predicted_vel = self.predict_velocity(
                latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
            )
            predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        loss_rfds_rev = torch.nn.functional.mse_loss(noise + latents, predicted_vel, reduction="mean")
        loss_rdfs_rev = loss_rfds_rev.mean()
        return loss_rdfs_rev

    @torch.no_grad()
    def generate(self, num_inference_steps: int, depth: torch.Tensor):
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        latents = self.prepare_latents(depth.shape[-1])
        for t in timesteps:
            velocity_pred = self.predict_velocity(latents, depth, t)
            latents = self.pipe.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        return self.decode_latents(latents)


    def prepare_latents(self, render_size):
        return self.pipe.prepare_latents(
            1,
            self.pipe.transformer.config.in_channels,
            render_size,
            render_size,
            self.pipe.transformer.dtype,
            self.device,
            self.generator
        )
    
    def decode_latents(self, latents):
        latents = latents.to(device=self.device, dtype=self.pipe.vae.dtype)

        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents).sample

        return self.pipe.image_processor.postprocess(image, output_type='pt')
    
    def encode_image(self, image, vae_shift_factor=None):
        image = image.to(device=self.device, dtype=self.pipe.vae.dtype)
        image = self.pipe.image_processor.preprocess(image)

        vae_shift_factor = vae_shift_factor or self.pipe.vae.config.shift_factor
        image = self.pipe.vae.encode(image).latent_dist.sample(generator=self.generator)
        return (image - vae_shift_factor) * self.pipe.vae.config.scaling_factor
    
    @torch.no_grad()
    def _encode_prompt(self, prompt: str, negative_prompt: str, do_cfg: bool) -> tuple[torch.Tensor, torch.Tensor]:
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=self.device,
        )

        if do_cfg:
            # Concatenate unconditional and conditional embeddings into a single forward pass
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # Returns the precomputed parameters as a dictionary
        return prompt_embeds, pooled_prompt_embeds


if __name__ == "__main__":
    # Example usage
    import mitsuba as mi


    config = SDConfig(
        prompt="A photorealistic image of a mountain landscape at sunset",
        negative_prompt="low quality, blurry",
        cn_cond_scale=0.0,
        render_size=1024,
        guidance_scale=7.5,
        min_time=0.02,
        max_time=0.98
    )
    sd = StableDiffusion(config, 'cuda', enable_offload=False)


    # Dummy depth map for testing
    depth = torch.rand((1, 3, config.render_size, config.render_size))
    generated_image = sd.generate(num_inference_steps=50, depth=depth)

    mi.util.write_bitmap("outputs/generated_image.exr", generated_image.squeeze(0).permute(1, 2, 0))
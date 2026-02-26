import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel


class StableDiffusion:

    def __init__(self, device: str = 'cuda', generator: torch.Generator = None, enable_offload: bool = True):
        self.device = device
        self.generator = generator
        
        # Instantiate pipeline
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)

        if enable_offload:
            self.pipe.enable_model_cpu_offload()


    @torch.no_grad()
    def prep_sd(self, prompt: str, negative_prompt: str, guidance_scale: float, cn_cond_scale: float, render_size: int):
        do_cfg = guidance_scale > 1.0

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
        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'cn_cond_scale': cn_cond_scale,
            'render_size': render_size,
            'guidance_scale': guidance_scale,
            'do_cfg': do_cfg,
        }
    
    @torch.no_grad()
    def predict_velocity(self, latents: torch.Tensor, depth: torch.Tensor, sd_config, timestep):
        device = self.device
        dtype = self.pipe.transformer.dtype

        latents = latents.to(device=device, dtype=dtype)
        depth = depth.to(device=device, dtype=dtype)

        prompt_embeds = sd_config['prompt_embeds']
        pooled_prompt_embeds = sd_config['pooled_prompt_embeds']
        do_cfg = sd_config['do_cfg']

        # Prepare depth image as controlnet conditioning (encoded to latent space)
        controlnet_config = self.pipe.controlnet.config
        vae_shift_factor = 0 if controlnet_config.force_zeros_for_pooled_projection else self.pipe.vae.config.shift_factor

        control_image = self.pipe.prepare_image(
                image=depth,
                width=depth.shape[-1],
                height=depth.shape[-2],
                batch_size=1,
                num_images_per_prompt=1,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=do_cfg,
                guess_mode=False,
            )
        
        control_image = self.encode_image(control_image, vae_shift_factor)
 
        # Expand inputs along batch dimension for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        t = timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep, device=device)
        t_expanded = t.expand(latent_model_input.shape[0])

        # Controlnet pooled projections (InstantX depth controlnet uses zero projections)
        if controlnet_config.force_zeros_for_pooled_projection:
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = pooled_prompt_embeds

        controlnet_encoder_hidden_states = (
            prompt_embeds if controlnet_config.joint_attention_dim is not None else None
        )

        # Apply controlnet conditioning
        control_block_samples = self.pipe.controlnet(
            hidden_states=latent_model_input,
            timestep=t_expanded,
            encoder_hidden_states=controlnet_encoder_hidden_states,
            pooled_projections=controlnet_pooled_projections,
            joint_attention_kwargs=None,
            controlnet_cond=control_image,
            conditioning_scale=sd_config['cn_cond_scale'],
            return_dict=False,
        )[0]

        # Transformer forward pass (single denoising step)
        velocity_pred = self.pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t_expanded,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Classifier-free guidance
        if do_cfg:
            velocity_pred_uncond, velocity_pred_text = velocity_pred.chunk(2)
            velocity_pred = velocity_pred_uncond + sd_config['guidance_scale'] * (velocity_pred_text - velocity_pred_uncond)

        return velocity_pred

    @torch.no_grad()
    def generate(self, prompt: str, negative_prompt: str, guidance_scale: float, 
                       num_inference_steps: int, cn_cond_scale: float, depth: torch.Tensor):
        
        sd_config = self.prep_sd(prompt, negative_prompt, guidance_scale, cn_cond_scale, depth.shape[-1])
        
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        latents = self.prepare_latents(depth.shape[-1])
        for t in timesteps:
            velocity_pred = self.predict_velocity(latents, depth, sd_config, t)
            latents = self.pipe.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        image = self.decode_latents(latents)
        image = self.pipe.image_processor.postprocess(image, output_type='pt')

        return image
    

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
        return self.pipe.vae.decode(latents, return_dict=False)[0]
    
    def encode_image(self, image, vae_shift_factor=None):
        image = image.to(device=self.device, dtype=self.pipe.vae.dtype)

        vae_shift_factor = vae_shift_factor or self.pipe.vae.config.shift_factor
        image = self.pipe.vae.encode(image).latent_dist.sample(generator=self.generator)
        return (image - vae_shift_factor) * self.pipe.vae.config.scaling_factor
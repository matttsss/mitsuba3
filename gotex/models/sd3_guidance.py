from dataclasses import dataclass

import torch

from diffusers.models import SD3ControlNetModel
from diffusers import StableDiffusion3ControlNetPipeline

import gotex
from gotex.config import RuntimeContext
from gotex.models.distilator import Distilator

@gotex.register("sd3_guidance")
class StableDiffusion(Distilator):

    @dataclass
    class Config(Distilator.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
        controlnet_conditioning_scale: float = 1.0

    cfg: Config

    def __init__(
        self,
        config: dict,
        runtime: RuntimeContext
    ):
        super().__init__(config, runtime=runtime)
        
        # Instantiate pipeline and extract components
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=None, text_encoder_2=None, text_encoder_3=None,
            tokenizer=None, tokenizer_2=None, tokenizer_3=None,
            controlnet=controlnet, torch_dtype=torch.float16
        ).to(self.runtime.device)

        if self.cfg.enable_offload:
            pipe.enable_model_cpu_offload()

        # Extract components from pipeline (do not store the pipeline itself)
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.controlnet = pipe.controlnet
        self.image_processor = pipe.image_processor
        self.scheduler = pipe.scheduler

        self.vae_scale_factor = pipe.vae_scale_factor
        self.num_train_steps = pipe.scheduler.config.num_train_timesteps

        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.controlnet.parameters():
            param.requires_grad_(False)
        for param in self.transformer.parameters():
            param.requires_grad_(False)

    @property
    def model(self):
        return self.transformer
    
    @torch.no_grad()
    def predict_velocity(self, prompt_embedings: dict[str, torch.Tensor], latents: torch.Tensor, depth: torch.Tensor, timestep):
        device = self.runtime.device
        dtype = self.transformer.dtype


        do_cfg = self.cfg.guidance_scale > 1.0
        
        prompt_embeds = prompt_embedings["prompt_embeds"]
        pooled_prompt_embeds = prompt_embedings["pooled_prompt_embeds"]

        # Check if the prompts are already aligned with the batch size of latents; if not, repeat them
        prompts_alligned = all(emb.shape[0] == latents.shape[0] for emb in prompt_embedings.values())
        if not prompts_alligned:
            prompt_embeds = torch.cat([prompt_embeds] * latents.shape[0])
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds] * latents.shape[0])

        if do_cfg:

            negative_prompt_embeds = prompt_embedings["negative_prompt_embeds"]
            negative_pooled_prompt_embeds = prompt_embedings["negative_pooled_prompt_embeds"]

            if not prompts_alligned:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds] * latents.shape[0])
                negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds] * latents.shape[0])

            # Concatenate unconditional and conditional embeddings into a single forward pass
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        t = timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep, device=device)
        t_expanded = t.expand(latent_model_input.shape[0])

        # Run controlnet if enabled and depth is provided
        control_block_samples = None
        if self.cfg.controlnet_conditioning_scale != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            if do_cfg:
                depth = torch.cat([depth] * 2)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            controlnet_config = self.controlnet.config
            vae_shift_factor = 0 if controlnet_config.force_zeros_for_pooled_projection else self.vae.config.shift_factor

            depth = self.vae.encode(depth).latent_dist.sample(generator=self.runtime.torch_generator)
            control_image = (depth - vae_shift_factor) * self.vae.config.scaling_factor

            # Controlnet pooled projections (InstantX depth controlnet uses zero projections)
            if controlnet_config.force_zeros_for_pooled_projection:
                controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
            else:
                controlnet_pooled_projections = pooled_prompt_embeds

            controlnet_encoder_hidden_states = (
                prompt_embeds if controlnet_config.joint_attention_dim is not None else None
            )

            # Apply controlnet conditioning
            control_block_samples = self.controlnet(
                hidden_states=latent_model_input,
                timestep=t_expanded,
                encoder_hidden_states=controlnet_encoder_hidden_states,
                pooled_projections=controlnet_pooled_projections,
                joint_attention_kwargs=None,
                controlnet_cond=control_image,
                conditioning_scale=self.cfg.controlnet_conditioning_scale,
                return_dict=False,
            )[0]

        # Transformer forward pass (single denoising step)
        velocity_pred = self.transformer(
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
            velocity_pred = velocity_pred_uncond + self.cfg.guidance_scale * (velocity_pred_text - velocity_pred_uncond)

        return velocity_pred

    def decode_latents(self, latents):
        latents = latents.to(device=self.runtime.device, dtype=self.vae.dtype)

        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents).sample

        return self.image_processor.postprocess(image, output_type='pt')
    
    def encode_image(self, image, vae_shift_factor=None):
        image = image.to(device=self.runtime.device, dtype=self.vae.dtype)
        image = self.image_processor.preprocess(image)

        vae_shift_factor = vae_shift_factor or self.vae.config.shift_factor
        image = self.vae.encode(image).latent_dist.sample(generator=self.runtime.torch_generator)
        return (image - vae_shift_factor) * self.vae.config.scaling_factor

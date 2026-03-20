import gc

import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel

from .distilator import Distilator

class StableDiffusion(Distilator):

    def __init__(self, config: dict, device: str = 'cuda', generator: torch.Generator = None, enable_offload: bool = True):
        super().__init__(generator, device, config)
        
        # Instantiate pipeline
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            text_encoder_3=None,tokenizer_3=None,
            controlnet=controlnet, torch_dtype=torch.float16
        ).to(device)

        if enable_offload:
            self.pipe.enable_model_cpu_offload()

        # rename the model for interface in the distilator
        self.pipe.model = self.pipe.transformer

        for param in self.pipe.vae.parameters():
            param.requires_grad_(False)
        for param in self.pipe.controlnet.parameters():
            param.requires_grad_(False)
        for param in self.pipe.model.parameters():
            param.requires_grad_(False)

    def cleanup_text_encoders(self):
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.text_encoder_3
        gc.collect()
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def predict_velocity(self, prompt_embedings: dict[str, torch.Tensor], latents: torch.Tensor, depth: torch.Tensor, timestep):
        device = self.device
        dtype = self.pipe.model.dtype

        do_cfg = self.config["guidance_scale"] > 1.0
        
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
        if self.config["cn_cond_scale"] != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            if do_cfg:
                depth = torch.cat([depth] * 2)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            controlnet_config = self.pipe.controlnet.config
            vae_shift_factor = 0 if controlnet_config.force_zeros_for_pooled_projection else self.pipe.vae.config.shift_factor
            control_image = self.encode_image(depth, vae_shift_factor)

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
                conditioning_scale=self.config["cn_cond_scale"],
                return_dict=False,
            )[0]

        # Transformer forward pass (single denoising step)
        velocity_pred = self.pipe.model(
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
            velocity_pred = velocity_pred_uncond + self.config["guidance_scale"] * (velocity_pred_text - velocity_pred_uncond)

        return velocity_pred

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
    def _encode_prompt(self, prompt: str, negative_prompt: str) -> dict[str, torch.Tensor]:
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
            do_classifier_free_guidance=True, # Worst case, we ignore the negative prompt results
            device=self.device,
        )

        # Returns the precomputed parameters as a dictionary
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
        }

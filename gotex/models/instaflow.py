import torch

from dataclasses import dataclass

from diffusers import ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

import gotex
from gotex.config import RuntimeContext
from gotex.models.distilator import Distilator

@gotex.register("instaflow_guidance")
class Instaflow(Distilator):

    @dataclass
    class Config(Distilator.Config):
        pretrained_model_name_or_path: str = "XCLiu/2_rectified_flow_from_sd_1_5"
        controlnet_conditioning_scale: float = 0.5
   
    def __init__(self, 
                 config: dict,
                 runtime: RuntimeContext):
        super().__init__(config, runtime=runtime)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            controlnet=controlnet, text_encoder=None, tokenizer=None,
            safety_checker=None, torch_dtype=torch.float16
        ).to(self.runtime.device)

        self.num_train_steps = self.pipe.scheduler.config.num_train_timesteps
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.num_train_steps)

        if self.cfg.enable_offload:
            self.pipe.enable_model_cpu_offload()

        for param in self.pipe.vae.parameters():
            param.requires_grad_(False)
        for param in self.pipe.controlnet.parameters():
            param.requires_grad_(False)
        for param in self.pipe.unet.parameters():
            param.requires_grad_(False)

    @property
    def model(self):
        return self.pipe.unet

    @torch.no_grad()
    def predict_velocity(self, prompt_embedings: dict[str, torch.Tensor], latents: torch.Tensor, depth: torch.Tensor, timestep):
        device = self.runtime.device
        dtype = self.pipe.unet.dtype

        guess_mode = self.pipe.controlnet.config.global_pool_conditions
        do_cfg = self.cfg.guidance_scale > 1.0
        
        prompt_embeds = prompt_embedings["prompt_embeds"]

        # Check if the prompts are already aligned with the batch size of latents; if not, repeat them
        prompts_alligned = all(emb.shape[0] == latents.shape[0] for emb in prompt_embedings.values())
        if not prompts_alligned:
            prompt_embeds = torch.cat([prompt_embeds] * latents.shape[0])

        if do_cfg:
            negative_prompt_embeds = prompt_embedings["negative_prompt_embeds"]

            if not prompts_alligned:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds] * latents.shape[0])

            # Concatenate unconditional and conditional embeddings into a single forward pass
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        t = timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep, device=device)
        t_expanded = t.expand(latent_model_input.shape[0])

 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

        # Run controlnet if enabled and depth is provided
        down_block_res_samples, mid_block_res_sample = None, None
        if self.cfg.controlnet_conditioning_scale != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            depth = self.pipe.control_image_processor.preprocess(depth)

            if do_cfg and not guess_mode:
                depth = torch.cat([depth] * 2)

            # controlnet(s) inference
            if guess_mode and do_cfg:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                control_model_input,
                timestep,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=depth,
                conditioning_scale=self.cfg.controlnet_conditioning_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            if guess_mode and self.do_classifier_free_guidance:
                # Inferred ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        velocity_pred = self.pipe.unet(
            latent_model_input, 
            timestep, 
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        if do_cfg:
            velocity_pred_text, velocity_pred_uncond = velocity_pred.chunk(2)
            velocity_pred = velocity_pred_uncond + self.cfg.guidance_scale * (
                velocity_pred_text - velocity_pred_uncond
            )

        return velocity_pred

    def encode_image(self, image: torch.FloatTensor):
        image = self.pipe.image_processor.preprocess(image)
        image = image.to(device=self.runtime.device, dtype=self.pipe.vae.dtype)

        return self.pipe.vae.encode(image).latent_dist.sample(generator=self.runtime.torch_generator) * self.pipe.vae.config.scaling_factor

    def decode_latents(self, latents: torch.FloatTensor):
        latents = latents.to(device=self.runtime.device, dtype=self.pipe.vae.dtype)

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=self.runtime.torch_generator)[0]
        return self.pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=[True])


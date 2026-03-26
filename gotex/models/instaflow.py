import torch

from diffusers import ControlNetModel
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .distilator import Distilator

class Instaflow(Distilator):

    class Config(Distilator.Config):
        pretrained_model_name_or_path: str = "XCLiu/instaflow_0_9B_from_sd_1_5"
        controlnet_conditioning_scale: float = 0.5
        render_size: int = 512
   
    def __init__(self, 
                 config: dict, device: str = 'cuda',
                 instaflow: bool = False, 
                 generator: torch.Generator = None):
        super().__init__(config, generator, device)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "XCLiu/instaflow_0_9B_from_sd_1_5" if instaflow else "XCLiu/2_rectified_flow_from_sd_1_5", 
            controlnet=controlnet, text_encoder=None, tokenizer=None,
            safety_checker=None, torch_dtype=torch.float16
        ).to(device)

        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.pipe.scheduler.config.num_train_timesteps)

        if self.cfg.enable_offload:
            self.pipe.enable_model_cpu_offload()

        for param in self.pipe.vae.parameters():
            param.requires_grad_(False)
        for param in self.pipe.controlnet.parameters():
            param.requires_grad_(False)
        for param in self.pipe.model.parameters():
            param.requires_grad_(False)

    @property
    def model(self):
        return self.pipe.unet

    @torch.no_grad()
    def predict_velocity(
        self,
        latents: torch.FloatTensor,
        depth: torch.FloatTensor,
        timestep: float
    ) -> torch.FloatTensor:
        device = self.device
        dtype = self.pipe.model.dtype

        do_cfg = self.cfg.guidance_scale > 1.0

        if do_cfg:
            prompt_embedings = torch.cat([self.cfg["negative_prompt_embeds"], self.cfg["prompt_embeds"]])
        else:
            prompt_embedings = self.cfg["prompt_embeds"]
 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

        # Run controlnet if enabled and depth is provided
        down_block_res_samples, mid_block_res_sample = None, None
        if self.cfg.controlnet_conditioning_scale != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            depth = self.pipe.control_image_processor.preprocess(depth, 
                            height=self.cfg.render_size, width=self.cfg.render_size
                    )

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embedings,
                controlnet_cond=depth,
                conditioning_scale=self.cfg.controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )

        velocity_pred = self.pipe.model(
            latent_model_input, 
            timestep, 
            encoder_hidden_states=prompt_embedings,
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
        image = self.pipe.image_processor.preprocess(image, 
                            height=self.cfg.render_size, width=self.cfg.render_size
            )
        image = image.to(device=self.device, dtype=self.pipe.vae.dtype)

        latents = self.pipe.vae.encode(image).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.FloatTensor):
        latents = latents.to(device=self.device, dtype=self.pipe.vae.dtype)

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=[True])
        
        return image

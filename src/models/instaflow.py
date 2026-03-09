import gc

import torch
from src.models.distilator import Distilator
from diffusers import ControlNetModel


from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

class Instaflow(Distilator):
   
    def __init__(self, 
                 config: dict, device: str = 'cuda',
                 instaflow: bool = False, 
                 generator: torch.Generator = None, 
                 enable_offload: bool = True):
        super().__init__(generator, device, config)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "XCLiu/instaflow_0_9B_from_sd_1_5" if instaflow else "XCLiu/2_rectified_flow_from_sd_1_5", 
            controlnet=controlnet, 
            safety_checker=None,
            torch_dtype=torch.float16
        ).to(device)

        self.pipe.model = self.pipe.unet
        del self.pipe.unet

        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.pipe.scheduler.config.num_train_timesteps)

        if enable_offload:
            self.pipe.enable_model_cpu_offload()

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds
            ) = self.pipe.encode_prompt(
                prompt=self.config["prompt"],
                negative_prompt=self.config["negative_prompt"],
                do_classifier_free_guidance=True, # Worst case, negative prompt_embeds are not used
                num_images_per_prompt=1,
                device=self.device,
            )
            self.config["prompt_embeds"] = prompt_embeds
            self.config["negative_prompt_embeds"] = negative_prompt_embeds

        # Cleanup text encoders
        del self.pipe.text_encoder
        gc.collect()
        torch.cuda.empty_cache()

        for param in self.pipe.vae.parameters():
            param.requires_grad_(False)
        for param in self.pipe.controlnet.parameters():
            param.requires_grad_(False)
        for param in self.pipe.model.parameters():
            param.requires_grad_(False)


    @torch.no_grad()
    def predict_velocity(
        self,
        latents: torch.FloatTensor,
        depth: torch.FloatTensor,
        timestep: float
    ) -> torch.FloatTensor:
        device = self.device
        dtype = self.pipe.model.dtype

        do_cfg = self.config["guidance_scale"] > 1.0

        if do_cfg:
            prompt_embedings = torch.cat([self.config["negative_prompt_embeds"], self.config["prompt_embeds"]])
        else:
            prompt_embedings = self.config["prompt_embeds"]
 
        # Expand inputs along batch dimension for classifier-free guidance
        latents = latents.to(device=device, dtype=dtype)
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

        # Run controlnet if enabled and depth is provided
        down_block_res_samples, mid_block_res_sample = None, None
        if self.config["cn_cond_scale"] != 0.0 and depth is not None:
            depth = depth.to(device=device, dtype=dtype)

            # Prepare depth image as controlnet conditioning (encoded to latent space)
            depth = self.pipe.control_image_processor.preprocess(depth, 
                            height=self.config["render_size"], width=self.config["render_size"]
                    )

            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embedings,
                controlnet_cond=depth,
                conditioning_scale=self.config["cn_cond_scale"],
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
            velocity_pred = velocity_pred_uncond + self.config["guidance_scale"] * (
                velocity_pred_text - velocity_pred_uncond
            )

        return velocity_pred

    def encode_image(self, image: torch.FloatTensor):
        image = self.pipe.image_processor.preprocess(image, 
                            height=self.config["render_size"], width=self.config["render_size"]
            )
        image = image.to(device=self.device, dtype=self.pipe.vae.dtype)

        latents = self.pipe.vae.encode(image).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.FloatTensor):
        latents = latents.to(device=self.device, dtype=self.pipe.vae.dtype)

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=[True])
        
        return image


if __name__ == "__main__":
    # Example usage
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')

    config = dict(
        prompt="A blue dragon on a piedestal",
        negative_prompt="",
        cn_cond_scale=0.7,
        render_size=512,
        guidance_scale=1
    )

    depth_img = mi.TensorXf(mi.Bitmap("outputs/depth.exr")).torch().permute(2, 0, 1).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, C, H, W)

    nb_steps = 50
    dt = 1 / nb_steps
    sd = Instaflow(config, instaflow=False, device='cuda', enable_offload=False)

    with torch.no_grad():
        latents = sd.prepare_latents(config["render_size"])

        timesteps = torch.linspace(1000, 1, steps=nb_steps, device=sd.device)

        for t in timesteps:
            velocity_pred = sd.predict_velocity(latents, depth_img, t)
            latents = latents + dt * velocity_pred

        image = sd.decode_latents(latents)
        mi.util.write_bitmap("outputs/generated_image.exr", image.squeeze(0).permute(1, 2, 0))

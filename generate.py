import torch
import mitsuba as mi

from gotex.models.sd import StableDiffusion

if __name__ == "__main__":

    mi.set_variant('cuda_ad_rgb')

    prompt="A photorealistic image of a mountain landscape at sunset"
    negative_prompt="low quality, blurry"

    config = dict(
        cn_cond_scale=0.0,
        render_size=1024,
        guidance_scale=7.5
    )
    sd = StableDiffusion(config, 'cuda', enable_offload=False)

    prompt_embeds = sd._encode_prompt(prompt, negative_prompt)

    sd.pipe.scheduler.set_timesteps(50, device=sd.device)
    timesteps = sd.pipe.scheduler.timesteps

    with torch.no_grad():
        latents = sd.prepare_latents(sd.config["render_size"])
        for t in timesteps:
            velocity_pred = sd.predict_velocity(prompt_embeds, latents, None, t)
            latents = sd.pipe.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        generated_image = sd.decode_latents(latents)

        mi.util.write_bitmap("outputs/generated_image.exr", generated_image.squeeze(0).permute(1, 2, 0))
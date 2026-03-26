import torch
import mitsuba as mi

from gotex.models.sd import StableDiffusion
from gotex.models.prompt_encoder import PromptEncoder

if __name__ == "__main__":

    mi.set_variant('cuda_ad_rgb')

    prompt="A photorealistic image of a mountain landscape at sunset"
    negative_prompt="low quality, blurry, unrealistic looking"

    config = dict(
        cn_cond_scale=0.0,
        render_size=1024,
        guidance_scale=7.5
    )
    sd = StableDiffusion(config, 'cuda', enable_offload=False)
    text_encoder = PromptEncoder(
        prompts=[prompt, negative_prompt],
        device=sd.device,
        dtype=sd.transformer.dtype
    )

    test_image = mi.TensorXf(mi.Bitmap("outputs/test_img.exr")).torch().permute(2, 0, 1).to(sd.device)

    prompt_embeds = text_encoder.encode_prompt(prompt, negative_prompt, test_image)

    for k, v in prompt_embeds.items():
        print(f"{k}: {v.shape}")

    sd.scheduler.set_timesteps(50, device=sd.device)
    timesteps = sd.scheduler.timesteps

    with torch.no_grad():
        latents = sd.prepare_latents(sd.config["render_size"])
        for t in timesteps:
            velocity_pred = sd.predict_velocity(prompt_embeds, latents, None, t)
            latents = sd.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        generated_image = sd.decode_latents(latents)

        mi.util.write_bitmap("outputs/generated_image.exr", generated_image.squeeze(0).permute(1, 2, 0))
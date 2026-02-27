
import torch

import mitsuba as mi

from renderer import get_depth
from scenes.dragon import load_scene

from sd import StableDiffusion

mi.set_variant('cuda_ad_rgb')


def save_latents_as_image(sd: StableDiffusion, latents, filename):
    image = sd.decode_latents(latents)
    image = sd.pipe.image_processor.postprocess(image, output_type='pt', do_denormalize=[False])
    image = image.squeeze(0).permute(1, 2, 0)
    mi.util.write_bitmap(filename, image)


def get_index_for_timestep(timesteps, t):
    for i, timestep in enumerate(timesteps):
        if timestep <= t:
            return i
    return len(timesteps) - 1


def hdr_to_sdr(img, exposure=1.0):
    # 1. Apply exposure
    img = img * exposure
    
    # 2. Tone mapping (ACES)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    img = (img*(a*img+b)) / (img*(c*img+d)+e)
    
    # 3. Clamp
    img = torch.clamp(img, 0, 1)
    
    # 4. Gamma correction
    img = torch.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * torch.pow(img, 1/2.4) - 0.055
    )
    
    return img


with torch.no_grad():
    scene, scene_params = load_scene(render_size=1024)
    sd = StableDiffusion(device='cuda', enable_offload=True)
    sd_config = sd.prep_sd("", "", 1.0, 0.5, 1024)

    depth: torch.Tensor = get_depth(scene, scene.sensors()[0]).torch()
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    image: torch.Tensor = mi.render(scene, params=scene_params).torch()
    image = hdr_to_sdr(image, exposure=0.1)
    mi.util.write_bitmap('outputs/dragon_rendered.exr', image)
    image = image.permute(2, 0, 1).unsqueeze(0)

    t = 0.1
    nb_steps = 100

    latent_img = sd.encode_image(image)
    noise = torch.randn_like(latent_img, generator=sd.generator, device=sd.device)
    
    scheduler = sd.pipe.scheduler
    scheduler.set_timesteps(nb_steps, device=sd.device)
    
    # Find the timestep index closest to t
    start_index = get_index_for_timestep(scheduler.timesteps, t*1000)
    timestep_for_noise = scheduler.timesteps[start_index]
    
    # Use scheduler's scale_noise for consistency
    latents_noisy = scheduler.scale_noise(latent_img, timestep_for_noise.unsqueeze(0), noise)
    save_latents_as_image(sd, latents_noisy, 'outputs/dragon_noisy.exr')

    scheduler.set_begin_index(start_index)
    for i in range(start_index, nb_steps):
        timestep = scheduler.timesteps[i]
        print(f"Step {i+1}/{nb_steps}, Time: {timestep}")

        velocity = sd.predict_velocity(
            latents_noisy, depth, sd_config, timestep
        )
        latents_noisy = scheduler.step(velocity, timestep, latents_noisy, return_dict=False)[0]

    # Decode and denormalize final denoised image
    save_latents_as_image(sd, latents_noisy, 'outputs/dragon_denoised.exr')

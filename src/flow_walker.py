
import torch

import mitsuba as mi

from renderer import get_depth
from scenes.dragon import load_scene
from utils import save_latents_as_image, get_index_for_timestep, hdr_to_sdr

from models.sd import StableDiffusion

mi.set_variant('cuda_ad_rgb')

with torch.no_grad():
    scene, scene_params, scene_metadata = load_scene(render_size=1024)
    sd = StableDiffusion(device='cuda', enable_offload=True)
    sd_config = sd.prep_sd("", "", 1.0, 0.5, 1024)

    depth: torch.Tensor = get_depth(scene, scene.sensors()[0]).torch()
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    image: torch.Tensor = mi.render(scene, params=scene_params).torch()
    image = hdr_to_sdr(image, exposure=0.1)
    mi.util.write_bitmap('outputs/dragon_rendered.exr', image)
    image = image.permute(2, 0, 1).unsqueeze(0)

    t = 0.3
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

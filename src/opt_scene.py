import torch

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.dragon import load_scene
from renderer import randomize_sensor, scene_step

mi.set_variant('cuda_ad_rgb')

# Parameters
camera_to_world_key = 'sensor.to_world'
render_size = 1024
prompt="Blue dragon on a piedestal, highly detailed, directional sunlight, 4k, photorealistic, grassy hills background"
negative_prompt="" #change geometry, change shape, change pose, change structure
guidance_scale=10
num_images_per_prompt=1
cn_cond_scale=0.6

# Instantiate scene
seed = 40
device = torch.device('cuda')
pt_generator = torch.Generator(device=device).manual_seed(seed)
dr_generator = dr.rng(seed=seed)

sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
sd_config = sd.prep_sd(
    prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
    cn_cond_scale=cn_cond_scale, render_size=render_size, min_time=0.02, max_time=0.25
)

scene, scene_params = load_scene(render_size)


scene_params.keep([r'.*\.reflectance\.data', camera_to_world_key])
opt = mi.ad.Adam(lr=5e-3, params = {
    k:v for k, v in scene_params.items() if camera_to_world_key not in k
})
scene_params.update(opt)


for i in range(500):
    randomize_sensor(dr_generator, scene_params, sensor_to_world_key=camera_to_world_key, target=[0, 7, 0], radius=60)
    image, loss = scene_step(scene, scene_params, sd, sd_config)

    opt.step()
    for k, v in opt.items():
        opt[k] = dr.clip(opt[k], 0, 1)

    scene_params.update(opt)

    dr.print("Iteration {}: Mean: {}, Max: {}, Min: {}, Loss: {}", i, dr.mean(image), dr.max(image), dr.min(image), loss)

    if i % 10 == 0:
        mi.util.write_bitmap('outputs/dragon_opt.png', image)
        for k, v in opt.items():
            mi.util.write_bitmap(f'outputs/dragon_tex/{k.split(".")[0]}.png', opt[k])

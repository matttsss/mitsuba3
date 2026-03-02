import torch, random

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.coffee_maker import load_scene
from renderer import randomize_sensor, scene_step

mi.set_variant('cuda_ad_rgb')

# Seed and set device
seed = 40
device = torch.device('cuda')
pt_generator = torch.Generator(device=device).manual_seed(seed)
dr_generator = dr.rng(seed=seed)

# Instantiate scene
render_size = 1024
scene, scene_params, scene_metadata = load_scene(render_size)

# Parameters
camera_to_world_key = 'sensor.to_world'
prompt=scene_metadata['prompt']
negative_prompt="" #change geometry, change shape, change pose, change structure
guidance_scale=0
cn_cond_scale=0.6

sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
sd_config = sd.prep_sd(
    prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
    cn_cond_scale=cn_cond_scale, render_size=render_size, min_time=0.02, max_time=0.1
)


scene_params.keep([r'.*\.reflectance\.data', camera_to_world_key])
opt = mi.ad.Adam(lr=5e-3, params = {
    k:v for k, v in scene_params.items() if camera_to_world_key not in k
})
scene_params.update(opt)

sensor_count = 32

for i in range(500):
    randomize_sensor(scene_params, sensor_to_world_key=camera_to_world_key,
                     sensor_idx=random.randint(0, sensor_count-1), sensor_count=sensor_count,
                     target=scene_metadata['target'], radius=scene_metadata['radius'])
    image, loss = scene_step(scene, scene_params, sd, sd_config)

    opt.step()
    for k, v in opt.items():
        opt[k] = dr.clip(opt[k], 0, 1)

    scene_params.update(opt)

    dr.print("Iteration {}: Mean: {}, Max: {}, Min: {}, Loss: {}", i, dr.mean(image), dr.max(image), dr.min(image), loss)

    if i % 10 == 0:
        mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{scene_metadata["scene_name"]}_opt.png', image)
        for k, v in opt.items():
            mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{k.split(".")[0]}.png', opt[k])

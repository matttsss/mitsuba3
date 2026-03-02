import torch, random

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.painting import load_scene
from renderer import randomize_sensor, scene_step

mi.set_variant('cuda_ad_rgb')

# Seed and set device
seed = 40
device = torch.device('cuda')
random.seed(seed)
pt_generator = torch.Generator(device=device).manual_seed(seed)
dr_generator = dr.rng(seed=seed)

# Instantiate scene
render_size = 512
scene, scene_params, scene_metadata = load_scene(render_size)

# Parameters
camera_to_world_key = 'sensor.to_world'
prompt=scene_metadata['prompt']
negative_prompt="" #change geometry, change shape, change pose, change structure
guidance_scale=50
cn_cond_scale=0

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

nb_sensors = 32
nb_opt_steps = 500
nb_acc_steps = 1
for i in range(nb_opt_steps * nb_acc_steps + 1):
    # randomize_sensor(scene_params, sensor_to_world_key=camera_to_world_key,
    #                  sensor_idx=random.randint(0, nb_sensors-1), sensor_count=nb_sensors,
    #                  target=scene_metadata['target'], radius=scene_metadata['radius'])
    image, loss = scene_step(scene, scene_params, sd, sd_config)

    if i % nb_acc_steps == 0:
        opt.step(grad_scale=1/nb_acc_steps)
        for k, v in opt.items():
            opt[k] = dr.clip(opt[k], 0, 1)

        scene_params.update(opt)

        dr.print("Iteration {}: Mean: {}, Max: {}, Min: {}, Loss: {}", i//nb_acc_steps, dr.mean(image), dr.max(image), dr.min(image), loss)
        
        mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{scene_metadata["scene_name"]}_opt.exr', image)
        for k, v in opt.items():
            mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{k.split(".")[0]}.exr', opt[k])

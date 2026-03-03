import torch, random
from tqdm.auto import trange

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.coffee_maker import load_scene
from renderer import randomize_sensor, get_depth

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
cn_cond_scale=0.6

sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
sd_config = sd.prep_sd(
    prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
    cn_cond_scale=cn_cond_scale, render_size=render_size, min_time=0.02, max_time=0.98
)


scene_params.keep([r'.*\.reflectance\.data', camera_to_world_key])
opt = mi.ad.Adam(lr=3e-2, params = {
    k:v for k, v in scene_params.items() if camera_to_world_key not in k
})
scene_params.update(opt)

nb_sensors = 32
nb_opt_steps = 5000
nb_acc_steps = 2
nb_steps_save = 10

iterator = trange(nb_opt_steps * nb_acc_steps, desc="Optimizing")
for i in iterator:
    randomize_sensor(scene_params, sensor_to_world_key=camera_to_world_key,
                     sensor_idx=random.randint(0, nb_sensors-1), sensor_count=nb_sensors,
                     target=scene_metadata['target'], radius=scene_metadata['radius'])

    # Render depth and image
    dr_depth = get_depth(scene, sensor=scene.sensors()[0])
    dr_image = mi.render(scene, params=scene_params, seed=i)

    # Convert to torch tensors
    loss_rdfs = sd.compute_rdfs_loss(dr_image, dr_depth, sd_config)
    loss = loss_rdfs / nb_acc_steps

    dr.backward(loss)

    if (i + 1) % nb_acc_steps == 0:

        opt.step()
        
        for k, v in opt.items():
            opt[k] = dr.clip(opt[k], 0, 1)

        scene_params.update(opt)
        iterator.set_postfix(loss=loss.item())

        actual_steps = (i + 1) // nb_acc_steps
        if actual_steps % nb_steps_save == 0:
            
            mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{scene_metadata["scene_name"]}_opt.exr', dr_image)
            for k, v in opt.items():
                mi.util.write_bitmap(f'outputs/{scene_metadata["scene_name"]}_tex/{k.split(".")[0]}.exr', opt[k])

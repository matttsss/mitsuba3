import torch, random

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.dragon import load_scene
from renderer import get_depth, randomize_sensor, scene_step

mi.set_variant('cuda_ad_rgb')

# Parameters
camera_to_world_key = 'sensor.to_world'
render_size = 1024
prompt="Blue dragon on a piedestal, highly detailed, directional sunlight, 4k, photorealistic, black background"
negative_prompt="" #change geometry, change shape, change pose, change structure
guidance_scale=10
num_images_per_prompt=1
cn_cond_scale=0.6

# Instantiate scene
seed = random.randint(0, 1000000)
device = torch.device('cuda')
pt_generator = torch.Generator(device=device).manual_seed(seed)
dr_generator = dr.rng(seed=seed)

sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
sd_config = sd.prep_sd(
    prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
    cn_cond_scale=cn_cond_scale, render_size=render_size
)
    
scene, scene_params = load_scene(render_size)


if True:
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



elif True:
    dr_depth = get_depth(scene, sensor=scene.sensors()[0])
    
    depth: torch.Tensor = dr_depth.torch()

    # Correct shapes
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    image = sd.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=guidance_scale,
        cn_cond_scale=cn_cond_scale,
        depth=depth
    )

    image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    mi.util.write_bitmap('outputs/dragon_sd.png', image)

else:

    from diffusers import StableDiffusion3ControlNetPipeline
    from diffusers.models import SD3ControlNetModel

    # Instantiate pipeline
    controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_model_cpu_offload()

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        controlnet_conditioning_scale=cn_cond_scale,
        control_image=depth
    ).images

    for i, img in enumerate(images):
        img.save(f'outputs/dragon_{i}.png')

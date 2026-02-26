import torch, random

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion
from scenes.dragon import load_scene
from renderer import get_depth, randomize_sensor, scene_step

mi.set_variant('cuda_ad_rgb')

# Parameters
render_size = 1024
prompt="Blue dragon on a piedestal, highly detailed, cinematic lighting, 4k, photorealistic"
negative_prompt="" #change geometry, change shape, change pose, change structure
num_inference_steps=28
guidance_scale=7.5
num_images_per_prompt=1
cn_cond_scale=0.6

# Instantiate scene
seed = random.randint(0, 1000000)
device = torch.device('cuda')
pt_generator = torch.Generator(device=device).manual_seed(seed)
dr_generator = dr.rng(seed=seed)

sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
scene, scene_params = load_scene(render_size)

if True:

    sd_config = sd.prep_sd(
        prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
        cn_cond_scale=cn_cond_scale, render_size=render_size
    )
    
    scene_step(scene, scene_params, sd, sd_config, debug_folder="outputs/")


    for key in scene_params.keys():
        if ".bsdf.reflectance.data" in key:
            obj_name = key.split(".")[0]
            mi.util.write_bitmap(f'outputs/dragon_{obj_name}_grad.png', scene_params[key].grad)

elif True:
    dr_depth = get_depth(scene, sensor=scene.sensors()[0])
    
    depth: torch.Tensor = dr_depth.torch()

    # Correct shapes
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    image = sd.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
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
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        controlnet_conditioning_scale=cn_cond_scale,
        control_image=depth
    ).images

    for i, img in enumerate(images):
        img.save(f'outputs/dragon_{i}.png')

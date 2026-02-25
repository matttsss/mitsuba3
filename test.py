import torch, random

import drjit as dr
import mitsuba as mi

from renderer import load_scene, randomize_sensor, get_frame, get_depth

mi.set_variant('cuda_ad_rgb')

# Parameters
render_size = 1024
prompt="Blue dragon on a piedestal, highly detailed, cinematic lighting, 4k, photorealistic"
negative_prompt="" #change geometry, change shape, change pose, change structure
num_inference_steps=28
guidance_scale=7.5
num_images_per_prompt=1
controlnet_conditioning_scale=0.6

# Instantiate scene
seed = random.randint(0, 1000000)
device = torch.device('cuda')
pt_generator = torch.manual_seed(seed)
dr_generator = dr.rng(seed=seed)

scene, scene_params = load_scene('../dragon/scene.xml', render_size)
#randomize_sensor(dr_generator, scene_params, 'camera.to_world')

# Render depth and RGB frame
depth = get_depth(scene, debug_path="outputs/dragon_depth.png")
image = get_frame(scene, debug_path="outputs/dragon_rgb.png")

depth: torch.Tensor = depth.torch()
depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

if True:
    from sd import StableDiffusion

    sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
    image = sd.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
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
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_image=depth
    ).images

    for i, img in enumerate(images):
        img.save(f'outputs/dragon_{i}.png')

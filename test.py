import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import random

import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel

def load_scene(path: str, sensor_size: int = 1024) -> mi.Scene:
    scene = mi.load_file(path)
    scene_params = mi.traverse(scene)

    scene_params['camera.film.size'] = [sensor_size, sensor_size]

    scene_params.update()
    return scene, scene_params

def random_look_at(generator: dr.random.Generator, target: mi.Vector3f, radius: float) -> mi.Transform4f:
    # Sample a random point on the sphere
    phi = generator.random(mi.Float, 1) * dr.two_pi
    theta = generator.normal(mi.Float, 1) * dr.inv_two_pi + dr.pi / 2

    offset = dr.sphdir(theta, phi)
    offset = mi.Vector3f(offset.y, offset.z, -offset.x)

    origin = target + offset * radius

    return mi.Transform4f.look_at(origin=origin, target=target, up=mi.Vector3f(0, 1, 0))

def get_frame(generator: dr.random.Generator, 
              scene: mi.Scene, scene_params: mi.SceneParameters, 
              sensor_to_world_key: str, debug: bool = False) -> tuple[mi.TensorXf, torch.Tensor]:
    
    scene_params[sensor_to_world_key] = random_look_at(generator, mi.Vector3f(0, 10, 0), 50)
    scene_params.update()

    image = mi.render(scene)

    depth = dr.detach(image[..., 3])
    image[..., 3] = 1.0

    if debug:
        mi.util.write_bitmap('outputs/rgb_image.exr', image)

    # Normalize depth values to the range [0, 1]
    depth /= dr.max(depth)

    # Invert depth values so that closer objects have higher values
    depth = dr.select(depth == 0, 0, 1 - depth)

    # Blur to smooth artifacts issued by the integrator
    depth = dr.convolve(
        depth, filter='box', filter_radius=3
    )

    # Convert depth to a 3-channel image and repeat it for the batch dimension
    depth: torch.Tensor = depth.torch()
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    return image, depth


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

# Render frame and extract depth
image, depth = get_frame(dr_generator, scene, scene_params, 'camera.to_world', debug=True)

if True:
    from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import retrieve_timesteps
    from sd import StableDiffusion

    with torch.no_grad():
        sd = StableDiffusion(device=device, generator=pt_generator, enable_offload=True)
        embs = sd.prep(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=guidance_scale)

        timesteps, _ = retrieve_timesteps(sd.pipe.scheduler, num_inference_steps=num_inference_steps, device=device)

        latents = sd.prepare_latents(render_size)
        for t in timesteps:
            latents = sd.latent_step(latents, depth, embs, t, controlnet_conditioning_scale=controlnet_conditioning_scale)

        image = sd.decode_latents(latents)
        image = sd.pipe.image_processor.postprocess(image, output_type='pt')

        image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        mi.util.write_bitmap('outputs/dragon_sd.png', image)

else:
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

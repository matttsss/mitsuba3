import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from gotex.depth_integrator import InfDepthIntegrator

import torch, numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3Transformer2DModel


@torch.no_grad()
def single_diffusion_step(
    pipe: StableDiffusion3ControlNetPipeline,
    latents: torch.Tensor,           # [B, C, H/8, W/8] — already in latent space
    control_image: torch.Tensor,     # [B, 3, H, W] — pixel-space image in [-1, 1]
    prompt_embeds: torch.Tensor,     # pre-computed via pipe.encode_prompt()
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    t: torch.Tensor,                 # scalar timestep, e.g. torch.tensor(500)
    guidance_scale: float = 7.0,
    controlnet_conditioning_scale: float = 0.5,
) -> torch.Tensor:
    """
    Runs exactly one denoising step with SD3 + ControlNet.

    Returns the denoised latents at t-1 (same shape as input `latents`).
    """
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    # --- Prepare control image latents ---
    vae_shift = pipe.vae.config.shift_factor
    ctrl = control_image.to(device=device, dtype=dtype)
    ctrl_latents = pipe.vae.encode(ctrl).latent_dist.sample()
    ctrl_latents = (ctrl_latents - vae_shift) * pipe.vae.config.scaling_factor

    do_cfg = guidance_scale > 1.0

    # Duplicate control latents for CFG
    if do_cfg:
        ctrl_latents = torch.cat([ctrl_latents] * 2)

    # Expand latents + timestep for CFG
    latent_input = torch.cat([latents] * 2) if do_cfg else latents
    timestep = t.expand(latent_input.shape[0]).to(device)

    # Merge positive/negative embeddings
    if do_cfg:
        full_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        full_pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        full_embeds = prompt_embeds
        full_pooled = pooled_prompt_embeds

    # --- ControlNet forward pass ---
    controlnet_cfg = pipe.controlnet.config
    if controlnet_cfg.force_zeros_for_pooled_projection:
        ctrl_pooled = torch.zeros_like(full_pooled)
        encoder_hidden = None
    else:
        ctrl_pooled = full_pooled
        encoder_hidden = full_embeds if controlnet_cfg.joint_attention_dim is not None else None

    control_block_samples = pipe.controlnet(
        hidden_states=latent_input,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden,
        pooled_projections=ctrl_pooled,
        controlnet_cond=ctrl_latents,
        conditioning_scale=controlnet_conditioning_scale,
        return_dict=False,
    )[0]

    # --- Transformer forward pass ---
    noise_pred = pipe.transformer(
        hidden_states=latent_input,
        timestep=timestep,
        encoder_hidden_states=full_embeds,
        pooled_projections=full_pooled,
        block_controlnet_hidden_states=control_block_samples,
        return_dict=False,
    )[0]

    # --- Classifier-free guidance ---
    if do_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

    # --- Scheduler step ---
    latents_out = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    return latents_out


def encode_image_to_latents(pipe: StableDiffusion3ControlNetPipeline, img_np: np.ndarray) -> torch.Tensor:
    """Converts an HxWx3 float32 [0,1] numpy image to SD3 latents."""
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    img_t = img_t * 2.0 - 1.0  # [0,1] -> [-1,1]
    latents = pipe.vae.encode(img_t).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latents


def decode_latents_to_image(pipe: StableDiffusion3ControlNetPipeline, latents: torch.Tensor) -> np.ndarray:
    """Decodes SD3 latents back to an HxWx3 float32 [0,1] numpy image."""
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(latents, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    return img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()

scene: mi.Scene = mi.load_file('../dragon/scene.xml')
scene_params = mi.traverse(scene)

film_size_key = list(filter(lambda key: "film.size" in key, scene_params.keys()))[0]
x_fov_key = list(filter(lambda key: ".x_fov" in key, scene_params.keys()))[0]
reflectance_key = list(filter(lambda key: "reflectance.data" in key, scene_params.keys()))[0]

scene_params[film_size_key] = mi.ScalarVector2u(1024, 1024)
scene_params[x_fov_key] = 20.0
scene_params[reflectance_key] *= 0.736
scene_params.update()

image = mi.render(scene).numpy()
image = np.where(np.isfinite(image), image, np.nan)
image /= np.nanmax(image)
mi.util.write_bitmap('outputs/render_dragon.exr', image)

image = np.where(np.isnan(image), 0, 1 - image)
mi.util.write_bitmap('outputs/render_dragon_inv.exr', image)

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

images = pipe(
    prompt="Blue dragon on a piedestal, highly detailed, cinematic lighting, 4k, photorealistic", 
    negative_prompt="change geometry, change shape, change pose, change structure",
    num_inference_steps=28,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    controlnet_conditioning_scale=0.5,
    control_image=image
).images

for i, img in enumerate(images):
    img.save(f'outputs/dragon_{i}.png')

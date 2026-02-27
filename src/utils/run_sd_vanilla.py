import torch

from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel

device = 'cuda'

# Instantiate pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, torch_dtype=torch.float16
).to(device)

pipe.enable_model_cpu_offload()


pipe(prompt="Blue dragon on a piedestal, highly detailed, directional sunlight, 4k, photorealistic, grassy hills background",
     control_image=torch.randn(1, 3, 1024, 1024).to(device))
import os
import random
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import numpy as np

from gotex.models.instaflow import Instaflow
from gotex.models.sd import StableDiffusion

import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

config = {
    "max_iters": 800,
    "seed": 1,
    "scheduler": None,
    "mode": "rgb", # "rgb" or "latent"

    "prompt": "A DSLR image of a hamburger",
    "guidance": {
        "guidance_scale": 50,
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
    },
    "render_size": 512,
    "n_accumulation_steps": 2,
    "save_interval": 50,
    "clip": True,
    "tanh": False,
    "lr": {
        "image": 3e-2,
    },
}

seed_everything(config["seed"])

sd_config = dict(
    prompt=config["prompt"],
    negative_prompt="",
    cn_cond_scale=0,
    render_size=config["render_size"],
    guidance_scale=config["guidance"]["guidance_scale"],
    min_time=config["guidance"]["min_step_percent"],
    max_time=config["guidance"]["max_step_percent"]
)

torch_generator = torch.Generator(device='cuda').manual_seed(config["seed"])
if True:
    sd = StableDiffusion(config=sd_config, device='cuda', generator=torch_generator, enable_offload=False)
else:
    sd = Instaflow(config=sd_config, instaflow=False, device='cuda', generator=torch_generator, enable_offload=False)

mode = config["mode"]
render_size = config["render_size"]
if mode == "rgb":
    target = nn.Parameter(torch.rand(1, 3, render_size, render_size, device=sd.device, dtype=torch.float32))
else:
    target = nn.Parameter(2 * torch.rand(1, 4 if isinstance(sd, Instaflow) else 16, render_size, render_size, device=sd.device, dtype=torch.float32) - 1)


optimizer = torch.optim.AdamW(
    [
        {"params": [target], "lr": config["lr"]["image"]},
    ],
    weight_decay=0,
)

out_dir = os.path.join("outputs", f"2d_RFDS")
os.makedirs(out_dir, exist_ok=True)
img_name = f"save_{sd.__class__.__name__}_{mode}.exr"

num_steps = config["max_iters"]
save_interval = config["save_interval"]
n_accumulation_steps = config["n_accumulation_steps"]
for step in tqdm(range(num_steps * n_accumulation_steps + 1)):

    if mode == "rgb":
        latents = sd.encode_image(target).float()
    elif mode == "latent":
        latents = torch.nn.functional.interpolate(target, size=(64, 64), mode='bilinear', align_corners=False)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    loss = sd.compute_rdfs_loss_torch(latents, None) / n_accumulation_steps
    loss.backward()

    if (step + 1) % n_accumulation_steps == 0:
        actual_step = (step + 1) // n_accumulation_steps

        optimizer.step()
        optimizer.zero_grad()

        if config["clip"]:
            with torch.no_grad():
                target.data = target.data.clip(0, 1) if mode == "rgb" else target.data.clip(-1, 1)

        if actual_step % save_interval == 0:
            if mode == "rgb":
                rgb = target
            else:
                del loss
                torch.cuda.empty_cache()
                with torch.no_grad():
                    latents = torch.nn.functional.interpolate(target, size=(64, 64), mode='bilinear', align_corners=False)
                    rgb = sd.decode_latents(latents)

            mi.util.write_bitmap(os.path.join(out_dir, img_name), rgb.squeeze(0).permute(1, 2, 0))

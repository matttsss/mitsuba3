import json
import os
import random
from datetime import datetime
from tqdm.auto import tqdm

import numpy as np
from sd import SDConfig, StableDiffusion
import torch
import torch.nn as nn

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
    "prompt_processor_type": "sd3-prompt-processor",
    "prompt_processor": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "prompt": "A DSLR image of a hamburger",
        "spawn": False,
    },
    "guidance_type": "RFDS-sd3",
    "guidance": {
        "half_precision_weights": True,
        "view_dependent_prompting": True,
        "guidance_scale": 50,
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "camera_condition_type": "extrinsics",
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

sd_config = SDConfig(
    prompt=config["prompt_processor"]["prompt"],
    negative_prompt="",
    cn_cond_scale=0,
    render_size=config["render_size"],
    guidance_scale=config["guidance"]["guidance_scale"],
    min_time=config["guidance"]["min_step_percent"],
    max_time=config["guidance"]["max_step_percent"]
)
sd = StableDiffusion(config=sd_config, device='cuda', generator=torch.Generator(device='cuda').manual_seed(config["seed"]), enable_offload=False)

mode = config["mode"]
render_size = config["render_size"]
if mode == "rgb":
    target = nn.Parameter(torch.rand(1, 3, render_size, render_size, device=sd.device, dtype=torch.float32))
else:
    target = nn.Parameter(2 * torch.rand(1, 16, render_size, render_size, device=sd.device, dtype=torch.float32) - 1)


optimizer = torch.optim.AdamW(
    [
        {"params": [target], "lr": config["lr"]["image"]},
    ],
    weight_decay=0,
)

# add time to out_dir
timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
out_dir = os.path.join(
    "outputs", "2d_RFDS_sd3", f"{config['prompt_processor']['prompt']}{timestamp}"
)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

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

    time = torch.rand(1, generator=sd.generator, device=sd.device) * (sd_config.max_time - sd_config.min_time) + sd_config.min_time
    noise = torch.randn_like(latents, generator=sd.generator, device=sd.device)
    latents_noisy = time * noise + (1.0 - time) * latents

    predicted_vel = sd.predict_velocity(
        latents_noisy, depth=None, timestep=time * sd.pipe.scheduler.config.num_train_timesteps
    )
    predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

    loss_rfds = torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")
    loss_rdfs = loss_rfds.mean()
    loss = loss_rdfs / n_accumulation_steps
    loss.backward()


    # wandb.log({"loss": loss_dict["loss_vsd"]})
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

            mi.util.write_bitmap(os.path.join(out_dir, f"save.exr"), rgb.squeeze(0).permute(1, 2, 0))

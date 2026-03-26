import argparse
import os
import random

import torch
from tqdm.auto import trange

import drjit as dr
import mitsuba as mi

from gotex.models.prompt_encoder import PromptEncoder
from gotex.utils import load_scene
from gotex.config import ExperimentConfig, load_config
from gotex.models.sd import StableDiffusion
from gotex.trainer import Trainer

def main(args):
    random.seed(args.seed)
    device = torch.device(args.device)
    mi.set_variant("cuda_ad_rgb" if "cuda" in args.device else "llvm_ad_rgb")

    config: ExperimentConfig = load_config(*args.config_files)

    scene = load_scene(config.scene, config.checkpoint)
    sd = StableDiffusion(
        config=config.guidance,
        device=config.device,
    )
    prompt_encoder = PromptEncoder(config.prompt_processor, device=device, dtype=sd.transformer.dtype)
        
    trainer = Trainer(
        config=config.trainer,
        camera_config=config.camera,
        scene=scene,
        guidance=sd,
        prompt_processor=prompt_encoder,
        seed=args.seed
    )

    out_folder = f'{config.exp_root_dir}/{config.name}'
    os.makedirs(out_folder, exist_ok=True)

    iterator = trange(args.nb_opt_steps, desc="Optimizing", disable=args.disable_tqdm)
    for step_idx in iterator:
        
        image, loss = trainer.step()            

        if step_idx % args.nb_steps_save == 0:
            if args.disable_tqdm:
                dr.print("step {step_idx} | {loss=}", step_idx=step_idx, loss=loss)
            else:
                iterator.set_postfix(loss=loss.item())
            
            images = image.torch().permute(1, 0, 2, 3)
            for i, img in enumerate(images):
                mi.util.write_bitmap(os.path.join(out_folder, f'render_{i}.exr'), img)

            for k, v in trainer.opt.items():
                mi.util.write_bitmap(os.path.join(out_folder, f'{k.split(".")[0]}.exr'), v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize scene texture parameters with Stable Diffusion guidance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (e.g. cuda or cpu).")
    parser.add_argument("--render-size", type=int, default=512, help="Render resolution.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Optimizer learning rate.")
    parser.add_argument("--nb-sensors", type=int, default=5, help="Number of camera sensors for randomization.")
    parser.add_argument("--nb-opt-steps", type=int, default=8000, help="Number of optimization steps.")
    parser.add_argument("--nb-steps-save", type=int, default=100, help="Save outputs every N optimization steps.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar output.")
    
    parser.add_argument("config_files", nargs="*", help="YAML configuration files to load.")
    args = parser.parse_args()
    # Set the Mitsuba variant based on the device

    main(args)

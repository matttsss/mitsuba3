import argparse
import os
import random

import torch
from tqdm.auto import trange

import drjit as dr
import mitsuba as mi

from models.sd import StableDiffusion
from scenes.dragon import load_scene
from renderer import random_transform, get_depth

def main(args):
    random.seed(args.seed)
    device = torch.device(args.device)
    dr_generator = dr.rng(args.seed)
    pt_generator = torch.Generator(device=device).manual_seed(args.seed)

    scene_config = load_scene(args.render_size, nb_sensors=args.nb_sensors)
    camera_config = scene_config['camera_config']
    sd_config = scene_config['sd_config']

    scene = mi.load_dict(scene_config['scene'], optimize=False)
    scene_params: mi.SceneParameters = mi.traverse(scene)
    camera_params: mi.SceneParameters = scene_params.copy()

    scene_params.keep(r'.*\.reflectance\.data')
    camera_params.keep(r'.*\.sensor_\d*\.to_world')

    sd = StableDiffusion(config=sd_config, device=device, generator=pt_generator, enable_offload=False)

    opt = mi.ad.Adam(lr=args.lr, params=scene_params)
    scene_params.update(opt)

    out_folder = f'outputs/{scene_config["scene_name"]}'
    os.makedirs(out_folder, exist_ok=True)

    iterator = trange(args.nb_opt_steps, desc="Optimizing", disable=args.disable_tqdm)
    for step_idx in iterator:
        if not camera_config['is_2d']:
            for k , _ in camera_params:
                camera_params[k] = random_transform(dr_generator, camera_config)
            camera_params.update()

        dr_depth = get_depth(scene, sensor=scene.sensors()[0])
        dr_image = mi.render(scene, params=scene_params, seed=step_idx)

        # Simple tone mapping for Stable Diffusion input
        dr_image = dr_image / (dr_image + 1)
        dr_image = dr_image ** (1/2.2)
        dr_image = dr.clip(dr_image, 0, 1)

        dr_depth = dr.reshape(mi.TensorXf, dr_depth, (args.render_size, args.nb_sensors, args.render_size))
        dr_image = dr.reshape(mi.TensorXf, dr_image, (args.render_size, args.nb_sensors, args.render_size, 3))

        loss = sd.compute_rdfs_loss(dr_image, dr_depth)
        dr.backward(loss)
        opt.step()

        for k, v in opt.items():
            opt[k] = dr.clip(v, 0, 1)

        scene_params.update(opt)
        if not args.disable_tqdm:
            iterator.set_postfix(loss=loss.item())

        if step_idx == 2000:
            sd.set_min_max_time(0.02, 0.7)

        if step_idx % args.nb_steps_save == 0:
            if args.disable_tqdm:
                print(f"step {step_idx:06d} | loss={loss.item():.6f}")
            
            images = dr_image.torch().permute(1, 0, 2, 3)
            for i, img in enumerate(images):
                mi.util.write_bitmap(os.path.join(out_folder, f'render_{i}.exr'), img)

            for k, v in opt.items():
                mi.util.write_bitmap(os.path.join(out_folder, f'{k.split(".")[0]}.exr'), opt[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize scene texture parameters with Stable Diffusion guidance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (e.g. cuda or cpu).")
    parser.add_argument("--render-size", type=int, default=512, help="Render resolution.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Optimizer learning rate.")
    parser.add_argument("--nb-sensors", type=int, default=8, help="Number of camera sensors for randomization.")
    parser.add_argument("--nb-opt-steps", type=int, default=8000, help="Number of optimization steps.")
    parser.add_argument("--nb-steps-save", type=int, default=100, help="Save outputs every N optimization steps.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar output.")
    args = parser.parse_args()

    # Set the Mitsuba variant based on the device
    mi.set_variant("cuda_ad_rgb" if "cuda" in args.device else "llvm_ad_rgb")

    main(args)

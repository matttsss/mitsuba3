import argparse
import os
import random

import torch
from tqdm.auto import trange

import drjit as dr
import mitsuba as mi

from models.sd import StableDiffusion
from scenes.dragon import load_scene
from renderer import randomize_sensor, get_depth

def main(args):
    random.seed(args.seed)
    device = torch.device(args.device)
    pt_generator = torch.Generator(device=device).manual_seed(args.seed)

    scene, scene_params, scene_metadata, sd_config = load_scene(args.render_size)
    sd = StableDiffusion(config=sd_config, device=device, generator=pt_generator, enable_offload=False)

    camera_to_world_key = 'sensor.to_world'
    scene_params.keep([r'.*\.reflectance\.data', camera_to_world_key])
    opt = mi.ad.Adam(lr=args.lr, params={
        k: v for k, v in scene_params.items() if camera_to_world_key not in k
    })
    scene_params.update(opt)

    out_folder = f'outputs/{scene_metadata["scene_name"]}'
    os.makedirs(out_folder, exist_ok=True)

    total_steps = args.nb_opt_steps * args.nb_acc_steps
    iterator = range(total_steps) if args.disable_tqdm else trange(total_steps, desc="Optimizing")
    for i in iterator:
        if not scene_metadata['is_2d']:
            randomize_sensor(
                scene_params,
                sensor_to_world_key=camera_to_world_key,
                sensor_idx=random.randint(0, args.nb_sensors - 1),
                sensor_count=args.nb_sensors,
                target=scene_metadata['target'],
                radius=scene_metadata['radius'],
            )

        dr_depth = get_depth(scene, sensor=scene.sensors()[0])
        dr_image = mi.render(scene, params=scene_params, seed=i)

        loss_rdfs = sd.compute_rdfs_loss(dr_image, dr_depth)
        loss = loss_rdfs / args.nb_acc_steps

        dr.backward(loss)

        if (i + 1) % args.nb_acc_steps == 0:
            actual_steps = (i + 1) // args.nb_acc_steps

            opt.step()

            for k, v in opt.items():
                opt[k] = dr.clip(v, 0, 1)

            scene_params.update(opt)
            if not args.disable_tqdm:
                iterator.set_postfix(loss=loss.item())

            if actual_steps == 5000:
                sd.set_min_max_time(0.02, 0.7)

            if actual_steps % args.nb_steps_save == 0:
                if args.disable_tqdm:
                    print(f"step {actual_steps:06d} | loss={loss.item():.6f}")
                mi.util.write_bitmap(os.path.join(out_folder, 'render.exr'), dr_image)
                for k, v in opt.items():
                    mi.util.write_bitmap(os.path.join(out_folder, f'{k.split(".")[0]}.exr'), opt[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize scene texture parameters with Stable Diffusion guidance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (e.g. cuda or cpu).")
    parser.add_argument("--render-size", type=int, default=512, help="Render resolution.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Optimizer learning rate.")
    parser.add_argument("--nb-sensors", type=int, default=512, help="Number of camera sensors for randomization.")
    parser.add_argument("--nb-opt-steps", type=int, default=15000, help="Number of optimization steps.")
    parser.add_argument("--nb-acc-steps", type=int, default=2, help="Number of gradient accumulation steps.")
    parser.add_argument("--nb-steps-save", type=int, default=100, help="Save outputs every N optimization steps.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar output.")
    args = parser.parse_args()

    # Set the Mitsuba variant based on the device
    mi.set_variant("cuda_ad_rgb" if "cuda" in args.device else "llvm_ad_rgb")

    main(args)

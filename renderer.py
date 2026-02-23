from __future__ import annotations

import torch

import drjit as dr
import mitsuba as mi

def load_scene(path: str, sensor_size: int = 1024) -> tuple[mi.Scene, mi.SceneParameters]:
    scene = mi.load_file(path)
    scene_params = mi.traverse(scene)

    scene_params['camera.film.size'] = [sensor_size, sensor_size]

    scene_params.update()
    return scene, scene_params

def random_look_at(generator: dr.random.Generator, target: mi.ScalarVector3f, radius: float) -> mi.Transform4f:
    # Sample a random azimuth uniformly
    phi = generator.random(mi.ScalarFloat, 1) * dr.two_pi
    # Sample near the horizon
    theta = generator.normal(mi.ScalarFloat, 1) * dr.inv_two_pi + dr.pi / 2

    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    offset = mi.ScalarVector3f(st * sp, ct, -cp * st)

    origin = target + offset * radius

    return mi.Transform4f.look_at(origin=origin, target=target, up=mi.ScalarVector3f(0, 1, 0))

def get_frame(generator: dr.random.Generator, 
              scene: mi.Scene, scene_params: mi.SceneParameters, 
              sensor_to_world_key: str, debug_path: str = "") -> tuple[torch.Tensor, torch.Tensor]:
    
    scene_params[sensor_to_world_key] = random_look_at(generator, mi.ScalarVector3f(0, 10, 0), 50)
    scene_params.update()

    image = mi.render(scene)

    depth = dr.detach(image[..., 3])
    image = image[..., :3]

    if debug_path != "":
        mi.util.write_bitmap(debug_path, image)

    # Normalize depth values to the range [0, 1]
    depth /= dr.max(depth)

    # Invert depth values so that closer objects have higher values
    depth = dr.select(depth == 0, 0, 1 - depth)

    # Blur to smooth artifacts issued by the integrator
    depth = dr.convolve(
        depth, filter='box', filter_radius=3
    )

    # Convert depth to a 3-channel image and repeat it for the channel dimension
    depth: torch.Tensor = depth.torch()
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    # Permute image to CxHxW and add the batch dimension
    image: torch.Tensor = image.torch()
    image = image.permute(2, 0, 1).unsqueeze(0)

    return image, depth

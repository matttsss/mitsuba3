from __future__ import annotations

import torch

import drjit as dr
import mitsuba as mi

def randomize_sensor(generator: dr.random.Generator, scene_params: mi.SceneParameters, 
                     sensor_to_world_key: str, target: mi.ScalarVector3f = [0, 10, 0], radius: float = 50) -> None:
    # Sample a random azimuth uniformly
    phi = generator.random(mi.ScalarFloat, 1) * dr.two_pi
    # Sample near the horizon
    theta = generator.normal(mi.ScalarFloat, 1) * dr.inv_two_pi + dr.pi / 2

    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    offset = mi.ScalarVector3f(st * sp, ct, -cp * st)

    origin = target + offset * radius

    scene_params[sensor_to_world_key] = mi.Transform4f.look_at(origin=origin, target=target, up=mi.ScalarVector3f(0, 1, 0))
    scene_params.update()

@dr.freeze
def get_depth(scene: mi.Scene, debug_path: str = "") -> mi.TensorXf:
    film = scene.sensors()[0].film()
    film_size = film.size()

    idx = dr.arange(mi.Int32, film_size[0] * film_size[1])
    pos_y = idx // film_size[0]
    pos = mi.Vector2f(mi.ScalarInt32(-film_size[0]) * pos_y + idx, pos_y)

    if film.sample_border():
        pos -= film.rfilter().border_size()

    pos += film.crop_offset()

    scale = 1. / mi.ScalarVector2f(film.crop_size())
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale

    sample_pos = pos + mi.ScalarVector2f(0.5, 0.5)
    adjusted_pos = dr.fma(sample_pos, scale, offset)

    ray, _ = scene.sensors()[0].sample_ray_differential(
        time=0, sample1=0, sample2=adjusted_pos, sample3=0.5)

    pi = scene.ray_intersect_preliminary(ray, coherent=True)

    depth = dr.select(pi.is_valid(), pi.t, 0.0)
    depth /= dr.max(depth)
    depth = dr.select(depth == 0, 0, 1 - depth)

    depth = mi.TensorXf(depth, shape=(film_size[1], film_size[0]))

    if debug_path != "":
        mi.util.write_bitmap(debug_path, depth)

    return depth


def get_frame(scene: mi.Scene, debug_path: str = "") -> torch.Tensor:

    image = mi.render(scene)
    image = image[..., :3]

    if debug_path != "":
        mi.util.write_bitmap(debug_path, image)

    # Permute image to CxHxW and add the batch dimension
    image: torch.Tensor = image.torch()
    return image.permute(2, 0, 1).unsqueeze(0)
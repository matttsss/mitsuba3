from __future__ import annotations

import drjit as dr
import mitsuba as mi

def randomize_sensor(scene_params: mi.SceneParameters, sensor_to_world_key: str, 
                     sensor_idx: int, sensor_count: int, 
                     target: mi.ScalarVector3f = [0, 10, 0], radius: float = 50) -> None:

    golden_ratio = (1 + 5**0.5)/2
    phi = 2 * dr.pi * sensor_idx / golden_ratio
    theta = dr.acos(1 - 2*(sensor_idx+0.5)/sensor_count)

    offset = mi.ScalarVector3f(
        dr.sin(phi) * dr.sin(theta),
        dr.abs(dr.cos(theta)),
        -dr.cos(phi) * dr.sin(theta),
    )

    origin = target + offset * radius

    scene_params[sensor_to_world_key] = mi.ScalarTransform4f.look_at(origin=origin, target=target, up=[0, 1, 0])
    scene_params.update()

@dr.freeze
def get_depth(scene: mi.Scene, sensor: mi.Sensor) -> mi.TensorXf:
    with dr.suspend_grad():
        film = sensor.film()
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

        return mi.TensorXf(depth, shape=(film_size[1], film_size[0]))

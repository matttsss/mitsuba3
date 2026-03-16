from __future__ import annotations

import random

import drjit as dr
import mitsuba as mi

def create_sensors(nb_sensors: int, render_size: int, fov: float = 30.0) -> tuple[mi.Sensor, mi.SceneParameters]:
    if nb_sensors == 1:
        sensor_dict = {
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f(),
        }
    else: 
        sensor_dict = {
            'type': 'batch',
            **{
                f'sensor_{i}': {
                    'type': 'perspective',
                    'fov': fov,
                    'to_world': mi.ScalarTransform4f()
                } for i in range(nb_sensors)
            },
        }


    sensor_dict.update({ 
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },

        'film': {
            'type': 'hdrfilm',
            'width' : nb_sensors * render_size,
            'height': render_size,
            'rfilter': {
                'type': 'gaussian',
            },
            'pixel_format': 'rgb',
            'component_format': 'float32',
        }
    })
    sensor = mi.load_dict(sensor_dict)
    sensor_params = mi.traverse(sensor)
    sensor_params.keep(r'.*\.to_world')
    
    return sensor, sensor_params

def random_transform(rng: dr.random.Generator, camera_config: dict) -> mi.ScalarTransform4f:
    # if random.random() < 0.5:
    #         elevation_deg = rng.uniform(mi.ScalarFloat, shape=1, 
    #                                          low=camera_config['elevation_min'], high=camera_config['elevation_max'])
    #         elevation = dr.deg2rad(elevation_deg)
    # else:
    #     elev_percent_low = (camera_config['elevation_min'] + 90.0) / 180.0
    #     elev_percent_high = (camera_config['elevation_max'] + 90.0) / 180.0
    #     u = rng.uniform(mi.ScalarFloat, shape=1, low=elev_percent_low, high=elev_percent_high)
    #     elevation = dr.asin(2.0 * u - 1.0)

    elevation_min_rad = dr.deg2rad(camera_config['elevation_min'])
    elevation_max_rad = dr.deg2rad(camera_config['elevation_max'])
    elevation = rng.uniform(mi.ScalarFloat, shape=1, low=dr.cos(elevation_min_rad), high=dr.cos(elevation_max_rad))
    azimuth = rng.uniform(mi.ScalarFloat, shape=1, low=-dr.pi, high=dr.pi)

    camera_distances = rng.uniform(
        mi.ScalarFloat, shape=1, 
        low=camera_config['radius_min'], high=camera_config['radius_max']
    )

    camera_perturb = 0.05

    sp, cp = dr.sincos(azimuth)
    st, ct = dr.sincos(elevation)
    camera_positions = camera_distances * mi.ScalarVector3f(
        sp * st, ct, -cp * st
    )
    camera_positions += rng.uniform(mi.ScalarVector3f, shape=(3,), 
                                        low=-camera_perturb, high=camera_perturb)

    center_perturb = 0.1
    center = camera_config['target'] + rng.normal(mi.ScalarVector3f, shape=(3,), scale=center_perturb)

    up_perturb = 0.02
    up = rng.normal(mi.ScalarVector3f, shape=(3,), loc=mi.ScalarVector3f(0.0, 1.0, 0.0), scale=up_perturb)

    return mi.ScalarTransform4f.look_at(camera_positions, center, up)


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

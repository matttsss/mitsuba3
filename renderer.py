from __future__ import annotations

import torch

import drjit as dr
import mitsuba as mi

from sd import StableDiffusion

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


#@dr.freeze
def scene_step(scene: mi.Scene, scene_params: mi.SceneParameters,
               sd: StableDiffusion, sd_config: dict, debug_folder: str = "") -> torch.Tensor:

    # TODO fixme
    # @dr.wrap(source=torch, target=dr)
    # def wraper(scene, scene_params) -> torch.Tensor:
    #     return mi.render(scene, params=scene_params)
    
    # image = wraper(scene, scene_params)

    dr_depth = get_depth(scene, sensor=scene.sensors()[0])
    dr_image = mi.render(scene, params=scene_params)
    
    if debug_folder != "":
        mi.util.write_bitmap(debug_folder + "dragon_depth.png", dr_depth)
        mi.util.write_bitmap(debug_folder + "dragon_rgb.png", dr_image)

    image: torch.Tensor = dr_image.torch()
    depth: torch.Tensor = dr_depth.torch()

    # Correct shapes
    image = image.permute(2, 0, 1).unsqueeze(0)
    depth = depth.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    # Stich gradients
    image.requires_grad_(dr.grad_enabled(dr_image))
    latent_img = sd.encode_image(image)

    noise = torch.randn_like(latent_img, generator=sd.generator, device=sd.device)
    timestep = torch.rand(1, generator=sd.generator, device=sd.device)

    noisy_latents = (1 - timestep) * latent_img + timestep * noise
    velocity = sd.predict_velocity(
        noisy_latents, depth, sd_config, timestep
    )
    
    u = torch.randn(size=(1,), generator=sd.generator, device=sd.device)
    weighting = torch.nn.functional.sigmoid(u)

    grad = torch.nan_to_num(velocity)

    target = grad.detach()
    loss_rfds = weighting * torch.nn.functional.mse_loss(noise - latent_img, target, reduction="mean") / 1 # batch_size
    loss_rdfs = loss_rfds.mean()

    loss_rdfs.backward()

    dr.set_grad(dr_image, image.grad.squeeze(0).permute(1, 2, 0))
    dr.backward(dr_image)

    return loss_rdfs.item()

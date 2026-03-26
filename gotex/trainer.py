from __future__ import annotations

import torch
import random

import drjit as dr
import mitsuba as mi

from gotex.models.prompt_encoder import PromptEncoder
from .models.sd import StableDiffusion

class Trainer:

    def __init__(self, scene_params: mi.SceneParameters, camera_config: dict, sd_config: dict, device: str, seed: int = 42):
        self.camera_config = camera_config
        self.generator = dr.rng(seed)

        print("Loading Stable Diffusion model...")
        self.sd = StableDiffusion(
            config=sd_config,
            device=device,
            enable_offload=False
        )
        print("Stable Diffusion model loaded.")

        self.lr = 3e-2
        self._step_idx = 0

        self.directions = ('side', 'front', 'back', 'overhead')

        self.prompt = self.sd.config['prompt']
        self.negative_prompt = self.sd.config["negative_prompt"]
        self.directional_prompts = {
                direction: f"{self.prompt}, {direction}" for direction in self.directions
        }
        prompts = list(self.directional_prompts.values()) + [self.negative_prompt]
        self.prompt_encoder = PromptEncoder(prompts=prompts, device=device, dtype=self.sd.transformer.dtype)

        scene_params.keep(r'.*\.reflectance\.data')
        self.opt = mi.ad.Adam(lr=self.lr, params=scene_params)
        scene_params.update(self.opt)

        self.opt_sensor = None
        self.opt_sensor_params = None

    @property
    def step_idx(self):
        return self._step_idx

    def step(self, scene: mi.Scene, scene_params: mi.SceneParameters) -> tuple[mi.TensorXf, float]:
        # Randomize camera viewpoint for 3D scenes
        if not self.camera_config['is_2d']:
            prompts = []
            # For every sensor, sample camera dir and get it's associated prompt embeddings
            for sensor_idx, k in enumerate(self.opt_sensor_params.keys()):
                transform, direction_label = self.random_transform(sensor_idx)
                self.opt_sensor_params[k] = transform

                prompts.append(self.directional_prompts[direction_label])

            self.opt_sensor_params.update()
        else:
            prompts = [self.prompt]  # Single prompt for 2D scenes

        # Render depth and image
        dr_depth = Trainer.get_depth(scene, sensor=self.opt_sensor)
        dr_image = mi.render(scene, params=scene_params, sensor=self.opt_sensor, seed=self._step_idx)

        dr_depth = dr.reshape(mi.TensorXf, dr_depth, (self.render_size, self.nb_sensors, self.render_size))
        dr_image = dr.reshape(mi.TensorXf, dr_image, (self.render_size, self.nb_sensors, self.render_size, 3))

        # Simple tone mapping for Stable Diffusion input
        dr_image = dr_image / (dr_image + 1)
        dr_image = dr_image ** (1/2.2)
        # dr_image = dr.clip(dr_image, 0, 1)
    
        loss = self.compute_loss(dr_image, dr_depth, prompts)

        # Backward pass
        dr.backward(loss)
        self.opt.step()
        
        # Clip values to valid range
        for k, v in self.opt.items():
            self.opt[k] = dr.clip(v, 0, 1)
        
        scene_params.update(self.opt)
        self._step_idx += 1

        return dr_image, loss
    
    @dr.wrap("drjit", "torch")
    def compute_loss(self, image: torch.Tensor, depth: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert (H, W, C) to (1, C, H, W)
            depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)
        elif image.ndim == 4:
            # Don't ask about the weird order, it's the batch sensor 
            image = image.permute(1, 3, 0, 2)  # Convert (H, B, W, C) to (B, C, H, W)
            depth = depth.detach().unsqueeze(-1).repeat(1, 1, 1, 3).permute(1, 3, 0, 2)  # Convert to (B, 3, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        prompt_embedings = self.prompt_encoder.encode_prompt(
            prompt=prompts,
            negative_prompt=self.negative_prompt,
            images=image,
        )

        # Compute loss using Stable Diffusion
        return self.sd.compute_rdfs_loss(prompt_embedings, image, depth)

    def _view_to_direction(self, azimuth: float, inclination: float) -> str:
        # Prioritize top-down views, then map azimuth to front/side/back buckets.
        if inclination <= dr.pi / 4:
            return 'overhead'

        if -dr.pi / 4 <= azimuth < dr.pi / 4:
            return 'front'
        if azimuth >= 3 * dr.pi / 4 or azimuth < -3 * dr.pi / 4:
            return 'back'
        return 'side'
    

    def setup_opt_sensors(self, nb_sensors: int, render_size: int, fov: float = 30.0):
        self.nb_sensors = nb_sensors
        self.render_size = render_size

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
        
        self.opt_sensor = mi.load_dict(sensor_dict, optimize=False)
        self.opt_sensor_params = mi.traverse(self.opt_sensor)
        self.opt_sensor_params.keep(r'.*to_world')

    def random_transform(self, sensor_idx: int) -> tuple[mi.ScalarTransform4f, str]:
        if random.random() < 0.5:
                elevation_deg = self.generator.uniform(mi.ScalarFloat, shape=1, 
                                                 low=self.camera_config['elevation_min'], high=self.camera_config['elevation_max'])
                elevation = dr.deg2rad(elevation_deg)
        else:
            elev_percent_low = (self.camera_config['elevation_min'] + 90.0) / 180.0
            elev_percent_high = (self.camera_config['elevation_max'] + 90.0) / 180.0
            u = self.generator.uniform(mi.ScalarFloat, shape=1, low=elev_percent_low, high=elev_percent_high)
            elevation = dr.asin(2.0 * u - 1.0)

        if True:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth = (
                sensor_idx + self.generator.uniform(mi.ScalarFloat, shape=1)
            ) / self.nb_sensors * dr.two_pi

        else:
            # simple random sampling
            azimuth = self.generator.uniform(mi.ScalarFloat, shape=1, low=0.0, high=dr.two_pi)
        
        elevation = 0.5 * dr.pi - elevation  # Convert to inclination
        direction_label = self._view_to_direction(azimuth, elevation)

        camera_distances = self.generator.uniform(
            mi.ScalarFloat, shape=1, 
            low=self.camera_config['radius_min'], high=self.camera_config['radius_max']
        )

        camera_perturb = self.camera_config['camera_perturb']

        st, ct = dr.sincos(elevation)
        sp, cp = dr.sincos(azimuth)
        camera_positions = camera_distances * mi.ScalarVector3f(
            sp * st, ct, -cp * st
        )
        camera_positions += self.generator.uniform(mi.ScalarVector3f, shape=(3,), 
                                            low=-camera_perturb, high=camera_perturb)

        center_perturb = self.camera_config['center_perturb']
        center = self.camera_config['target'] + self.generator.normal(mi.ScalarVector3f, shape=(3,), scale=center_perturb)

        up_perturb = self.camera_config['up_perturb']
        up = self.generator.normal(mi.ScalarVector3f, shape=(3,), loc=mi.ScalarVector3f(0.0, 1.0, 0.0), scale=up_perturb)

        return mi.ScalarTransform4f.look_at(camera_positions, center, up), direction_label
    
    @staticmethod
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

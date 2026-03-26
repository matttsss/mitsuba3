from __future__ import annotations

from dataclasses import dataclass, field

import torch
import random

import drjit as dr
import mitsuba as mi

from gotex.config import Configurable, RuntimeContext, parse_structured
from gotex.models.sd import StableDiffusion

from .models.prompt_encoder import PromptEncoder
from .utils import load_scene

class Trainer(Configurable):

    @dataclass
    class Config(Configurable.Config):
        lr: float = 0.01
        max_steps: int = 1500
        save_every: int = 50
        opt_param_names: list[str]= field(default_factory=lambda: [r'.*\.reflectance\.data'])

        scene: str = ""
        guidance: dict = field(default_factory=dict)
        camera: dict = field(default_factory=dict)
        prompt_processor: dict = field(default_factory=dict)

    @dataclass
    class CameraConfig(Configurable.Config):
        is_2d: bool = False
        nb_sensors: int = 1
        spp: int = 16
        fov: float = 30.0
        render_size: int = 512
        radius_min: float = 1.0
        radius_max: float = 1.0
        elevation_min: float = -5.0
        elevation_max: float = 45.0
        target: tuple[float, float, float] = (0.0, 0.0, 0.0)
        camera_perturb: float = 0.0
        center_perturb: float = 0.0
        up_perturb: float = 0.0

    def __init__(self, config: dict, runtime: RuntimeContext):
        super().__init__(config, runtime=runtime)
        self.generator = runtime.dr_generator
        self.camera_cfg = parse_structured(self.CameraConfig, self.cfg.camera)

        self.scene = load_scene(self.cfg.scene)
        self.scene_params: mi.SceneParameters = mi.traverse(self.scene)
        self.guidance = StableDiffusion(self.cfg.guidance, runtime=self.runtime)
        self.prompt_processor = PromptEncoder(
            self.cfg.prompt_processor,
            runtime=self.runtime,
        )

        self._step_idx = 0

        self.scene_params.keep(list(self.cfg.opt_param_names))
        self.opt = mi.ad.Adam(lr=self.cfg.lr, params=self.scene_params)
        self.scene_params.update(self.opt)

        self.setup_opt_sensors()

    @property
    def step_idx(self):
        return self._step_idx

    def step(self) -> tuple[mi.TensorXf, float]:
        # Randomize camera viewpoint for 3D scenes
        camera_angles = None # If 2D
        if not self.camera_cfg.is_2d:
            camera_angles = []
            # For every sensor, sample camera dir and get it's associated prompt embeddings
            for sensor_idx, k in enumerate(self.opt_sensor_params.keys()):
                transform, cam_angles = self.random_transform(sensor_idx)
                self.opt_sensor_params[k] = transform

                camera_angles.append(cam_angles)

            self.opt_sensor_params.update()

        # Render depth and image
        dr_depth = Trainer.get_depth(self.scene, sensor=self.opt_sensor)
        dr_image = mi.render(self.scene, params=self.scene_params, sensor=self.opt_sensor, seed=self._step_idx)

        dr_depth = dr.reshape(mi.TensorXf, dr_depth, (self.camera_cfg.render_size, self.camera_cfg.nb_sensors, self.camera_cfg.render_size))
        dr_image = dr.reshape(mi.TensorXf, dr_image, (self.camera_cfg.render_size, self.camera_cfg.nb_sensors, self.camera_cfg.render_size, 3))

        # Simple tone mapping for Stable Diffusion input
        dr_image = dr_image / (dr_image + 1)
        dr_image = dr_image ** (1/2.2)
        # dr_image = dr.clip(dr_image, 0, 1)
    
        loss = self.compute_loss(dr_image, dr_depth, camera_angles)

        # Backward pass
        dr.backward(loss)
        self.opt.step()
        
        # Clip values to valid range
        for k, v in self.opt.items():
            self.opt[k] = dr.clip(v, 0, 1)
        
        self.scene_params.update(self.opt)
        self._step_idx += 1

        return dr_image, loss
    
    @dr.wrap("drjit", "torch")
    def compute_loss(self, image: torch.Tensor, depth: torch.Tensor, camera_angles: list[tuple[float, float]]) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert (H, W, C) to (1, C, H, W)
            depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)
        elif image.ndim == 4:
            # Don't ask about the weird order, it's the batch sensor 
            image = image.permute(1, 3, 0, 2)  # Convert (H, B, W, C) to (B, C, H, W)
            depth = depth.detach().unsqueeze(-1).repeat(1, 1, 1, 3).permute(1, 3, 0, 2)  # Convert to (B, 3, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        prompt_embedings = self.prompt_processor.fetch_prompt(
            camera_angles=camera_angles,
            images=image,
        )

        # Compute loss using Stable Diffusion
        return self.guidance.compute_rdfs_loss(prompt_embedings, image, depth)
    

    def setup_opt_sensors(self):


        if self.camera_cfg.nb_sensors == 1:
            sensor_dict = {
                'type': 'perspective',
                'fov': self.camera_cfg.fov,
                'to_world': mi.ScalarTransform4f(),
            }
        else: 
            sensor_dict = {
                'type': 'batch',
                **{
                    f'sensor_{i}': {
                        'type': 'perspective',
                        'fov': self.camera_cfg.fov,
                        'to_world': mi.ScalarTransform4f()
                    } for i in range(self.camera_cfg.nb_sensors)
                },
            }


        sensor_dict.update({ 
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },

            'film': {
                'type': 'hdrfilm',
                'width' : self.camera_cfg.nb_sensors * self.camera_cfg.render_size,
                'height': self.camera_cfg.render_size,
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

        print(f"Using {self.camera_cfg.nb_sensors} optimization sensors with render size {self.camera_cfg.render_size}x{self.camera_cfg.render_size} and FOV {self.camera_cfg.fov} degrees.")

    def random_transform(self, sensor_idx: int) -> tuple[mi.ScalarTransform4f, (float, float)]:
        if random.random() < 0.5:
                elevation_deg = self.generator.uniform(mi.ScalarFloat, shape=1, 
                                                 low=self.camera_cfg.elevation_min, high=self.camera_cfg.elevation_max)
                elevation = dr.deg2rad(elevation_deg)
        else:
            elev_percent_low = (self.camera_cfg.elevation_min + 90.0) / 180.0
            elev_percent_high = (self.camera_cfg.elevation_max + 90.0) / 180.0
            u = self.generator.uniform(mi.ScalarFloat, shape=1, low=elev_percent_low, high=elev_percent_high)
            elevation = dr.asin(2.0 * u - 1.0)
        
        elevation = 0.5 * dr.pi - elevation  # Convert to inclination

        if True:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth = (
                sensor_idx + self.generator.uniform(mi.ScalarFloat, shape=1)
            ) / self.camera_cfg.nb_sensors * dr.two_pi

        else:
            # simple random sampling
            azimuth = self.generator.uniform(mi.ScalarFloat, shape=1, low=0.0, high=dr.two_pi)

        camera_distances = self.generator.uniform(
            mi.ScalarFloat, shape=1, 
            low=self.camera_cfg.radius_min, high=self.camera_cfg.radius_max
        )

        camera_perturb = self.camera_cfg.camera_perturb

        st, ct = dr.sincos(elevation)
        sp, cp = dr.sincos(azimuth)
        camera_positions = camera_distances * mi.ScalarVector3f(
            sp * st, ct, -cp * st
        )
        camera_positions += self.generator.uniform(mi.ScalarVector3f, shape=(3,), 
                                            low=-camera_perturb, high=camera_perturb)

        center_perturb = self.camera_cfg.center_perturb
        center = mi.ScalarPoint3f(self.camera_cfg.target) + self.generator.normal(mi.ScalarVector3f, shape=(3,), scale=center_perturb)

        up_perturb = self.camera_cfg.up_perturb
        up = self.generator.normal(mi.ScalarVector3f, shape=(3,), loc=mi.ScalarVector3f(0.0, 1.0, 0.0), scale=up_perturb)

        return mi.ScalarTransform4f.look_at(camera_positions, center, up), (azimuth, elevation)
    
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

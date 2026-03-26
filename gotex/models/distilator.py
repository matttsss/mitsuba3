import torch

from dataclasses import dataclass
from gotex.config import Configurable

class Distilator(Configurable):

    @dataclass
    class Config(Configurable.Config):
        min_time: float = 0.0
        max_time: float = 1.0
        guidance_scale: float = 7.5
        enable_offload: bool = False


    cfg: Config

    def __init__(self, config: dict, generator: torch.Generator, device: torch.device):
        super().__init__(config)
        self.generator = generator
        self.device = device

    
    def encode_image(self, image: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
    
    def decode_latents(self, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def predict_velocity(self, prompt_embeds: torch.FloatTensor, latents: torch.FloatTensor, depth: torch.FloatTensor, timestep: float) -> torch.FloatTensor:
        raise NotImplementedError
    
    def set_min_max_time(self, min_time: float, max_time: float):
        self.cfg.min_time = min_time
        self.cfg.max_time = max_time

    @torch.no_grad()
    def dump_latents(self, old_latents: torch.FloatTensor, latents: torch.FloatTensor):
        import mitsuba as mi
        img = self.decode_latents(latents).permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
        old_img = self.decode_latents(old_latents).permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
        
        for i in range(img.shape[0]):
            mi.util.write_bitmap("outputs/latents_{:04d}.png".format(i), img[i])
            mi.util.write_bitmap("outputs/old_img_{:04d}.png".format(i), old_img[i])

    def compute_rdfs_loss(self, prompt_embeds: torch.FloatTensor, image: torch.FloatTensor, depth: torch.FloatTensor):
        latents = self.encode_image(image).float()

        with torch.no_grad():
            time = torch.rand(1, generator=self.generator, device=self.device) 
            time = time * (self.cfg.max_time - self.cfg.min_time) + self.cfg.min_time
            
            noise = torch.randn_like(latents, generator=self.generator, device=self.device)
            latents_noisy = time * noise + (1.0 - time) * latents

            predicted_vel = self.predict_velocity(
                prompt_embeds, latents_noisy, depth, timestep=time * self.num_train_steps
            )
            predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        return torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")

    def compute_rdfs_loss_substeps(
        self,
        prompt_embeds: torch.FloatTensor,
        image: torch.FloatTensor,
        depth: torch.FloatTensor,
        nb_sub_steps: int,
        dump_intermediate: bool = False
    ):
        if nb_sub_steps < 1:
            raise ValueError(f"nb_sub_steps must be >= 1, got {nb_sub_steps}")

        original_latents = self.encode_image(image).float()
        latents = original_latents.clone()

        with torch.no_grad():
            for _ in range(nb_sub_steps):
                time = torch.rand(1, generator=self.generator, device=self.device)
                time = time * (self.cfg.max_time - self.cfg.min_time) + self.cfg.min_time

                noise = torch.randn_like(latents, generator=self.generator, device=self.device)

                latents_noisy = time * noise + (1.0 - time) * latents

                pred_vel = self.predict_velocity(
                    prompt_embeds, latents_noisy, depth, timestep=time * self.num_train_steps
                )

                analytic_vel = noise - latents

                # Gradient descent step on the latents
                latents = latents - 1e-2 * (pred_vel - analytic_vel)

        if dump_intermediate:
            self.dump_latents(original_latents, latents)

        return torch.nn.functional.mse_loss(original_latents, latents, reduction="mean")
    
    def compute_rdfs_rev_loss(self, prompt_embeds: torch.Tensor, image: torch.Tensor, depth: torch.Tensor):
        latents = self.encode_image(image).float()

        with torch.no_grad():
            time = torch.rand(1, generator=self.generator, device=self.device)
            time = time * (self.cfg.max_time - self.cfg.min_time) + self.cfg.min_time

            # Sample noise and do a one step optimization of the noise
            noise = torch.randn_like(latents, generator=self.generator, device=self.device)
            latents_noisy = time * noise + (1.0 - time) * latents

            predicted_vel = self.predict_velocity(
                prompt_embeds, latents_noisy, depth, timestep=time * self.num_train_steps
            )
            noise = noise + (1 - time) * (predicted_vel + latents - noise)

            # Back to the original RDFS loss with the optimized noise
            latents_noisy = time * noise + (1.0 - time) * latents
            predicted_vel = self.predict_velocity(
                prompt_embeds, latents_noisy, depth, timestep=time * self.num_train_steps
            )
            predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        return torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")

    def prepare_latents(self, render_size):
        shape = (
            1, self.model.config.in_channels,
            int(render_size // self.vae_scale_factor),
            int(render_size // self.vae_scale_factor),
        )

        return torch.randn(shape, generator=self.generator, device=self.device, dtype=self.model.dtype)

import torch

import drjit as dr


class Distilator:

    def __init__(self, generator: torch.Generator, device: torch.device, config):
        self.generator = generator
        self.device = device
        self.config = config

    
    def encode_image(self, image: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError
    
    def decode_latents(self, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def predict_velocity(self, latents: torch.FloatTensor, depth: torch.FloatTensor, timestep: float) -> torch.FloatTensor:
        raise NotImplementedError
    
    def set_min_max_time(self, min_time: float, max_time: float):
        self.config["min_time"] = min_time
        self.config["max_time"] = max_time
    
    @dr.wrap(source='drjit', target='torch')
    def compute_rdfs_loss(self, image: torch.FloatTensor, depth: torch.FloatTensor):

        if image.ndim == 3:
            image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert (H, W, C) to (1, C, H, W)
            depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)
        elif image.ndim == 4:
            # Don't ask about the weird order, it's the batch sensor 
            image = image.permute(1, 3, 0, 2)  # Convert (H, B, W, C) to (B, C, H, W)
            depth = depth.detach().unsqueeze(-1).repeat(1, 1, 1, 3).permute(1, 3, 0, 2)  # Convert to (B, 3, H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        latents = self.encode_image(image).float()
        return self.compute_rdfs_loss_torch(latents, depth)
    
    def compute_rdfs_loss_torch(self, latents: torch.FloatTensor, depth: torch.FloatTensor):
        with torch.no_grad():
            time = torch.rand(1, generator=self.generator, device=self.device) 
            time = time * (self.config["max_time"] - self.config["min_time"]) + self.config["min_time"]
            
            noise = torch.randn_like(latents, generator=self.generator, device=self.device)
            latents_noisy = time * noise + (1.0 - time) * latents

            predicted_vel = self.predict_velocity(
                latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
            )
            predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        return torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")

    @dr.wrap(source='drjit', target='torch')
    def compute_rdfs_rev_loss(self, image: torch.Tensor, depth: torch.Tensor):
        image = image.unsqueeze(0).permute(0, 3, 1, 2)  # Convert to (1, C, H, W)
        depth = depth.detach().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # Convert to (1, 3, H, W)

        latents = self.encode_image(image).float()
        return self.compute_rdfs_rev_loss_torch(latents, depth)

    def compute_rdfs_rev_loss_torch(self, latents: torch.Tensor, depth: torch.Tensor):
        with torch.no_grad():
            time = torch.rand(1, generator=self.generator, device=self.device) 
            time = time * (self.config["max_time"] - self.config["min_time"]) + self.config["min_time"]

            # Sample noise and do a one step optimization of the noise
            noise = torch.randn_like(latents, generator=self.generator, device=self.device)
            latents_noisy = time * noise + (1.0 - time) * latents

            predicted_vel = self.predict_velocity(
                latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
            )
            noise = noise + (1 - time) * (predicted_vel + latents - noise)

            # Back to the original RDFS loss with the optimized noise
            latents_noisy = time * noise + (1.0 - time) * latents
            predicted_vel = self.predict_velocity(
                latents_noisy, depth, timestep=time * self.pipe.scheduler.config.num_train_timesteps
            )
            predicted_vel = torch.nan_to_num(predicted_vel).float()  # Cast to float32 for loss computation

        return torch.nn.functional.mse_loss(noise - latents, predicted_vel, reduction="mean")
    

    def prepare_latents(self, render_size):
        shape = (
            1,
            self.pipe.model.config.in_channels,
            int(render_size // self.pipe.vae_scale_factor),
            int(render_size // self.pipe.vae_scale_factor),
        )

        return torch.randn(shape, generator=self.generator, device=self.device, dtype=self.pipe.model.dtype)

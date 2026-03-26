from dataclasses import dataclass

import torch
from diffusers import StableDiffusion3ControlNetPipeline

from gotex.config import Configurable, RuntimeContext

class PromptEncoder(Configurable):

    @dataclass
    class Config(Configurable.Config):
        prompt: str
        negative_prompt: str = ""

        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
        use_directional_prompts: bool = True

    """Prompt encoder that computes CLIP+T5 embeddings and caches combined prompt encodings."""
    def __init__(
        self,
        config: dict,
        runtime: RuntimeContext,
    ):
        super().__init__(config, runtime=runtime)

        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            vae=None, transformer=None, controlnet=None, torch_dtype=torch.float16
        ).to(self.runtime.device)
        self.pipe.enable_sequential_cpu_offload()

        self.directions = (', side view', ', front view', ', back view', ', overhead view', '')
        prompts = [
                f"{self.cfg.prompt}{direction}" for direction in self.directions
        ] + [self.cfg.negative_prompt]


        self.cached_prompt_encodings: dict[str, dict[str, torch.Tensor]] = {}
        prompt_embeds, pooled_prompt_embeds = self.compute_text_prompt_encodings(prompts, device=self.runtime.device)
        
        for idx, prompt in enumerate(prompts):
            self.cached_prompt_encodings[prompt] = {
                "prompt_embeds": prompt_embeds[idx : idx + 1],
                "pooled_prompt_embeds": pooled_prompt_embeds[idx : idx + 1],
            }

    @torch.no_grad()
    def compute_text_prompt_encodings(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ):
        device = device or self.runtime.device

        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompts, prompts, prompts, do_classifier_free_guidance=False, device=device)
        
        return prompt_embeds, pooled_prompt_embeds

    def _lookup_cached_prompt_encodings(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.runtime.device
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        missing = [p for p in prompt_list if p not in self.cached_prompt_encodings]
        if missing:
            raise ValueError(
                "Missing cached text prompt encodings for prompts: "
                f"{missing}. Provide these prompts at constructor time in `t5_precompute_prompts`."
            )

        prompt_embeds = torch.cat(
            [self.cached_prompt_encodings[p]["prompt_embeds"].to(device) for p in prompt_list]
        )
        pooled_prompt_embeds = torch.cat(
            [self.cached_prompt_encodings[p]["pooled_prompt_embeds"].to(device) for p in prompt_list]
        )

        return prompt_embeds, pooled_prompt_embeds

    def angles_to_prompt(self, azimuth: float, inclination: float) -> str:
        # Prioritize top-down views, then map azimuth to front/side/back buckets.
        direction = ', side view'
        if inclination <= torch.pi / 4:
            direction = ', overhead view'
        elif -torch.pi / 4 <= azimuth < torch.pi / 4:
            direction = ', front view'
        elif azimuth >= 3 * torch.pi / 4 or azimuth < -3 * torch.pi / 4:
            direction = ', back view'
        
        return f"{self.cfg['prompt']}{direction}"
    
    @torch.no_grad()
    def fetch_prompt(
        self,
        camera_angles: list[tuple[float, float]] | None = None ,
        images: torch.Tensor | None = None,
        device: torch.device | None = "cuda",
        do_classifier_free_guidance: bool = True,
    ):
        device = device or self.runtime.device

        if camera_angles is None:
            prompt_embeds, pooled_prompt_embeds = self._lookup_cached_prompt_encodings(
                prompt=self.cfg["prompt"],
                device=device,
            )
        else:
            prompts = [self.angles_to_prompt(az, inc) for az, inc in camera_angles]
            prompt_embeds, pooled_prompt_embeds = self._lookup_cached_prompt_encodings(prompts, device=device)

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_classifier_free_guidance:
            batch_size = len(camera_angles) if camera_angles is not None else 1
            negative_prompt_embeds, negative_pooled_prompt_embeds = self._lookup_cached_prompt_encodings(
                self.cfg["negative_prompt"],
                device=device,
            )
            negative_prompt_embeds = negative_prompt_embeds.expand(batch_size, -1, -1)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.expand(batch_size, -1)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
        }

from dataclasses import dataclass

import torch
from diffusers import StableDiffusion3ControlNetPipeline

import gotex
from gotex.config import RuntimeContext
from gotex.models.prompt_encoder import PromptEncoder

@gotex.register("sd3_prompt_encoder")
class SD3PromptEncoder(PromptEncoder):

    @dataclass
    class Config(PromptEncoder.Config):
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"

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

    @torch.no_grad()
    def compute_text_prompt_encodings(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ) -> PromptEncoder.PromptEncoding:
        device = device or self.runtime.device

        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompts, prompts, prompts, do_classifier_free_guidance=False, device=device)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    
    def format_prompt_output(
        self,
        prompt_encoding: PromptEncoder.PromptEncoding,
        negative_prompt_encoding: PromptEncoder.PromptEncoding | None,
    ) -> dict[str, torch.Tensor | None]:
        return {
            "prompt_embeds": prompt_encoding["prompt_embeds"],
            "negative_prompt_embeds": None if negative_prompt_encoding is None else negative_prompt_encoding["prompt_embeds"],
            "pooled_prompt_embeds": prompt_encoding["pooled_prompt_embeds"],
            "negative_pooled_prompt_embeds": None if negative_prompt_encoding is None else negative_prompt_encoding["pooled_prompt_embeds"],
        }

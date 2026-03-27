from dataclasses import dataclass

import torch
from diffusers import StableDiffusionControlNetPipeline

import gotex
from gotex.config import RuntimeContext
from gotex.models.prompt_encoder import PromptEncoder

@gotex.register("instaflow_prompt_encoder")
class InstaflowPromptEncoder(PromptEncoder):

    @dataclass
    class Config(PromptEncoder.Config):
        pretrained_model_name_or_path: str = "XCLiu/2_rectified_flow_from_sd_1_5"

    def __init__(
        self,
        config: dict,
        runtime: RuntimeContext,
    ):
        super().__init__(config, runtime=runtime)

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            vae=None, unet=None, controlnet=None, torch_dtype=torch.float16
        ).to(self.runtime.device)
        self.pipe.enable_sequential_cpu_offload()

    @torch.no_grad()
    def compute_text_prompt_encodings(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ) -> PromptEncoder.PromptEncoding:
        device = device or self.runtime.device

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompts, device, num_images_per_prompt=1, do_classifier_free_guidance=False)

        return {
            "prompt_embeds": prompt_embeds,
        }
    
    def format_prompt_output(
        self,
        prompt_encoding: PromptEncoder.PromptEncoding,
        negative_prompt_encoding: PromptEncoder.PromptEncoding | None,
    ) -> dict[str, torch.Tensor | None]:
        return {
            "prompt_embeds": prompt_encoding["prompt_embeds"],
            "negative_prompt_embeds": None if negative_prompt_encoding is None else negative_prompt_encoding["prompt_embeds"]
        }

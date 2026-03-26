import torch

from diffusers import StableDiffusion3ControlNetPipeline

class PromptEncoder(torch.nn.Module):
    """Prompt encoder that computes CLIP+T5 embeddings and caches combined prompt encodings."""

    def __init__(self, prompts: list[str] | None, device: torch.device, dtype: torch.dtype):
        super().__init__()
    
        self.device = device
        self.dtype = dtype

        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            vae=None, transformer=None, controlnet=None, torch_dtype=torch.float16
        ).to(device)

        self.pipe.enable_sequential_cpu_offload()

        self.cached_prompt_encodings: dict[str, dict[str, torch.Tensor]] = {}
        if prompts is not None:
            prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
            prompt_embeds, pooled_prompt_embeds = self.compute_text_prompt_encodings(prompt_list, device=device)
            
            for idx, prompt in enumerate(prompt_list):
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
        device = device or self.device

        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompts, prompts, prompts, do_classifier_free_guidance=False, device=device)
        
        return prompt_embeds, pooled_prompt_embeds

    def _lookup_cached_prompt_encodings(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.device
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

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        images: torch.Tensor | None = None,
        device: torch.device | None = "cuda",
        do_classifier_free_guidance: bool = True,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, pooled_prompt_embeds = self._lookup_cached_prompt_encodings(
            prompt=prompt,
            device=device,
        )

        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_pooled_prompt_embeds = self._lookup_cached_prompt_encodings(
                negative_prompt,
                device=device,
            )

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
        }

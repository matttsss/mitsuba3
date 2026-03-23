import warnings
import gc

import torch

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

class PromptEncoder(torch.nn.Module):
    """Prompt encoder that computes CLIP+T5 embeddings and caches combined prompt encodings."""

    def __init__(self, prompts: list[str] | None, device: torch.device, dtype: torch.dtype):
        super().__init__()
    
        self.device = device
        self.dtype = dtype

        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=dtype, use_safetensors=False
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=dtype, use_safetensors=False
        )
        self.text_encoder_3 = T5EncoderModel.from_pretrained(
            "google/t5-v1_1-xxl", torch_dtype=dtype
        )


        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=dtype)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", torch_dtype=dtype)

        # T5's max input length
        self.max_sequence_length = 256
        # CLIP's max input length (same for both models)
        self.tokenizer_max_length = self.tokenizer.model_max_length if self.tokenizer is not None else 77

        self.cached_prompt_encodings: dict[str, dict[str, torch.Tensor]] = {}
        if prompts is not None:
            prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
            prompt_embeds, pooled_prompt_embeds = self.compute_text_prompt_encodings(prompt_list, device=device)
            
            for idx, prompt in enumerate(prompt_list):
                self.cached_prompt_encodings[prompt] = {
                    "prompt_embeds": prompt_embeds[idx : idx + 1],
                    "pooled_prompt_embeds": pooled_prompt_embeds[idx : idx + 1],
                }

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        self.text_encoder_3.to(device=device)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.max_sequence_length - 1 : -1])
            warnings.warn(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {self.max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        self.text_encoder_3.to("cpu")
        return prompt_embeds

    @torch.no_grad()
    def compute_text_prompt_encodings(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ):
        """Precompute and cache combined CLIP+T5 encodings for prompt lookup."""
        device = device or self.device

        prompts = [prompts] if isinstance(prompts, str) else list(prompts)

        self.text_encoder.to(device=device)
        self.text_encoder_2.to(device=device)
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompts,
            device=device,
            clip_skip=None,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompts,
            device=device,
            clip_skip=None,
            clip_model_index=1,
        )
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")

        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self._get_t5_prompt_embeds(prompt=prompts, device=device)

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

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
            [self.cached_prompt_encodings[p]["prompt_embeds"].to(device) for p in prompt_list],
            dim=0,
        )
        pooled_prompt_embeds = torch.cat(
            [self.cached_prompt_encodings[p]["pooled_prompt_embeds"].to(device) for p in prompt_list],
            dim=0,
        )

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        clip_skip: int | None = None,
        clip_model_index: int = 0,
    ):
        device = device or self.device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            warnings.warn(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size, -1)

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
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

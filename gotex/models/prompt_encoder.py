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
    """Base prompt encoder that precomputes and caches T5 prompt embeddings.

    This class is responsible only for T5 embeddings. It can precompute embeddings
    for a set of prompts and free the T5 encoder afterwards to save memory.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        """Initialize the base encoder and load the T5 components."""
        super().__init__()
        self.text_encoder_3 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl", torch_dtype=dtype).to(device)
        self.tokenizer_3 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl", torch_dtype=dtype)

        self.max_sequence_length = 256 # T5's maximum input length; can be adjusted if needed

        self.device = device
        self.dtype = dtype
        self.t5_prompt_embeds: dict[str, torch.Tensor] = {}

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            raise RuntimeError(
                "T5 encoder is not available. Call `precompute_t5_prompt_embeds` before freeing it, "
                "or instantiate a new encoder instance."
            )

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

        return prompt_embeds

    @torch.no_grad()
    def precompute_t5_prompt_embeds(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ):
        """Precompute and cache T5 embeddings, then free T5 model memory."""
        device = device or self.device

        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompt_list:
            return

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt_list,
            device=device,
        )

        for idx, prompt in enumerate(prompt_list):
            self.t5_prompt_embeds[prompt] = prompt_embeds[idx : idx + 1]

        # T5 is only used for precompute; release it to reduce memory pressure.
        del self.text_encoder_3
        del self.tokenizer_3
        self.text_encoder_3 = None
        self.tokenizer_3 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
    ):
        """Return cached T5 embeddings for one prompt or a batch of prompts."""
        device = device or self.device
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        missing = [p for p in prompt_list if p not in self.t5_prompt_embeds]
        if missing:
            raise ValueError(
                "Missing precomputed T5 embeddings for prompts: "
                f"{missing}. Call `precompute_t5_prompt_embeds` with these prompts first."
            )

        embeds = [
            self.t5_prompt_embeds[p].to(device)
            for p in prompt_list
        ]
        return torch.cat(embeds, dim=0)


class TextPromptEncoder(PromptEncoder):
    """Prompt encoder that computes CLIP embeddings and combines them with cached T5 embeddings."""

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=dtype, use_safetensors=False
        ).to(device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=dtype, use_safetensors=False
        ).to(device)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=dtype
        )
        self.tokenizer_max_length = self.tokenizer.model_max_length if self.tokenizer is not None else 77

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
        images: torch.Tensor,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        device: torch.device | None = "cuda",
        do_classifier_free_guidance: bool = True,
        clip_skip: int | None = None,
    ):
        device = device or self.device

        # # set lora scale so that monkey patched LoRA
        # # function of text encoder can correctly access it
        # if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
        #     self._lora_scale = lora_scale

        #     # dynamically adjust the LoRA scale
        #     if self.text_encoder is not None and USE_PEFT_BACKEND:
        #         scale_lora_layers(self.text_encoder, lora_scale)
        #     if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
        #         scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            clip_skip=clip_skip,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            clip_skip=clip_skip,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self.get_t5_prompt_embeds(
            prompt=prompt,
            device=device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

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

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self.get_t5_prompt_embeds(
                prompt=negative_prompt,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        # if self.text_encoder is not None:
        #     if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder, lora_scale)

        # if self.text_encoder_2 is not None:
        #     if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
        #         # Retrieve the original scale by scaling back the LoRA layers
        #         unscale_lora_layers(self.text_encoder_2, lora_scale)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
        }

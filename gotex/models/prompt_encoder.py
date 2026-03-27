from dataclasses import dataclass

import torch

from gotex.config import Configurable

class PromptEncoder(Configurable):

    @dataclass
    class Config(Configurable.Config):
        prompt: str
        negative_prompt: str = ""

        use_directional_prompts: bool = True

    PromptEncoding = dict[str, torch.Tensor]

    def prepare_encodings(self):
        directions = (', side view', ', front view', ', back view', ', overhead view') \
                      if self.cfg.use_directional_prompts else ('',)

        prompts = [
                f"{self.cfg.prompt}{direction}" for direction in directions
        ] + [self.cfg.negative_prompt]

        self.cached_prompt_encodings: dict[str, PromptEncoder.PromptEncoding] = {}
        batched_encodings = self.compute_text_prompt_encodings(prompts, device=self.runtime.device)

        for prompt, encoding in zip(prompts, self.split_prompt_encodings(batched_encodings)):
            self.cached_prompt_encodings[prompt] = encoding

    def compute_text_prompt_encodings(
        self,
        prompts: str | list[str],
        device: torch.device | None = None,
    ) -> PromptEncoding:
        raise NotImplementedError("Subclasses must implement compute_text_prompt_encodings to compute prompt encodings for a list of prompts.")  

    def format_prompt_output(
        self,
        prompt_encoding: PromptEncoding,
        negative_prompt_encoding: PromptEncoding | None,
    ) -> dict[str, torch.Tensor | None]:
        raise NotImplementedError("Subclasses must implement format_prompt_output to specify the output format of the prompt encodings used by the model.")

    @torch.no_grad()
    def fetch_prompt(
        self,
        camera_angles: list[tuple[float, float]] | None = None ,
        images: torch.Tensor | None = None,
        device: torch.device | None = "cuda",
        do_classifier_free_guidance: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        device = device or self.runtime.device

        if camera_angles is None:
            prompt_encoding = self._lookup_cached_prompt_encodings(
                prompt=self.cfg.prompt,
                device=device,
            )
        else:
            prompts = [self.angles_to_prompt(az, inc) for az, inc in camera_angles]
            prompt_encoding = self._lookup_cached_prompt_encodings(prompts, device=device)

        negative_prompt_encoding = None
        if do_classifier_free_guidance:
            batch_size = len(camera_angles) if camera_angles is not None else 1
            negative_prompt_encoding = self._lookup_cached_prompt_encodings(
                self.cfg.negative_prompt,
                device=device,
            )
            negative_prompt_encoding = self.expand_prompt_encoding_batch(
                negative_prompt_encoding,
                batch_size=batch_size,
            )

        return self.format_prompt_output(
            prompt_encoding=prompt_encoding,
            negative_prompt_encoding=negative_prompt_encoding,
        )
    
    def angles_to_prompt(self, azimuth: float, inclination: float) -> str:
        if not self.cfg.use_directional_prompts:
            return self.cfg.prompt

        # Prioritize top-down views, then map azimuth to front/side/back buckets.
        direction = ', side view'
        if inclination <= torch.pi / 4:
            direction = ', overhead view'
        elif -torch.pi / 4 <= azimuth < torch.pi / 4:
            direction = ', front view'
        elif azimuth >= 3 * torch.pi / 4 or azimuth < -3 * torch.pi / 4:
            direction = ', back view'

        return f"{self.cfg.prompt}{direction}"

    def split_prompt_encodings(
        self,
        batched_encodings: PromptEncoding,
    ) -> list[PromptEncoding]:
        if not batched_encodings:
            return []

        batch_size = next(iter(batched_encodings.values())).shape[0]
        return [
            {
                key: tensor[idx : idx + 1]
                for key, tensor in batched_encodings.items()
            }
            for idx in range(batch_size)
        ]

    def merge_prompt_encodings(
        self,
        encodings: list[PromptEncoding],
        device: torch.device,
    ) -> PromptEncoding:
        if not encodings:
            raise ValueError("Cannot merge an empty list of prompt encodings.")

        keys = encodings[0].keys()
        return {
            key: torch.cat([encoding[key] for encoding in encodings]).to(device)
            for key in keys
        }

    def expand_prompt_encoding_batch(
        self,
        encoding: PromptEncoding,
        batch_size: int,
    ) -> PromptEncoding:
        return {
            key: tensor.expand(batch_size, *([-1] * (tensor.ndim - 1)))
            for key, tensor in encoding.items()
        }

    def _lookup_cached_prompt_encodings(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
    ) -> PromptEncoding:
        device = device or self.runtime.device
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        missing = [p for p in prompt_list if p not in self.cached_prompt_encodings]
        if missing:
            raise ValueError(
                "Missing cached text prompt encodings for prompts: "
                f"{missing}. Add them to the prompt set built during encoder initialization."
            )

        return self.merge_prompt_encodings(
            [self.cached_prompt_encodings[p] for p in prompt_list],
            device=device,
        )

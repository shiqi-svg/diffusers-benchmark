"""Qwen-Image — text-to-image (3-stage: text → transformer → VAE)."""

from __future__ import annotations

import logging

import torch

from benchmark.config import MODEL_DEFAULTS
from benchmark.models import register_model
from benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


@register_model("Qwen-Image")
class QwenImageModel(BaseModel):
    model_name = "Qwen-Image"
    model_type = "text-to-image"
    model_id = "Qwen/Qwen-Image"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["Qwen-Image"]["guidance_scale"]
    default_resolution = {"height": 1024, "width": 1024}

    def load(self) -> None:
        from diffusers import QwenImagePipeline

        logger.info("Loading %s from %s …", self.model_name, self.model_id)
        self.pipe = QwenImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype(),
        )
        self.pipe.to("cuda")

    def infer(self) -> None:
        defaults = MODEL_DEFAULTS["Qwen-Image"]
        self.pipe(
            prompt=defaults["prompt"],
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=defaults["guidance_scale"],
            true_cfg_scale=defaults["true_cfg_scale"],
            num_images_per_prompt=defaults["num_images_per_prompt"],
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

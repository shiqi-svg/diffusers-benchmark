"""FLUX.1-dev — text-to-image."""

from __future__ import annotations

import logging

import torch

from benchmark.config import MODEL_DEFAULTS
from benchmark.models import register_model
from benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


@register_model("FLUX.1-dev")
class FluxModel(BaseModel):
    model_name = "FLUX.1-dev"
    model_type = "text-to-image"
    model_id = "black-forest-labs/FLUX.1-dev"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["FLUX.1-dev"]["guidance_scale"]
    default_resolution = {"height": 1024, "width": 1024}

    def load(self) -> None:
        from diffusers import FluxPipeline

        logger.info("Loading %s from %s …", self.model_name, self.model_id)
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype(),
        )
        self.pipe.to("cuda")

    def infer(self) -> None:
        defaults = MODEL_DEFAULTS["FLUX.1-dev"]
        self.pipe(
            prompt=defaults["prompt"],
            height=self.height,
            width=self.width,
            num_inference_steps=self.steps,
            guidance_scale=defaults["guidance_scale"],
            num_images_per_prompt=defaults["num_images_per_prompt"],
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

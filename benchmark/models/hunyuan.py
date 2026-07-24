"""HunyuanVideo — text-to-video (3-stage: clip → llama → transformer)."""

from __future__ import annotations

import logging

import torch

from benchmark.config import MODEL_DEFAULTS
from benchmark.models import register_model
from benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


@register_model("HunyuanVideo")
class HunyuanModel(BaseModel):
    model_name = "HunyuanVideo"
    model_type = "text-to-video"
    model_id = "hunyuanvideo-community/HunyuanVideo"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["HunyuanVideo"]["guidance_scale"]
    default_resolution = {"height": 320, "width": 512, "num_frames": 61}

    def load(self) -> None:
        from diffusers import HunyuanVideoPipeline

        logger.info("Loading %s from %s …", self.model_name, self.model_id)
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype(),
        )
        self.pipe.to("cuda")

    def infer(self) -> None:
        defaults = MODEL_DEFAULTS["HunyuanVideo"]
        self.pipe(
            prompt=defaults["prompt"],
            height=self.height,
            width=self.width,
            num_frames=defaults["num_frames"],
            num_inference_steps=self.steps,
            guidance_scale=defaults["guidance_scale"],
            num_videos_per_prompt=1,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

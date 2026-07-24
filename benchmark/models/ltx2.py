"""LTX-2 — text-to-video (single-process, exports .mp4)."""

from __future__ import annotations

import logging

import torch

from benchmark.config import MODEL_DEFAULTS
from benchmark.models import register_model
from benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


@register_model("LTX-2")
class LTX2Model(BaseModel):
    model_name = "LTX-2"
    model_type = "text-to-video"
    model_id = "Lightricks/LTX-2"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["LTX-2"]["guidance_scale"]
    default_resolution = {"height": 512, "width": 768, "num_frames": 121}

    def load(self) -> None:
        from diffusers import LTX2Pipeline

        logger.info("Loading %s from %s …", self.model_name, self.model_id)
        self.pipe = LTX2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype(),
        )
        self.pipe.to("cuda")

    def infer(self) -> None:
        defaults = MODEL_DEFAULTS["LTX-2"]
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

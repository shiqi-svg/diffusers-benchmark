"""Wan 2.1 & 2.2 — text-to-video (2-stage: transformer → VAE)."""

from __future__ import annotations

import logging

import torch

from benchmark.config import MODEL_DEFAULTS
from benchmark.models import register_model
from benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


class BaseWanModel(BaseModel):
    """Shared logic for Wan 2.1 and Wan 2.2."""

    model_type = "text-to-video"

    # Subclasses override these three:
    model_name: str = ""
    model_id: str = ""
    default_resolution = {"height": 480, "width": 832, "num_frames": 9}

    def load(self) -> None:
        from diffusers import WanPipeline

        logger.info("Loading %s from %s …", self.model_name, self.model_id)
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype(),
        )
        self.pipe.to("cuda")

    def infer(self) -> None:
        defaults = MODEL_DEFAULTS.get(self.model_name)
        if defaults is None:
            defaults = MODEL_DEFAULTS.get("Wan2.1-T2V", {})
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


@register_model("Wan2.1-T2V")
class Wan21Model(BaseWanModel):
    model_name = "Wan2.1-T2V"
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["Wan2.1-T2V"]["guidance_scale"]
    default_resolution = {"height": 480, "width": 832, "num_frames": 9}


@register_model("Wan2.2-T2V")
class Wan22Model(BaseWanModel):
    model_name = "Wan2.2-T2V"
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    precision_str = "bfloat16"
    default_guidance = MODEL_DEFAULTS["Wan2.2-T2V"]["guidance_scale"]
    default_resolution = {"height": 480, "width": 832, "num_frames": 9}

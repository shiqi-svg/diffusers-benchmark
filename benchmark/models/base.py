"""Abstract base class for all benchmarked models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from benchmark.config import STEPS

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Every model adapter must subclass this.

    Subclasses define *how* to load and run a specific pipeline; the
    framework (BenchmarkRunner) decides *when* and measures *how long*.
    """

    # Set by subclasses  ---------------------------------------------------
    model_name: str = ""  # e.g. "FLUX.1-dev"
    model_type: str = ""  # "text-to-image" | "text-to-video"
    model_id: str = ""  # HuggingFace repo id
    precision_str: str = "bfloat16"  # human-readable, stored in results
    default_guidance: float = 3.5
    # Subclasses MUST override this with their own dict:
    default_resolution: dict[str, int] = {"height": 1024, "width": 1024}

    def __init__(self, height: int, width: int, steps: int = STEPS, **kwargs: Any) -> None:
        self.height = height
        self.width = width
        self.steps = steps
        self.kwargs = kwargs
        self.pipe: Any = None  # set by load()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load weights from HF hub onto GPU.  Timed by the runner."""
        ...

    @abstractmethod
    def infer(self) -> Any:
        """Run a single forward pass.  Timed by the runner (N iterations).

        Returns whatever the pipeline returns (image, video frames, …).
        """
        ...

    def warmup(self) -> None:
        """Single dry-run before timed iterations.  Default: call infer() once."""
        logger.debug("Warming up %s …", self.model_name)
        self.infer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dtype(self) -> torch.dtype:
        """Resolve precision string → torch.dtype."""
        mapping: dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.precision_str not in mapping:
            raise ValueError(
                f"Unknown precision '{self.precision_str}' for {self.model_name}. "
                f"Valid: {list(mapping.keys())}"
            )
        return mapping[self.precision_str]

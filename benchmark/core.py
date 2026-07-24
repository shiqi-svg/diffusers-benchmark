"""Core benchmarking infrastructure: Timer, BenchmarkResult, BenchmarkRunner."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from benchmark.config import DEFAULT_ITERATIONS, OUTPUT_DIR, RESOLUTIONS, STEPS

logger = logging.getLogger(__name__)


class Timer:
    """High-precision GPU-aware timer.

    Usage as context manager::

        with Timer() as t:
            model.load()
        print(t.elapsed)  # seconds, float
    """

    def __enter__(self) -> "Timer":
        torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        torch.cuda.synchronize()
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self._end - self._start


def measure_iteration(model: Any, iterations: int = DEFAULT_ITERATIONS) -> tuple[float, list[float]]:
    """Run *iterations* timed inference calls, return (mean, [individual_times])."""
    times: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.infer()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times), times


def cleanup_gpu() -> None:
    """Release GPU resources between model runs to get clean measurements."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchmarkResult:
    """Single benchmark data point."""

    model_name: str
    model_type: str  # "text-to-image" | "text-to-video"
    resolution: str  # e.g. "1024x1024" or "480x832x9"
    precision: str
    steps: int
    guidance_scale: float
    weight_load_time_s: float
    pure_inference_time_s: float
    total_inference_time_s: float
    gpu_memory_peak_mb: int
    iterations: int
    error: str | None = None

    # Per-iteration detail (optional, for deeper analysis)
    iteration_times: list[float] = field(default_factory=list)


class BenchmarkRunner:
    """Orchestrates warm-up, load timing, inference timing, and GPU-memory tracking
    across a set of model classes and resolutions."""

    def __init__(
        self,
        steps: int = STEPS,
        resolutions: list[int] | None = None,
        iterations: int = DEFAULT_ITERATIONS,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.steps = steps
        self.resolutions = resolutions or RESOLUTIONS
        self.iterations = iterations
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, model_classes: list[type]) -> list[BenchmarkResult]:
        """Run the full benchmark matrix and return all results."""
        results: list[BenchmarkResult] = []
        total = len(model_classes) * len(self.resolutions)

        for m_idx, model_cls in enumerate(model_classes):
            for r_idx, res in enumerate(self.resolutions):
                nth = m_idx * len(self.resolutions) + r_idx + 1
                label = f"[{nth}/{total}]"
                logger.info(
                    "%s Benchmarking %s @ %s ...",
                    label,
                    model_cls.__name__,
                    res,
                )
                result = self._run_one(model_cls, res)
                results.append(result)
                cleanup_gpu()

        return results

    def _run_one(self, model_cls: type, resolution: int) -> BenchmarkResult:
        """Benchmark a single model + resolution combination."""
        # ------------------------------------------------------------------
        # Resolution mapping
        # ------------------------------------------------------------------
        model_type = model_cls.model_type
        if model_type == "text-to-image":
            height = width = resolution
            res_str = f"{resolution}x{resolution}"
        elif model_type == "text-to-video":
            # For video models, height=resolution; width is computed to
            # preserve the model's default aspect ratio.
            defaults = model_cls.default_resolution
            aspect = defaults["width"] / defaults["height"]
            height = resolution
            width = int(round(resolution * aspect))
            # Make dimensions divisible by 16 (required by Wan VAE and others)
            width = (width // 16) * 16
            height = (height // 16) * 16
            num_frames = defaults.get("num_frames", 9)
            res_str = f"{height}x{width}x{num_frames}"
        else:
            height = width = resolution
            res_str = f"{resolution}x{resolution}"

        # ------------------------------------------------------------------
        # Instantiate & load (timed)
        # ------------------------------------------------------------------
        try:
            instance = model_cls(height=height, width=width, steps=self.steps)

            with Timer() as t:
                instance.load()
            weight_load_time = t.elapsed

        except Exception as exc:
            logger.exception("Failed to load %s @ %s", model_cls.__name__, res_str)
            return BenchmarkResult(
                model_name=model_cls.model_name,
                model_type=model_type,
                resolution=res_str,
                precision=model_cls.precision_str,
                steps=self.steps,
                guidance_scale=model_cls.default_guidance,
                weight_load_time_s=0.0,
                pure_inference_time_s=0.0,
                total_inference_time_s=0.0,
                gpu_memory_peak_mb=0,
                iterations=self.iterations,
                error=str(exc),
            )

        # ------------------------------------------------------------------
        # Warm-up (not timed)
        # ------------------------------------------------------------------
        try:
            instance.warmup()
        except Exception as exc:
            logger.exception("Warmup failed for %s @ %s", model_cls.__name__, res_str)
            return BenchmarkResult(
                model_name=model_cls.model_name,
                model_type=model_type,
                resolution=res_str,
                precision=model_cls.precision_str,
                steps=self.steps,
                guidance_scale=model_cls.default_guidance,
                weight_load_time_s=weight_load_time,
                pure_inference_time_s=0.0,
                total_inference_time_s=0.0,
                gpu_memory_peak_mb=0,
                iterations=self.iterations,
                error=f"warmup: {exc}",
            )

        # ------------------------------------------------------------------
        # Timed inference iterations
        # ------------------------------------------------------------------
        try:
            mean_time, iter_times = measure_iteration(instance, self.iterations)
        except Exception as exc:
            logger.exception("Inference failed for %s @ %s", model_cls.__name__, res_str)
            mean_time = 0.0
            iter_times = []
            error = str(exc)
        else:
            error = None

        # ------------------------------------------------------------------
        # GPU memory peak
        # ------------------------------------------------------------------
        peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

        # ------------------------------------------------------------------
        # Clean up this model instance
        # ------------------------------------------------------------------
        del instance
        cleanup_gpu()

        return BenchmarkResult(
            model_name=model_cls.model_name,
            model_type=model_type,
            resolution=res_str,
            precision=model_cls.precision_str,
            steps=self.steps,
            guidance_scale=model_cls.default_guidance,
            weight_load_time_s=round(weight_load_time, 3),
            pure_inference_time_s=round(mean_time, 3),
            total_inference_time_s=round(weight_load_time + mean_time, 3),
            gpu_memory_peak_mb=peak_mb,
            iterations=self.iterations,
            iteration_times=[round(t, 3) for t in iter_times],
            error=error,
        )

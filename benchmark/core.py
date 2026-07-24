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
    """High-precision GPU-aware context manager.

    Usage::

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


def measure_iteration(model: Any, iterations: int) -> tuple[float, list[float]]:
    """Run *iterations* timed inference calls, return (mean, [individual_times])."""
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
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
    """Release GPU resources to get clean measurements for the next run."""
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, model_classes: list[type]) -> list[BenchmarkResult]:
        """Run the full benchmark matrix and return all results.

        Each model is **pre-loaded once** (not timed) to cache weights on disk.
        Then per-resolution load + inference are timed independently.
        """
        results: list[BenchmarkResult] = []
        total = len(model_classes) * len(self.resolutions)

        for m_idx, model_cls in enumerate(model_classes):
            # -- Pre-download / cache weights (not timed) --
            self._preload_model(model_cls)

            for r_idx, res in enumerate(self.resolutions):
                nth = m_idx * len(self.resolutions) + r_idx + 1
                logger.info(
                    "[%d/%d] Benchmarking %s @ %d …",
                    nth, total,
                    model_cls.__name__,
                    res,
                )
                result = self._run_one(model_cls, res)
                results.append(result)
                cleanup_gpu()

        return results

    # ------------------------------------------------------------------
    # Pre-download (not timed)
    # ------------------------------------------------------------------

    def _preload_model(self, model_cls: type) -> None:
        """Load a model once at its smallest target resolution to warm the
        HuggingFace cache.  Not timed — purely an I/O priming step."""
        smallest_res = min(self.resolutions)
        try:
            height, width, _res_str = self._compute_resolution(model_cls, smallest_res)
        except Exception:
            logger.warning("Cannot compute resolution for preload of %s, skipping", model_cls.__name__)
            return

        logger.info("Pre-loading %s (caching weights) …", model_cls.__name__)
        try:
            instance = model_cls(height=height, width=width, steps=self.steps)
            instance.load()
            instance.warmup()
        except Exception:
            logger.warning("Pre-load of %s failed (will retry in timed run)", model_cls.__name__)
        finally:
            del instance
            cleanup_gpu()

    # ------------------------------------------------------------------
    # Per-resolution benchmark
    # ------------------------------------------------------------------

    def _run_one(self, model_cls: type, resolution: int) -> BenchmarkResult:
        """Benchmark a single model + resolution combination."""
        error: str | None = None
        weight_load_time = 0.0
        mean_time = 0.0
        iter_times: list[float] = []

        # -- Resolution mapping (guarded) --
        try:
            height, width, res_str = self._compute_resolution(model_cls, resolution)
        except Exception as exc:
            logger.exception("Resolution mapping failed for %s @ %d", model_cls.__name__, resolution)
            return BenchmarkResult(
                model_name=model_cls.model_name,
                model_type=model_cls.model_type,
                resolution=f"{resolution}x{resolution}",
                precision=model_cls.precision_str,
                steps=self.steps,
                guidance_scale=model_cls.default_guidance,
                weight_load_time_s=0.0,
                pure_inference_time_s=0.0,
                total_inference_time_s=0.0,
                gpu_memory_peak_mb=0,
                iterations=self.iterations,
                error=f"resolution: {exc}",
            )

        # -- Instantiate & load (timed) --
        try:
            instance = model_cls(height=height, width=width, steps=self.steps)
            with Timer() as t:
                instance.load()
            weight_load_time = t.elapsed
        except Exception as exc:
            logger.exception("Failed to load %s @ %s", model_cls.__name__, res_str)
            return BenchmarkResult(
                model_name=model_cls.model_name,
                model_type=model_cls.model_type,
                resolution=res_str,
                precision=model_cls.precision_str,
                steps=self.steps,
                guidance_scale=model_cls.default_guidance,
                weight_load_time_s=0.0,
                pure_inference_time_s=0.0,
                total_inference_time_s=0.0,
                gpu_memory_peak_mb=0,
                iterations=self.iterations,
                error=f"load: {exc}",
            )

        # -- Warm-up (not timed) --
        try:
            instance.warmup()
        except Exception as exc:
            logger.exception("Warmup failed for %s @ %s", model_cls.__name__, res_str)
            del instance
            cleanup_gpu()
            return BenchmarkResult(
                model_name=model_cls.model_name,
                model_type=model_cls.model_type,
                resolution=res_str,
                precision=model_cls.precision_str,
                steps=self.steps,
                guidance_scale=model_cls.default_guidance,
                weight_load_time_s=round(weight_load_time, 3),
                pure_inference_time_s=0.0,
                total_inference_time_s=round(weight_load_time, 3),
                gpu_memory_peak_mb=0,
                iterations=self.iterations,
                error=f"warmup: {exc}",
            )

        # -- Timed inference iterations --
        try:
            mean_time, iter_times = measure_iteration(instance, self.iterations)
        except Exception as exc:
            logger.exception("Inference failed for %s @ %s", model_cls.__name__, res_str)
            error = f"infer: {exc}"
        else:
            error = None

        # -- GPU memory peak (includes load + warmup + inference) --
        peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

        # -- Clean up --
        del instance

        return BenchmarkResult(
            model_name=model_cls.model_name,
            model_type=model_cls.model_type,
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

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_resolution(model_cls: type, resolution: int) -> tuple[int, int, str]:
        """Return (height, width, display_string) for a model+resolution pair."""
        model_type = model_cls.model_type
        defaults = getattr(model_cls, "default_resolution", {"height": resolution, "width": resolution})

        if model_type == "text-to-image":
            height = width = resolution
            res_str = f"{resolution}x{resolution}"
        elif model_type == "text-to-video":
            aspect = defaults["width"] / defaults["height"]
            height = (resolution // 16) * 16  # divisible by 16
            width = (int(round(resolution * aspect)) // 16) * 16
            num_frames = defaults.get("num_frames", 9)
            res_str = f"{height}x{width}x{num_frames}"
        else:
            height = width = resolution
            res_str = f"{resolution}x{resolution}"

        return height, width, res_str
